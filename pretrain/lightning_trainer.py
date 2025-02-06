import os
import re
import torch
import numpy as np
import torch.optim as optim
import MinkowskiEngine as ME
import pytorch_lightning as pl
from pretrain.criterion import NCELoss
from pytorch_lightning.utilities import rank_zero_only
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast


class LightningPretrain(pl.LightningModule):
    def __init__(self, model_points, model_images, config):
        super().__init__()
        self.model_points = model_points
        self.model_images = model_images
        self._config = config
        self.losses = config["losses"]
        self.train_losses = []
        self.val_losses = []
        self.num_matches = config["num_matches"]
        self.batch_size = config["batch_size"]
        self.num_epochs = config["num_epochs"]
        self.superpixel_size = config["superpixel_size"]
        self.epoch = 0
        self.dbound = config['dbound']
        self.downsample_factor = 4
        self.depth_channels = 118
        self.alpha = config['alpha'] if 'alpha' in config.keys() else 0.25
        self.beta = config['beta'] if 'beta' in config.keys() else 1
        self.yita = config['yita'] if 'yita' in config.keys() else 60
        if config["resume_path"] is not None:
            self.epoch = int(
                re.search(r"(?<=epoch=)[0-9]+", config["resume_path"])[0]
            )
        self.criterion = NCELoss(temperature=config["NCE_temperature"])
        self.working_dir = os.path.join(config["working_dir"], config["datetime"])
        if os.environ.get("LOCAL_RANK", 0) == 0:
            os.makedirs(self.working_dir, exist_ok=True)

    def configure_optimizers(self):
        optimizer = optim.SGD(
            list(self.model_points.parameters()) + list(self.model_images.parameters()),
            lr=self._config["lr"],
            momentum=self._config["sgd_momentum"],
            dampening=self._config["sgd_dampening"],
            weight_decay=self._config["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_epochs)
        return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def training_step(self, batch, batch_idx):
        sparse_input = ME.SparseTensor(batch["sinput_F"], batch["sinput_C"])
        output_points, sparse_output_points = self.model_points(sparse_input)
        points_feature = sparse_output_points.F#N,64
        tensor_stride = torch.IntTensor(sparse_output_points.tensor_stride[:-1]).to(sparse_output_points.C.device)
        coords = sparse_output_points.C[:, 1:-1]
        batch_indices = sparse_output_points.C[:, 0]
        coords = coords // tensor_stride +torch.IntTensor([128,128]).to(sparse_output_points.C.device)
        in_range_flags = (
            (coords[:, 0] < 256)
            & (coords[:, 1] < 256 )
            & (coords[:, 0] > 0 )
            & (coords[:, 1] > 0 )
        )
        coords = coords[in_range_flags]
        points_feature = points_feature[in_range_flags]
        batch_indices = batch_indices[in_range_flags]
        
        self.model_images.eval()
        self.model_images.decoder.train()
        self.model_images.decoder_2.train()
        output_images, output_images_bev, depth_pred = self.model_images(batch["input_I"],batch["points"],batch["lidar2egos"],
        batch["lidar2cameras"],batch["lidar2images"],batch["camera2egos"],batch["camera_intrinsicss"],
        batch["lidar_aug_matrixs"],batch["image_aug_matrixs"])

        loss  = 0 
        if 'loss_superpixels_average' in self.losses:
            ipv_loss = self.loss_superpixels_average(batch, output_points.F, output_images)
            loss += ipv_loss
            self.log(
            "ipv_loss", ipv_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
        )
            
        if 'depth_loss' in self.losses:
            depth_loss = self.loss_depth(batch['depth_map'], depth_pred)
            loss += depth_loss
            self.log(
                "depth_loss", depth_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
            )
        
        if 'bev_loss' in self.losses:
            bev_loss = self.loss_bev_average(points_feature, output_images_bev[batch_indices.type(torch.long),:,coords[:,0].type(torch.long),coords[:,1].type(torch.long)])
            loss += bev_loss
            self.log(
                "bev_loss", bev_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
            )
        
        if 'loss' in self.losses:
            ppkt_loss = self.loss(batch, output_points.F, output_images)
            loss += ppkt_loss
            self.log(
                "ppkt_loss", ppkt_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
            )

        losses = [loss]
    
        loss = torch.mean(torch.stack(losses))

        torch.cuda.empty_cache()
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
        )            
        
        self.train_losses.append(loss.detach().cpu())
        return loss

    def loss(self, batch, output_points, output_images):
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        idx = np.random.choice(pairing_points.shape[0], self.num_matches, replace=False)
        k = output_points[pairing_points[idx]]
        m = tuple(pairing_images[idx].T.long())
        q = output_images.permute(0, 2, 3, 1)[m]
        return self.criterion(k, q)
    
    def loss_bev_average(self, output_bev_points, output_bev_images):
        return self.alpha* self.criterion(output_bev_points, output_bev_images)

    def loss_superpixels_average(self, batch, output_points, output_images):
        # compute a superpoints to superpixels loss using superpixels
        torch.cuda.empty_cache()  # This method is extremely memory intensive
        superpixels = batch["superpixels"]
        pairing_images = batch["pairing_images"]
        pairing_points = batch["pairing_points"]

        superpixels = (
            torch.arange(
                0,
                output_images.shape[0] * self.superpixel_size,
                self.superpixel_size,
                device=self.device,
            )[:, None, None] + superpixels
        )
        m = tuple(pairing_images.cpu().T.long())

        superpixels_I = superpixels.flatten()
        idx_P = torch.arange(pairing_points.shape[0], device=superpixels.device)
        total_pixels = superpixels_I.shape[0]
        idx_I = torch.arange(total_pixels, device=superpixels.device)

        with torch.no_grad():
            one_hot_P = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels[m], idx_P
                ), dim=0),
                torch.ones(pairing_points.shape[0], device=superpixels.device),
                (superpixels.shape[0] * self.superpixel_size, pairing_points.shape[0])
            )

            one_hot_I = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels_I, idx_I
                ), dim=0),
                torch.ones(total_pixels, device=superpixels.device),
                (superpixels.shape[0] * self.superpixel_size, total_pixels)
            )

        k = one_hot_P @ output_points[pairing_points]
        k = k / (torch.sparse.sum(one_hot_P, 1).to_dense()[:, None] + 1e-6)
        q = one_hot_I @ output_images.permute(0, 2, 3, 1).flatten(0, 2)
        q = q / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6)
        mask = torch.where(k[:, 0] != 0)
        k = k[mask]
        q = q[mask]

        return self.beta * self.criterion(k, q)

    def loss_depth(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, self.depth_channels)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        
        with autocast(enabled=False):
            depth_loss = (F.binary_cross_entropy(
                depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction='mean',
            ))

        return depth_loss*self.yita


    def training_epoch_end(self, outputs):
        self.epoch += 1
        if self.epoch == self.num_epochs:
            self.save()
        return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        sparse_input = ME.SparseTensor(batch["sinput_F"], batch["sinput_C"])
        output_points,bev_points = self.model_points(sparse_input)
        points_feature = bev_points.F#N,64
        tensor_stride = torch.IntTensor(bev_points.tensor_stride[:-1]).to(bev_points.C.device)
        coords = bev_points.C[:, 1:-1]
        batch_indices = bev_points.C[:, 0]
        coords = coords // tensor_stride + torch.IntTensor([128,128]).to(bev_points.C.device)
        in_range_flags = (
            (coords[:, 0] < 256)
            & (coords[:, 1] < 256 )
            & (coords[:, 0] > 0 )
            & (coords[:, 1] > 0 )
        )
        coords = coords[in_range_flags]
        points_feature = points_feature[in_range_flags]
        batch_indices = batch_indices[in_range_flags]

        self.model_images.eval()

        output_images, bev_images, depth_pred = self.model_images(batch["input_I"],batch["points"],batch["lidar2egos"],
        batch["lidar2cameras"],batch["lidar2images"],batch["camera2egos"],batch["camera_intrinsicss"],
        batch["lidar_aug_matrixs"],batch["image_aug_matrixs"])

        loss  = 0 
        if 'loss_superpixels_average' in self.losses:
            ipv_loss = self.loss_superpixels_average(batch, output_points.F, output_images)
            loss += ipv_loss
            self.log(
            "ipv_loss", ipv_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
        )
            
        if 'depth_loss' in self.losses:
            depth_loss = self.loss_depth(batch['depth_map'], depth_pred)
            loss += depth_loss
            self.log(
                "depth_loss", depth_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
            )
        
        if 'bev_loss' in self.losses:
            bev_loss = self.loss_bev_average(points_feature, bev_images[batch_indices.type(torch.long),:,coords[:,0].type(torch.long),coords[:,1].type(torch.long)])
            loss += bev_loss
            self.log(
                "bev_loss", bev_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
            )
            
        if 'loss' in self.losses:
            ppkt_loss = self.loss(batch, output_points.F, output_images)
            loss += ppkt_loss
            self.log(
            "ppkt_loss", ppkt_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
        )

        losses = [loss]
        loss = torch.mean(torch.stack(losses))
        self.val_losses.append(loss.detach().cpu())

        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size
        )
        return loss

    @rank_zero_only
    def save(self):
        path = os.path.join(self.working_dir, "model.pt")
        torch.save(
            {
                "model_points": self.model_points.state_dict(),
                "model_images": self.model_images.state_dict(),
                "epoch": self.epoch,
                "config": self._config,
            },
            path,
        )
    
    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]#[4, 6, 1, 224, 416])
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, _, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            1,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
        )
        gt_depths = gt_depths.permute(0, 2, 4, 1, 3, 5).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample_factor,
                                   W // self.downsample_factor)

        gt_depths = (gt_depths -
                     (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1).view(
                                  -1, self.depth_channels + 1)[:, 1:]

        return gt_depths.float()
