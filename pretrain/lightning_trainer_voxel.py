import os
import re
import torch
import numpy as np
import torch.optim as optim
import MinkowskiEngine as ME
import pytorch_lightning as pl
from pretrain.criterion import NCELoss
from pytorch_lightning.utilities import rank_zero_only


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
        output_points,sparse_output_points = self.model_points(sparse_input)
        points_feature = sparse_output_points.F#N,64
        # tensor_stride = torch.IntTensor(sparse_output_points.tensor_stride[:]).to(sparse_output_points.C.device)
        # coords = sparse_output_points.C[:, 1:-1]
        tensor_stride = torch.IntTensor(sparse_output_points.tensor_stride[:]).to(sparse_output_points.C.device)
        coords = sparse_output_points.C[:, 1:]
        batch_indices = sparse_output_points.C[:, 0]
        # coords = coords // tensor_stride +torch.IntTensor([128,128]).to(sparse_output_points.C.device)
        coords = coords // tensor_stride +torch.IntTensor([128,128,5]).to(sparse_output_points.C.device)
        in_range_flags = (
            (coords[:, 0] < 256)
            & (coords[:, 1] < 256 )
            & (coords[:, 0] > 0 )
            & (coords[:, 1] > 0 )
            & (coords[:, 1] < 8 )
            & (coords[:, 0] > 0 )
        )
        coords = coords[in_range_flags]
        points_feature = points_feature[in_range_flags]
        batch_indices = batch_indices[in_range_flags]
        
        self.model_images.eval()
        # self.model_images.decoder.train()
        self.model_images.decoder.train()
        self.model_images.decoder_2.train()
        # output_images = self.model_images(batch["input_I"])
        output_images,output_images_bev = self.model_images(batch["input_I"],batch["points"],batch["lidar2egos"],
        batch["lidar2cameras"],batch["lidar2images"],batch["camera2egos"],batch["camera_intrinsicss"],
        batch["lidar_aug_matrixs"],batch["image_aug_matrixs"])
        # bev_loc = output_images_bev.sum(dim = 1)
        # nonzero = torch.nonzero(bev_loc)
        # print('the bev map ----------------\n and this is the bev map\n', nonzero.shape[0] /(bev_loc.shape[0]* bev_loc.shape[1]* bev_loc.shape[2]), nonzero.shape[0])

        del batch["sinput_F"]
        del batch["sinput_C"]
        del sparse_input
        # each loss is applied independtly on each GPU
        losses = [
            getattr(self, loss)(batch,output_points.F,output_images, points_feature, output_images_bev[batch_indices.type(torch.long),:,coords[:,2].type(torch.long),coords[:,0].type(torch.long),coords[:,1].type(torch.long)])
            for loss in self.losses
        ]
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

    def loss_superpixels_average(self, batch, output_points, output_images,bev_points,bev_images):
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
        #最终以superpixels为单位进行特征的蒸馏
        mask = torch.where(k[:, 0] != 0)
        k = k[mask]
        q = q[mask]

        return self.criterion(k, q,bev_points,bev_images)

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
        coords = coords // tensor_stride

        self.model_images.eval()
        # output_images = self.model_images(batch["input_I"])
        output_images,bev_images = self.model_images(batch["input_I"],batch["points"],batch["lidar2egos"],
        batch["lidar2cameras"],batch["lidar2images"],batch["camera2egos"],batch["camera_intrinsicss"],
        batch["lidar_aug_matrixs"],batch["image_aug_matrixs"])
        #self.model_images(batch["input_I"],batch["points"],batch["lidar2egos"],batch["lidar2cameras"],batch["lidar2images"],batch["camera2egos"],batch["camera_intrinsicss"])


        losses = [
            getattr(self, loss)(batch,output_points.F,output_images, points_feature, bev_images[batch_indices.type(torch.long),:,coords[:,0].type(torch.long),coords[:,1].type(torch.long)])
            for loss in self.losses
        ]
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
