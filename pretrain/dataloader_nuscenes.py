import os
import copy
import torch
import numpy as np
from PIL import Image
import MinkowskiEngine as ME
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud

later_workdir= 'datasets/nuscenes/samples/LIDAR_TOP_keypoints/'
point_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
CUSTOM_SPLIT = [#100
    "scene-0008", "scene-0009", "scene-0019", "scene-0029", "scene-0032", "scene-0042",
    "scene-0045", "scene-0049", "scene-0052", "scene-0054", "scene-0056", "scene-0066",
    "scene-0067", "scene-0073", "scene-0131", "scene-0152", "scene-0166", "scene-0168",
    "scene-0183", "scene-0190", "scene-0194", "scene-0208", "scene-0210", "scene-0211",
    "scene-0241", "scene-0243", "scene-0248", "scene-0259", "scene-0260", "scene-0261",
    "scene-0287", "scene-0292", "scene-0297", "scene-0305", "scene-0306", "scene-0350",
    "scene-0352", "scene-0358", "scene-0361", "scene-0365", "scene-0368", "scene-0377",
    "scene-0388", "scene-0391", "scene-0395", "scene-0413", "scene-0427", "scene-0428",
    "scene-0438", "scene-0444", "scene-0452", "scene-0453", "scene-0459", "scene-0463",
    "scene-0464", "scene-0475", "scene-0513", "scene-0533", "scene-0544", "scene-0575",
    "scene-0587", "scene-0589", "scene-0642", "scene-0652", "scene-0658", "scene-0669",
    "scene-0678", "scene-0687", "scene-0701", "scene-0703", "scene-0706", "scene-0710",
    "scene-0715", "scene-0726", "scene-0735", "scene-0740", "scene-0758", "scene-0786",
    "scene-0790", "scene-0804", "scene-0806", "scene-0847", "scene-0856", "scene-0868",
    "scene-0882", "scene-0897", "scene-0899", "scene-0976", "scene-0996", "scene-1012",
    "scene-1015", "scene-1016", "scene-1018", "scene-1020", "scene-1024", "scene-1044",
    "scene-1058", "scene-1094", "scene-1098", "scene-1107",
]


def minkunet_collate_pair_fn(list_data):
    """
    Collate function adapted for creating batches with MinkowskiEngine.
    """
    (
        coords,
        feats,
        images,
        pairing_points,
        pairing_images,
        depth_map,
        inverse_indexes,
        superpixels,
        points,lidar2egos,ego2globals,lidar2cameras,lidar2images,camera2egos,camera_intrinsicss,
        lidar_aug_matrix,image_aug_matrix,
    ) = list(zip(*list_data))
    batch_n_points, batch_n_pairings = [], []

    offset = 0
    for batch_id in range(len(coords)):

        # Move batchids to the beginning
        coords[batch_id][:, 0] = batch_id
        pairing_points[batch_id][:] += offset
        pairing_images[batch_id][:, 0] += batch_id * images[0].shape[0]

        batch_n_points.append(coords[batch_id].shape[0])
        batch_n_pairings.append(pairing_points[batch_id].shape[0])
        offset += coords[batch_id].shape[0]

    # Concatenate all lists
    coords_batch = torch.cat(coords, 0).int()
    pairing_points = torch.tensor(np.concatenate(pairing_points))
    pairing_images = torch.tensor(np.concatenate(pairing_images))
    depth_map = torch.tensor(depth_map)
    feats_batch = torch.cat(feats, 0).float()
    images_batch = torch.cat(images, 0).float()
    superpixels_batch = torch.tensor(np.concatenate(superpixels))
    lidar2egos_batch = torch.tensor(lidar2egos).float()
    ego2globals_batch = torch.tensor(ego2globals).float()
    lidar_aug_matrixs =torch.tensor(lidar_aug_matrix).float()#torch.tensor(lidar_aug_matrix).float() #torch.cat(lidar_aug_matrix[0], 0).float()
    image_aug_matrixs = torch.stack(image_aug_matrix).float()
    lidar2cameras_batch = torch.tensor(lidar2cameras).float()
    camera2egos_batch = torch.tensor(camera2egos).float()
    camera_intrinsicss_batch = torch.tensor(camera_intrinsicss).float()
    return {
        "sinput_C": coords_batch,
        "sinput_F": feats_batch,
        "input_I": images_batch,
        "pairing_points": pairing_points,
        "pairing_images": pairing_images,
        'depth_map': depth_map,
        "batch_n_pairings": batch_n_pairings,
        "inverse_indexes": inverse_indexes,
        "superpixels": superpixels_batch,
        "lidar2egos":lidar2egos_batch,
        "lidar2cameras":lidar2cameras_batch,
        "lidar2images":lidar2images,#
        "camera2egos":camera2egos_batch,
        "camera_intrinsicss":camera_intrinsicss_batch,
        "points":points,
        "lidar_aug_matrixs": lidar_aug_matrixs,
        "image_aug_matrixs":image_aug_matrixs,
    }


class NuScenesMatchDataset(Dataset):
    """
    Dataset matching a 3D points cloud and an image using projection.
    """

    def __init__(
        self,
        phase,
        config,
        shuffle=False,
        cloud_transforms=None,
        mixed_transforms=None,
        **kwargs,
    ):
        self.phase = phase
        self.shuffle = shuffle
        self.cloud_transforms = cloud_transforms
        self.mixed_transforms = mixed_transforms
        self.voxel_size = config["voxel_size"]
        self.cylinder = config["cylindrical_coordinates"]
        self.superpixels_type = config["superpixels_type"]
        self.bilinear_decoder = config["decoder"] == "bilinear"

        if "cached_nuscenes" in kwargs:
            self.nusc = kwargs["cached_nuscenes"]
        else:
            self.nusc = NuScenes(
                version="v1.0-trainval", dataroot="dataset/nuscense", verbose=False
            )


        self.list_keyframes = []
        # a skip ratio can be used to reduce the dataset size and accelerate experiments
        try:
            skip_ratio = config["dataset_skip_step"]
        except KeyError:
            skip_ratio = 1
        skip_counter = 0
        if phase in ("train", "val", "test"):
            phase_scenes = create_splits_scenes()[phase]
        elif phase == "parametrizing":
            phase_scenes = list(
                set(create_splits_scenes()["train"]) - set(CUSTOM_SPLIT)
            )
        elif phase == "verifying":
            phase_scenes = CUSTOM_SPLIT
        # create a list of camera & lidar scans
        for scene_idx in range(len(self.nusc.scene)):
            scene = self.nusc.scene[scene_idx]
            if scene["name"] in phase_scenes:
                skip_counter += 1
                if skip_counter % skip_ratio == 0:
                    self.create_list_of_scans(scene)

    def create_list_of_scans(self, scene):
        # Get first and last keyframe in the scene
        current_sample_token = scene["first_sample_token"]

        # Loop to get all successive keyframes
        list_data = []
        while current_sample_token != "":
            current_sample = self.nusc.get("sample", current_sample_token)
            list_data.append(current_sample["data"])
            current_sample_token = current_sample["next"]

        # Add new scans in the list
        self.list_keyframes.extend(list_data)

    def map_pointcloud_to_image(self, data, min_dist: float = 1.0):
        """
        Given a lidar token and camera sample_data token, load pointcloud and map it to
        the image plane. Code adapted from nuscenes-devkit
        https://github.com/nutonomy/nuscenes-devkit.
        :param min_dist: Distance from the camera below which points are discarded.
        """
        pointsensor = self.nusc.get("sample_data", data["LIDAR_TOP"])
        pcl_path = os.path.join(self.nusc.dataroot, pointsensor["filename"])
        pc_original = LidarPointCloud.from_file(pcl_path.replace('+','-'))
        # pc_ref = pc_original.points
        pc_ref = pc_original.points.T
        in_range_flags = (
            (pc_ref[:, 0] > point_range[0])
            & (pc_ref[:, 1] > point_range[1])
            & (pc_ref[:, 2] > point_range[2])
            & (pc_ref[:, 0] < point_range[3])
            & (pc_ref[:, 1] < point_range[4])
            & (pc_ref[:, 2] < point_range[5])
        )
        pc_ref = pc_ref[in_range_flags].T
        pc_original.points = pc_ref

        images = []
        superpixels = []
        pairing_points = np.empty(0, dtype=np.int64)
        pairing_images = np.empty((0, 4), dtype=np.float32)
        camera_list = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_FRONT_LEFT",
        ]
        if self.shuffle:
            np.random.shuffle(camera_list)
        
        # lidar to ego transform
        lidar2egos = []
        ego2globals = []
        lidar2cameras = []
        lidar2images = []
        camera2egos = []
        camera_intrinsicss = []
        lidar2ego = np.eye(4).astype(np.float32)
        cs_record = self.nusc.get(
                "calibrated_sensor", pointsensor["calibrated_sensor_token"]
            )
        lidar2ego_translation = cs_record["translation"]
        lidar2ego_rotation = cs_record["rotation"]
        lidar2ego[:3,:3] = Quaternion(lidar2ego_rotation).rotation_matrix
        lidar2ego[:3,3 ] = lidar2ego_translation

        # ego to global transform
        ego2global = np.eye(4).astype(np.float32)
        poserecord = self.nusc.get("ego_pose", pointsensor["ego_pose_token"])
        ego2global_translation = poserecord["translation"]
        ego2global_rotation =  poserecord["rotation"]
        ego2global[:3,:3] = Quaternion(ego2global_rotation).rotation_matrix
        ego2global[:3,3 ] = ego2global_translation

        for i, camera_name in enumerate(camera_list):
            pc = copy.deepcopy(pc_original)
            cam = self.nusc.get("sample_data", data[camera_name])
            im = np.array(Image.open(os.path.join(self.nusc.dataroot, cam["filename"].replace('+','-'))))
            sp = Image.open(
                f"superpixels/nuscenes/"
                f"superpixels_{self.superpixels_type}/{cam['token']}.png"
            )

            superpixels.append(np.array(sp))

            # Points live in the point sensor frame. So they need to be transformed via
            # global to the image plane.
            # First step: transform the pointcloud to the ego vehicle frame for the
            # timestamp of the sweep.
            cs_record = self.nusc.get(
                "calibrated_sensor", pointsensor["calibrated_sensor_token"]
            )
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
            pc.translate(np.array(cs_record["translation"]))

            # Second step: transform from ego to the global frame.
            poserecord = self.nusc.get("ego_pose", pointsensor["ego_pose_token"])
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
            pc.translate(np.array(poserecord["translation"]))

            # Third step: transform from global into the ego vehicle frame for the
            # timestamp of the image.
            poserecord = self.nusc.get("ego_pose", cam["ego_pose_token"])
            pc.translate(-np.array(poserecord["translation"]))
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

            # Fourth step: transform from ego into the camera.
            cs_record = self.nusc.get(
                "calibrated_sensor", cam["calibrated_sensor_token"]
            )
            pc.translate(-np.array(cs_record["translation"]))
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

            # Fifth step: actually take a "picture" of the point cloud.
            # Grab the depths (camera frame z axis points away from the camera).
            depths = pc.points[2, :]

            # Take the actual picture
            # (matrix multiplication with camera-matrix + renormalization).
            points = view_points(
                pc.points[:3, :],
                np.array(cs_record["camera_intrinsic"]),
                normalize=True,
            )

            l2e_r_s = cs_record["rotation"]
            l2e_t_s = cs_record["translation"]
            e2g_r_s = poserecord["rotation"]
            e2g_t_s = poserecord["translation"]

            # obtain the RT from sensor to Top LiDAR
            # sweep->ego->global->ego'->lidar
            l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
            e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
            e2g_r_mat = ego2global[:3,:3]
            e2g_t = ego2global_translation
            l2e_r_mat = lidar2ego[:3,:3]
            l2e_t = lidar2ego_translation
            R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
                np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
            )
            T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
                np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
            )
            T -= (
                e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
                + l2e_t @ np.linalg.inv(l2e_r_mat).T
            )

             # lidar to camera transform
            lidar2camera_r = np.linalg.inv(R.T)
            lidar2camera_t = (
                T @ lidar2camera_r.T
            )
            lidar2camera_rt = np.eye(4).astype(np.float32)
            lidar2camera_rt[:3, :3] = lidar2camera_r.T
            lidar2camera_rt[3, :3] = -lidar2camera_t

            # camera intrinsics
            camera_intrinsics = np.eye(4).astype(np.float32)
            camera_intrinsics[:3, :3] = cs_record["camera_intrinsic"]

            # lidar to image transform
            lidar2image = camera_intrinsics @ lidar2camera_rt.T

            # camera to ego transform
            camera2ego = np.eye(4).astype(np.float32)
            camera2ego[:3, :3] = Quaternion(
                 cs_record["rotation"]
            ).rotation_matrix
            camera2ego[:3, 3] = cs_record["translation"]

            # Remove points that are either outside or behind the camera.
            # Also make sure points are at least 1m in front of the camera to avoid
            # seeing the lidar points on the camera
            # casing for non-keyframes which are slightly out of sync.
            points = points[:2].T
            mask = np.ones(depths.shape[0], dtype=bool)
            mask = np.logical_and(mask, depths > min_dist)
            mask = np.logical_and(mask, points[:, 0] > 0)
            mask = np.logical_and(mask, points[:, 0] < im.shape[1] - 1)
            mask = np.logical_and(mask, points[:, 1] > 0)
            mask = np.logical_and(mask, points[:, 1] < im.shape[0] - 1)
            matching_points = np.where(mask)[0]
            matching_pixels_old = np.round(
                np.flip(points[matching_points], axis=1)
            ).astype(np.int64)
            coloring = depths
            coloring = coloring[mask]
            matching_pixels = np.concatenate([np.round(
                np.flip(points[matching_points], axis=1)
            ).astype(np.int64),coloring[:,None]],axis=1).astype(np.float32)
            images.append(im / 255)

            lidar2egos.append(lidar2ego)
            ego2globals.append(ego2global)
            lidar2cameras.append(lidar2camera_rt.T)
            lidar2images.append(lidar2image)
            camera2egos.append(camera2ego)
            camera_intrinsicss.append(camera_intrinsics)

            pairing_points = np.concatenate((pairing_points, matching_points))
            pairing_images = np.concatenate(
                (
                    pairing_images,
                    np.concatenate(
                        (
                            np.ones((matching_pixels.shape[0], 1), dtype=np.float32) * i,
                            matching_pixels,
                        ),
                        axis=1,
                    ),
                )
            )
        return pc_ref.T, images, pairing_points, pairing_images, np.stack(superpixels),lidar2egos,ego2globals,lidar2cameras,lidar2images,camera2egos,camera_intrinsicss

    def __len__(self):
        return len(self.list_keyframes)

    def __getitem__(self, idx):
        (
            pc,
            images,
            pairing_points,
            pairing_images,
            superpixels,
            lidar2egos,ego2globals,lidar2cameras,lidar2images,camera2egos,camera_intrinsicss,
        ) = self.map_pointcloud_to_image(self.list_keyframes[idx])
        superpixels = torch.tensor(superpixels)

        intensity = torch.tensor(pc[:, 3:])
        pc = torch.tensor(pc[:, :3])
        images = torch.tensor(np.array(images, dtype=np.float32).transpose(0, 3, 1, 2))

        lidar2egos = np.array(lidar2egos, dtype=np.float32)
        ego2globals = np.array(ego2globals, dtype=np.float32)
        lidar2cameras = np.array(lidar2cameras, dtype=np.float32)
        lidar2images = torch.tensor(np.array(lidar2images, dtype=np.float32))
        camera2egos = np.array(camera2egos, dtype=np.float32)
        camera_intrinsicss = np.array(camera_intrinsicss, dtype=np.float32)

        lidar_aug_matrix =np.eye(4).astype(np.float32)
        if self.cloud_transforms:
            pc,lidar_aug_matrix = self.cloud_transforms(pc)#3*4
        
        image_aug_matrix =torch.eye(4).unsqueeze(0).expand(6, -1,-1)
        if self.mixed_transforms:
            (
                pc,
                intensity,
                images,
                pairing_points,
                pairing_images,
                superpixels,
                image_aug_matrix
            ) = self.mixed_transforms(
                pc, intensity, images, pairing_points, pairing_images, superpixels
            )# 6 4*4
            image_aug_matrix = torch.stack(image_aug_matrix)

        if self.cylinder:
            # Transform to cylinder coordinate and scale for voxel size
            x, y, z = pc.T
            rho = torch.sqrt(x ** 2 + y ** 2) / self.voxel_size
            phi = torch.atan2(y, x) * 180 / np.pi  # corresponds to a split each 1°
            z = z / self.voxel_size
            coords_aug = torch.cat((rho[:, None], phi[:, None], z[:, None]), 1)
        else:
            coords_aug = pc / self.voxel_size

        # Voxelization with MinkowskiEngine
        discrete_coords, indexes, inverse_indexes = ME.utils.sparse_quantize(
            coords_aug.contiguous(), return_index=True, return_inverse=True
        )
        # indexes here are the indexes of points kept after the voxelization
        pairing_points = inverse_indexes[pairing_points]

        unique_feats = intensity[indexes]

        discrete_coords = torch.cat(
            (
                torch.zeros(discrete_coords.shape[0], 1, dtype=torch.int32),
                discrete_coords,
            ),
            1,
        )

        pairing_depth = pairing_images[:,3:]
        pairing_images = pairing_images[:,:3].astype(np.int64)
        depth_map = np.zeros((6, 1, images.shape[2],images.shape[3]))
        depth_map[pairing_images[:,0], :, pairing_images[:,1], pairing_images[:, 2]] = pairing_depth
        return (
            discrete_coords,
            unique_feats,
            images,
            pairing_points,
            pairing_images,
            depth_map,
            inverse_indexes,
            superpixels,
            pc,lidar2egos,ego2globals,lidar2cameras,lidar2images,camera2egos,camera_intrinsicss,
            lidar_aug_matrix,image_aug_matrix,
        )
