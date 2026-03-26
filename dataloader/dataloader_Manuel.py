#include <pybind11/stl.h>
import torch.utils.data as data
from PIL import Image
import os
import os.path
from os import listdir, scandir
from sys import exit
import torch
import re
import cv2
import numpy as np
import torchvision.transforms as transforms
import argparse
import time
import random
#from lib.transformations import quaternion_from_euler, euler_matrix, random_quaternion, quaternion_matrix
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy.ma as ma
import open3d as o3d
import open3d.visualization as vis
from tqdm import tqdm
from pathlib import Path

 
class PoseDataset2(data.Dataset):
    def __init__(self, mode="all", num_pt=15000, concatmethod="depth", maskedmethod="depth", run=False):
        self.concatmethod = concatmethod
        self.maskedmethod = maskedmethod
        self.mode = mode
        self.run = run

        current_file = os.path.abspath(__file__)
        root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

        dataset_path = os.path.join(root, "DATASETS")

        self.path_depth = f"{dataset_path}/Dataset 6DManuel/results"
        self.path_rgb = f"{dataset_path}/Dataset 6DManuel/results"

        all_folders = [d for d in os.listdir(self.path_depth) if os.path.isdir(os.path.join(self.path_depth, d))]
        all_folders.sort()

        self.num_pt_mesh_large = num_pt
        self.num_pt_mesh_small = num_pt
        self.num_points = num_pt
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406,0.5], std=[0.229, 0.224, 0.225,0.5])

        self.noise_pc_std = 0.05
        self.noise_depth_std = 0.05
        self.h_std = 0.05
        self.s_std = 0.05
        self.v_std = 0.05

        self.list_pc = []
        self.list_pc_depth = []
        self.list_pc_velod = []
        self.list_pc_model = []

        self.list_label = []
        self.list_rgb = []
        self.list_mask = []
        self.list_depth = []
        self.list_depthV = []
        self.list_depthM = []
        self.list_T = []

        for folder in tqdm(all_folders, desc="Dataloader"):
            f_path = os.path.join(self.path_depth, folder)
            for file in sorted(os.scandir(f_path), key=lambda x: x.name):
                if 'RGB' in file.name:
                    fn = file.name
                    fpls = fn.split('_')
                    d1 = fpls[-1][3:]
                    d2 = d1.split('.')
                    str_det = fpls[1] + '_' + fpls[2] + '_det' + d2[0]
                    str_seg = fpls[1] + '_' + fpls[2] + '_seg' + d2[0]
                    detX = 'det' + d2[0]

                    vals = np.fromstring(open(f"{f_path}/{detX}.txt", 'r').read().replace(';',' '), sep=' ')

                    # verificacao de distancia
                    M = vals.reshape(4, 4)
                    t = M[:3, 3]
                    dist = np.linalg.norm(t)
                    if dist > 20.0 or dist <= 0.5:
                        continue

                    # verificacao pontos
                    pcd_depth = o3d.io.read_point_cloud(f"{f_path}/PC_DEPTH_{str_det}.ply")
                    pcd_velodyne = o3d.io.read_point_cloud(f"{f_path}/PC_VELODYNE_{str_det}.ply")
                    pcd_model = o3d.io.read_point_cloud(f"{f_path}/PC_MODEL_{str_det}.ply")
                    num_pontos = len(pcd_velodyne.points)
                    if num_pontos <= 100:
                        continue

                    self.list_pc_depth.append(f"{f_path}/PC_DEPTH_{str_det}.ply")
                    self.list_pc_velod.append(f"{f_path}/PC_VELODYNE_{str_det}.ply")
                    self.list_pc_model.append(f"{f_path}/PC_MODEL_{str_det}.ply")

                    self.list_pc_model.append(f"{f_path}/PC_MODEL_{str_det}.ply")

                    self.list_rgb.append(f"{f_path}/RGB_{str_det}.png")
                    self.list_mask.append(f"{f_path}/mask_{str_seg}.png")
                    self.list_depth.append(f"{f_path}/DEPTH_{str_det}.png")
                    self.list_depthV.append(f"{f_path}/VELODYNE_{str_det}.png")
                    self.list_depthM.append(f"{f_path}/MODEL_{str_det}.png")

                    self.list_label.append(f"{f_path}/{detX}.txt")

        total_examples = len(self.list_rgb)
        indices = list(range(total_examples))
        random.Random(666).shuffle(indices)

        n_base = int(0.9 * total_examples)
        base_indices = indices[:n_base]      # sem augmentation
        aug_indices  = indices[n_base:]      # com augmentation

        train_split = int(0.8 * total_examples)

        if mode == 'train':
            selected_indices = indices[:train_split]
        elif mode == 'test':
            selected_indices = indices[train_split:]
        elif mode == 'all':
            selected_indices = indices

        self.base_indices = [i for i in base_indices if i in selected_indices]
        self.aug_indices  = [i for i in aug_indices  if i in selected_indices]

        # guardar listas SEM augmentation
        self.list_rgb_base      = [self.list_rgb[i] for i in self.base_indices]
        self.list_mask_base     = [self.list_mask[i] for i in self.base_indices]
        self.list_depth_base    = [self.list_depth[i] for i in self.base_indices]
        self.list_depthV_base   = [self.list_depthV[i] for i in self.base_indices]
        self.list_depthM_base   = [self.list_depthM[i] for i in self.base_indices]
        self.list_pc_depth_base = [self.list_pc_depth[i] for i in self.base_indices]
        self.list_pc_velod_base = [self.list_pc_velod[i] for i in self.base_indices]
        self.list_pc_model_base = [self.list_pc_model[i] for i in self.base_indices]
        self.list_label_base    = [self.list_label[i] for i in self.base_indices]

        # guardar listas COM augmentation
        self.list_rgb_aug      = [self.list_rgb[i] for i in self.aug_indices]
        self.list_mask_aug     = [self.list_mask[i] for i in self.aug_indices]
        self.list_depth_aug    = [self.list_depth[i] for i in self.aug_indices]
        self.list_depthV_aug   = [self.list_depthV[i] for i in self.aug_indices]
        self.list_depthM_aug   = [self.list_depthM[i] for i in self.aug_indices]
        self.list_pc_depth_aug = [self.list_pc_depth[i] for i in self.aug_indices]
        self.list_pc_velod_aug = [self.list_pc_velod[i] for i in self.aug_indices]
        self.list_pc_model_aug = [self.list_pc_model[i] for i in self.aug_indices]
        self.list_label_aug    = [self.list_label[i] for i in self.aug_indices]

        self.list_rgb      = self.list_rgb_base + self.list_rgb_aug
        self.list_mask     = self.list_mask_base + self.list_mask_aug
        self.list_depth    = self.list_depth_base + self.list_depth_aug
        self.list_depthV   = self.list_depthV_base + self.list_depthV_aug
        self.list_depthM   = self.list_depthM_base + self.list_depthM_aug
        self.list_pc_depth = self.list_pc_depth_base + self.list_pc_depth_aug
        self.list_pc_velod = self.list_pc_velod_base + self.list_pc_velod_aug
        self.list_pc_model = self.list_pc_model_base + self.list_pc_model_aug
        self.list_label    = self.list_label_base + self.list_label_aug

        # salvar quantos exemplos têm cada subset
        self.n_base = len(self.list_rgb_base)
        self.n_aug  = len(self.list_rgb_aug)

        # comprimento final
        self.length = len(self.list_rgb)

    def __getitem__(self, j):
        is_aug = j >= len(self.list_rgb_base)
        idx = int(re.search(r'class(\d+)', self.list_depth[j]).group(1))

        info_points = self.load_pointclouds(j)
        pointcloud_cam_W, pointcloud_cam, pointcloud_vel_W, pointcloud_vel, pointcloud_model_W, pointcloud_model, modelPoints_W, modelPoints, rt = info_points

        # self.show_pointclouds(pointcloud_cam_W, pointcloud_vel_W, pointcloud_model_W, "camW", "depthW", "modelW")
        # self.show_pointclouds(pointcloud_cam, pointcloud_vel, pointcloud_model, "cam", "depth", "model")

        with Image.open(self.list_rgb[j]) as img_open, \
            Image.open(self.list_depth[j]) as depth_open, \
            Image.open(self.list_depthV[j]) as depth_openV, \
            Image.open(self.list_depthM[j]) as depth_openM, \
            Image.open(self.list_mask[j]) as mask_open:

            width, height = img_open.size
            resize_size = (224, 224)
            img_open = img_open.resize(resize_size)
            depth_open = depth_open.resize(resize_size)
            depth_openV = depth_openV.resize(resize_size)
            depth_openM = depth_openM.resize(resize_size)
            mask_open = mask_open.resize(resize_size)

            img = np.array(img_open)
            depth = np.array(depth_open)
            depthV = np.array(depth_openV)
            depthM = np.array(depth_openM)
            mask = np.array(mask_open)
        
        # Normalize depth to [0, 1] range
        depth_normalized = depth / 65535.0
        depth_expanded = np.expand_dims(depth_normalized, axis=-1)
        
        depth_normalizedV = depthV / 65535.0
        depth_expandedV = np.expand_dims(depth_normalizedV, axis=-1)

        depth_normalizedM = depthM / 65535.0
        depth_expandedM = np.expand_dims(depth_normalizedM, axis=-1)

        mask = np.expand_dims(mask, axis=-1)

        if is_aug:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            h, s, v = cv2.split(hsv)

            # Aplica ruído gaussiano em cada canal
            h += np.random.randn(*h.shape) * (self.h_std * 180)
            s += np.random.randn(*s.shape) * (self.s_std * 255)
            v += np.random.randn(*v.shape) * (self.v_std * 255)

            # Clipa valores para intervalos válidos
            h = np.clip(h, 0, 179)
            s = np.clip(s, 0, 255)
            v = np.clip(v, 0, 255)

            hsv = cv2.merge([h, s, v]).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            noise_depth = np.random.normal(loc=0.0, scale=self.noise_depth_std, size=depth_expanded.shape)

            depth_expanded = depth_expanded + noise_depth
            depth_expandedV = depth_expandedV + noise_depth
            depth_expandedM = depth_expandedM + noise_depth

        img4c = np.zeros((4, img.shape[1], img.shape[2]), dtype=np.float32)

        if self.concatmethod == "depth":
            concat = np.concatenate((img/256.0, depth_expanded), axis=-1)
        elif self.concatmethod == "velodyne":
            concat = np.concatenate((img/256.0, depth_expandedV), axis=-1)
        elif self.concatmethod == "model":
            concat = np.concatenate((img/256.0, depth_expandedM), axis=-1)
        img4c = np.transpose(concat, (2, 0, 1))

        img_masked = img4c
        img_masked = torch.from_numpy(img_masked.astype(np.float32))

        if self.maskedmethod == "depth":
            depth_expanded = torch.from_numpy(depth_expanded.astype(np.float32))
        elif self.maskedmethod == "velodyne":
            depth_expanded = torch.from_numpy(depth_expandedV.astype(np.float32))
        elif self.maskedmethod == "model":
            depth_expanded = torch.from_numpy(depth_expandedM.astype(np.float32))

        choose = torch.LongTensor([0])

        # # Open3D visualization
        # pc_depth_3dd = o3d.io.read_point_cloud(self.list_pc_depth[j])
        # pc_depthvel_3dd = o3d.io.read_point_cloud(self.list_pc_velod[j])
        # vis.draw(geometry=pc_depth_3dd, non_blocking_and_return_uid=True, title='PC DEPTH')
        # vis.draw(geometry=pc_depthvel_3dd, non_blocking_and_return_uid=True, title='PC VEL')

        if is_aug and self.mode == "train":
            noise_pc = np.random.normal(loc=0.0, scale=self.noise_pc_std, size=pointcloud_cam_W.shape)
            
            pointcloud_cam_W = pointcloud_cam_W + noise_pc
            pointcloud_cam = pointcloud_cam + noise_pc
            pointcloud_vel_W = pointcloud_vel_W + noise_pc
            pointcloud_vel = pointcloud_vel + noise_pc
            pointcloud_model_W = pointcloud_model_W + noise_pc
            pointcloud_model = pointcloud_model + noise_pc

        """print(pointcloud_cam_W.shape, pointcloud_cam.shape)
        print(pointcloud_vel_W.shape, pointcloud_vel.shape)
        print(pointcloud_model_W.shape, pointcloud_model.shape)
        print(img_masked.shape)
        print(depth_expanded.shape)
        print(modelPoints.shape)
        print(modelPointsGT.shape)
        print(rt.shape)
        print("------------------------")"""

        file_name = self.list_label[j]

        if self.run:
            return  torch.from_numpy(pointcloud_cam_W.astype(np.float32)),\
                    torch.from_numpy(pointcloud_cam.astype(np.float32)),\
                    torch.from_numpy(pointcloud_vel_W.astype(np.float32)),\
                    torch.from_numpy(pointcloud_vel.astype(np.float32)),\
                    torch.from_numpy(pointcloud_model_W.astype(np.float32)),\
                    torch.from_numpy(pointcloud_model.astype(np.float32)),\
                    img_masked, depth_expanded, modelPoints, modelPoints_W, rt, idx, file_name
        else:
            return  torch.from_numpy(pointcloud_cam_W.astype(np.float32)),\
                    torch.from_numpy(pointcloud_cam.astype(np.float32)),\
                    torch.from_numpy(pointcloud_vel_W.astype(np.float32)),\
                    torch.from_numpy(pointcloud_vel.astype(np.float32)),\
                    torch.from_numpy(pointcloud_model_W.astype(np.float32)),\
                    torch.from_numpy(pointcloud_model.astype(np.float32)),\
                    img_masked, depth_expanded, modelPoints, modelPoints_W, rt, idx

    def load_pointclouds(self, j):
        # Load pointclouds
        try:
            pointcloud_cam_W = np.asarray(o3d.io.read_point_cloud(self.list_pc_depth[j]).points)
            pointcloud_cam_W_colors = np.asarray(o3d.io.read_point_cloud(self.list_pc_depth[j]).colors)
            pointcloud_vel_W = np.asarray(o3d.io.read_point_cloud(self.list_pc_velod[j], format = "ply").points)
            pointcloud_vel_W_colors = np.asarray(o3d.io.read_point_cloud(self.list_pc_velod[j]).colors)
            pointcloud_model_W = np.asarray(o3d.io.read_point_cloud(self.list_pc_model[j], format = "ply").points)
            pointcloud_model_W_colors = np.asarray(o3d.io.read_point_cloud(self.list_pc_model[j]).colors)
        except FileNotFoundError:
            exit("ERROR: Necessary PC files not found. Exiting program")

        # self.show_pointclouds(pointcloud_cam_W, pointcloud_vel_W, pointcloud_model_W, "camW", "depthW", "modelW")

        # views = self.point_cloud_to_views_centered(pointcloud_cam_W, pointcloud_cam_W_colors, img_size=224)
        
        voxel_grid = self.voxelization(j)

        voxel_centers = np.array([voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
                          for voxel in voxel_grid.get_voxels()])

        # criar uma PointCloud
        pcd_from_voxels = o3d.geometry.PointCloud()
        pcd_from_voxels.points = o3d.utility.Vector3dVector(voxel_centers)

        o3d.visualization.draw_geometries(
            [pcd_from_voxels],
            window_name="Voxelizado (janela pequena)"
        )

        """fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(
            voxel_grid[:, 0],
            voxel_grid[:, 1],
            voxel_grid[:, 2],
            s=1,
            alpha=0.5,
            label="voxel"
        )

        ax.legend()
        plt.show()"""

        # Load rotation matrix
        try:
            with open(self.list_label[j], 'r') as f:
                rt = f.read().replace(';', ' ')
            rt = np.fromstring(rt, sep=' ').reshape((4, 4))
        except:
            rt = np.identity(4)
            print("Matriz T nao encontrada = identidade")

        # Compute the inverse matrix
        rt_inv = np.linalg.inv(rt)

        rotation_inv = rt_inv[:3, :3]
        translation_inv = rt_inv[:3, 3]

        pointcloud_cam = (rotation_inv @ pointcloud_cam_W.T).T + translation_inv
        pointcloud_vel = (rotation_inv @ pointcloud_vel_W.T).T + translation_inv
        pointcloud_model = (rotation_inv @ pointcloud_model_W.T).T + translation_inv

        # self.show_pointclouds(pointcloud_cam, pointcloud_vel, pointcloud_model, "cam", "depth", "model")

        # SEED
        # random.seed(42)
        # Ensure we don't sample more points than are available
        array = random.choices(range(0, pointcloud_cam.shape[0]), k=self.num_points)
        pointcloud_cam = pointcloud_cam[array,:]
        pointcloud_cam_W = pointcloud_cam_W[array,:]

        array2 = random.choices(range(0, pointcloud_vel.shape[0]), k=self.num_points)
        pointcloud_vel = pointcloud_vel[array2,:]
        pointcloud_vel_W = pointcloud_vel_W[array2,:]

        array3 = random.choices(range(0, pointcloud_model.shape[0]), k=self.num_points)
        pointcloud_model = pointcloud_model[array3,:]
        pointcloud_model_W = pointcloud_model_W[array3,:]

        modelPoints = np.array([[0., 0., 0.],
                                [1., 0., 0.],
                                [0., 1., 0.],
                                [0., 0., 1.]],dtype=np.float32)

        rotation = rt[:3, :3]
        translation = rt[:3, 3]

        modelPoints_W = (rotation @ modelPoints.T).T + translation

        return pointcloud_cam_W, pointcloud_cam, pointcloud_vel_W, pointcloud_vel, pointcloud_model_W, pointcloud_model, modelPoints_W, modelPoints, rt
    
    def point_cloud_to_rgb_projection(self, points, colors, xlim, ylim, zlim, img_size=224):
        # Filter points within limits
        mask = (
            (points[:,0] >= xlim[0]) & (points[:,0] <= xlim[1]) &
            (points[:,1] >= ylim[0]) & (points[:,1] <= ylim[1]) &
            (points[:,2] >= zlim[0]) & (points[:,2] <= zlim[1])
        )
        pts = points[mask]
        cols = (colors[mask] * 255).astype(np.uint8)  # converter para [0,255]

        """# Map to image grid
        def scale_to_grid(values, vmin, vmax):
            return np.clip(((values - vmin) / (vmax - vmin)) * (img_size - 1), 0, img_size - 1).astype(int)

        xs = scale_to_grid(pts[:,0], *xlim)
        ys = scale_to_grid(pts[:,1], *ylim)
        zs = scale_to_grid(pts[:,2], *zlim)"""

        coords = (pts * (img_size - 1)).astype(int)

        # Inicializa imagem RGB
        rgb_image = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        # Preenche imagem com cores dos pontos
        for (coord, c) in zip(coords, cols):
            x, y, z, = coord
            rgb_image[img_size-1-y, x] = c   # podes mudar para projecções diferentes
            # Se quiseres guardar info também em vistas Coronal / Sagittal
            rgb_image[img_size-1-z, x] = c
            rgb_image[img_size-1-z, y] = c

        return rgb_image
    
    def point_cloud_to_views(self, points, colors, img_size=224):
        # Normaliza cores para [0,255]
        cols = (colors * 255).astype(np.uint8)

        # Obtem bounding box da point cloud
        mins = points.min(axis=0)
        maxs = points.max(axis=0)

        # Função para escalar valores para [0, img_size-1]
        def scale(values, vmin, vmax):
            if vmax == vmin:  # evitar divisão por zero
                return np.zeros_like(values, dtype=int)
            return np.clip(((values - vmin) / (vmax - vmin)) * (img_size - 1), 0, img_size - 1).astype(int)

        # Escala coordenadas para pixel space
        xs = scale(points[:,0], mins[0], maxs[0])
        ys = scale(points[:,1], mins[1], maxs[1])
        zs = scale(points[:,2], mins[2], maxs[2])

        # Inicializa imagens
        axial   = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        coronal = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        sagittal= np.zeros((img_size, img_size, 3), dtype=np.uint8)

        # Preenche cada vista
        for (x, y, z, c) in zip(xs, ys, zs, cols):
            axial[img_size-1-y, x]   = c   # XY
            coronal[img_size-1-z, x] = c   # XZ
            sagittal[img_size-1-z, y]= c   # YZ

        return {"axial": axial, "coronal": coronal, "sagittal": sagittal}
    
    def point_cloud_to_views_centered(self, points, colors, img_size=224, border_ratio=0.02):
        """Zmax = 30  # metros (frente-trás)

        FOVx = np.deg2rad(85.2)   # horizontal
        FOVy = np.deg2rad(58.0)   # vertical

        Xmax = Zmax * np.tan(FOVx / 2)
        Ymax = Zmax * np.tan(FOVy / 2)

        xlim = [-Xmax, Xmax]
        ylim = [-Ymax, Ymax]
        zlim = [0.0, Zmax]

        mask = (
            (points[:,0] >= xlim[0]) & (points[:,0] <= xlim[1]) &
            (points[:,1] >= ylim[0]) & (points[:,1] <= ylim[1]) &
            (points[:,2] >= zlim[0]) & (points[:,2] <= zlim[1])
        )
        points = points[mask]"""

        cols = (colors * 255).astype(np.uint8)

        """if points.shape[0] == 0:
            H = W = 224
            axial   = np.zeros((H, W, 3), dtype=np.uint8)
            coronal = np.zeros((H, W, 3), dtype=np.uint8)
            sagittal= np.zeros((H, W, 3), dtype=np.uint8)
            return {"axial": axial, "coronal": coronal, "sagittal": sagittal}"""
        
        # Bounding box original
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        ranges = maxs - mins

        # Adiciona borda (ex: 10%)
        mins = mins - border_ratio * ranges
        maxs = maxs + border_ratio * ranges

        # Função para mapear valores para pixel space
        def scale(values, vmin, vmax, H, W):
            return np.clip(((values - vmin) / (vmax - vmin)) * (W - 1), 0, W - 1).astype(int)

        # Escolher resolução intermediária proporcional
        max_range = ranges.max()
        W = H = int(max_range * 10)  # fator de escala arbitrário (podes ajustar)

        xs = scale(points[:,0], mins[0], maxs[0], H, W)
        ys = scale(points[:,1], mins[1], maxs[1], H, W)
        zs = scale(points[:,2], mins[2], maxs[2], H, W)

        # Inicializa imagens
        axial   = np.zeros((H, W, 3), dtype=np.uint8)
        coronal = np.zeros((H, W, 3), dtype=np.uint8)
        sagittal= np.zeros((H, W, 3), dtype=np.uint8)

        # Preenche imagens
        for (x, y, z, c) in zip(xs, ys, zs, cols):
            axial[H-1-y, x]   = c   # XY
            coronal[H-1-z, x] = c   # XZ
            sagittal[H-1-z, y]= c   # YZ

        # Resize final para 224x224
        axial   = cv2.resize(axial,   (img_size, img_size), interpolation=cv2.INTER_AREA)
        coronal = cv2.resize(coronal, (img_size, img_size), interpolation=cv2.INTER_AREA)
        sagittal= cv2.resize(sagittal,(img_size, img_size), interpolation=cv2.INTER_AREA)

        return {"axial": axial, "coronal": coronal, "sagittal": sagittal}
        
    def voxelization(self, j):
        try:
            pointcloud_vel_W = o3d.io.read_point_cloud(self.list_pc_velod[j], format = "ply").points
        except FileNotFoundError:
            exit("ERROR: Necessary PC files not found. Exiting program")

        pointcloud_vel_W = np.asarray(pointcloud_vel_W)

        # Load rotation matrix
        try:
            with open(self.list_label[j], 'r') as f:
                rt = f.read().replace(';', ' ')
            rt = np.fromstring(rt, sep=' ').reshape((4, 4))
        except:
            rt = np.identity(4)
            print("Matriz T nao encontrada = identidade")

        # Compute the inverse matrix
        rt_inv = np.linalg.inv(rt)

        rotation_inv = rt_inv[:3, :3]
        translation_inv = rt_inv[:3, 3]

        # pointcloud_vel_W = np.asarray(pointcloud_vel_W)

        pointcloud_vel = (rotation_inv @ pointcloud_vel_W.T).T + translation_inv

        random.seed(42)
        array2 = random.choices(range(0, pointcloud_vel.shape[0]), k=self.num_points)
        pointcloud_vel = pointcloud_vel[array2,:]

        pcd_vel = o3d.geometry.PointCloud()
        pcd_vel.points = o3d.utility.Vector3dVector(pointcloud_vel)

        voxel_size = 0.05
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_vel, voxel_size=voxel_size)
    
        # o3d.visualization.draw_geometries(
        #     [voxel_grid],
        #     window_name="Voxelizado (janela pequena)"
        # )

        return voxel_grid

    def __len__(self):
        return self.length

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small

    def show_pointclouds(self, pc1, pc2, pc3, label1, label2, label3):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(
            pc1[:, 0],
            pc1[:, 1],
            pc1[:, 2],
            s=1,
            alpha=0.5,
            label=label1
        )

        ax.scatter(
            pc2[:, 0],
            pc2[:, 1],
            pc2[:, 2],
            s=1,
            alpha=0.5,
            label=label2
        )

        ax.scatter(
            pc3[:, 0],
            pc3[:, 1],
            pc3[:, 2],
            s=1,
            alpha=0.5,
            label=label3
        )

        ax.legend()

        plt.show()

def show(pc_depth_W, pc_depth, pc_velodyne_W, pc_velodyne, pc_model_W, pc_model, img, depth_vel, views, modelPoints, modelPoints_W, rt, idx):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        pc_depth_W[:, 0],
        pc_depth_W[:, 1],
        pc_depth_W[:, 2],
        s=1,
        alpha=0.5,
        label='pointcloud depth mundo'
    )

    ax.scatter(
        pc_depth[:, 0],
        pc_depth[:, 1],
        pc_depth[:, 2],
        s=1,
        alpha=0.5,
        label='pointcloud depth'
    )

    """ax.scatter(
        pc_velodyne_W[:, 0],
        pc_velodyne_W[:, 1],
        pc_velodyne_W[:, 2],
        s=1,
        alpha=0.5,
        label='pointcloud velodyne mundo'
    )

    ax.scatter(
        pc_velodyne[:, 0],
        pc_velodyne[:, 1],
        pc_velodyne[:, 2],
        s=1,
        alpha=0.5,
        label='pointcloud velodyne'
    )

    ax.scatter(
        pc_model_W[:, 0],
        pc_model_W[:, 1],
        pc_model_W[:, 2],
        s=1,
        alpha=0.5,
        label='pointcloud modelo mundo'
    )

    ax.scatter(
        pc_model[:, 0],
        pc_model[:, 1],
        pc_model[:, 2],
        s=1,
        alpha=0.5,
        label='pointcloud modelo'
    )"""

    for i in range(1, len(modelPoints)):
        x1 = [modelPoints_W[0, 0], modelPoints_W[i, 0]]
        y1 = [modelPoints_W[0, 1], modelPoints_W[i, 1]]
        z1 = [modelPoints_W[0, 2], modelPoints_W[i, 2]]

        x_pred = [modelPoints[0, 0], modelPoints[i, 0]]
        y_pred = [modelPoints[0, 1], modelPoints[i, 1]]
        z_pred = [modelPoints[0, 2], modelPoints[i, 2]]

        ax.plot(x1, y1, z1, color='blue', linewidth=1)
        ax.plot(x_pred, y_pred, z_pred, color='orange', linewidth=1)

    # Adicionar texto no final dos eixos
    ax.text(x1[1], y1[1], z1[1], "World", color='blue', fontsize=7)
    ax.text(x_pred[1], y_pred[1], z_pred[1], "Origin", color='orange', fontsize=7)

    rgb = img[:3, :, :]
    depth_img = img[3:4, :, :]

    rgb_plot = rgb.permute(1, 2, 0).numpy()

    depth_img_plot = depth_img.squeeze(0).numpy()
    depth_vel_plot = depth_vel.squeeze(0).numpy()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 6, 1)
    plt.imshow(rgb_plot)
    plt.title("RGB")
    plt.axis("off")
    plt.subplot(1, 6, 2)
    plt.imshow(depth_img_plot, cmap='viridis')
    plt.title("Depth Img 1")
    plt.axis("off")
    plt.subplot(1, 6, 3)
    plt.imshow(depth_vel_plot, cmap='viridis')
    plt.title("Depth Img 2")
    plt.axis("off")
    plt.subplot(1, 6, 4)
    # plt.imshow(views["axial"])
    plt.title("view axial")
    plt.axis("off")
    plt.subplot(1, 6, 5)
    # plt.imshow(views["coronal"])
    plt.title("view coronal")
    plt.axis("off")
    plt.subplot(1, 6, 6)
    # plt.imshow(views["sagittal"])
    plt.title("view sagittal")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    concat = "depth"
    mask = "model"

    dataset = PoseDataset2('all', num_pt=15000, concatmethod=concat, maskedmethod=mask)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=10)

    for data in tqdm(dataset):
        pc_depth_W, pc_depth, pc_velodyne_W, pc_velodyne, pc_model_W, pc_model, img, depth_vel, modelPoints, modelPoints_W, rt, idx = data

        show(pc_depth_W, pc_depth, pc_velodyne_W, pc_velodyne, pc_model_W, pc_model, img, depth_vel, None, modelPoints, modelPoints_W, rt, idx)
    
    """for data in tqdm(dataset):
        pc_depth_W, pc_depth, pc_velodyne_W, pc_velodyne, pc_model_W, pc_model, img, depth_vel, views, modelPoints, modelPoints_W, rt, idx = data

        show(pc_depth_W, pc_depth, pc_velodyne_W, pc_velodyne, pc_model_W, pc_model, img, depth_vel, views, modelPoints, modelPoints_W, rt, idx)


        if idx == 5:
            ...
            # show(pc_depth_W, pc_depth, pc_velodyne_W, pc_velodyne, pc_model_W, pc_model, img, depth_vel, modelPoints, modelPoints_W, rt, idx)"""