# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# --------------------------------------------------------

import os
import shutil
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from tqdm import tqdm

class Run:
    def __init__(self, runloader, estimator, criterion, opt):
        self.runloader = runloader
        self.option = opt.option
        self.modalities = opt.modalities
        self.opt = opt
        self.estimator = estimator
        self.criterion = criterion

    def view_results(self):
        self.estimator.eval()

        new_base_folder = '/home/goncalo/Manuel/DATASETS/results modelo/'

        if os.path.exists(new_base_folder):
            shutil.rmtree(new_base_folder)
        os.makedirs(new_base_folder, exist_ok=True)
        
        for j, data in tqdm(enumerate(self.runloader, 0), total=len(self.runloader), unit='batch'):
            pc_depth_W, pc_depth, pc_velodyne_W, pc_velodyne, pc_model_W, pc_model, img, depth_vel, modelPoints, modelPointsGT, rt, idx, file_name = data

            if self.opt.class_id != None:
                mask = idx == self.opt.class_id

                if mask.sum() == 0:
                    continue

                pc_depth_W = pc_depth_W[mask]
                pc_depth = pc_depth[mask]
                pc_velodyne_W = pc_velodyne_W[mask]
                pc_velodyne = pc_velodyne[mask]
                pc_model_W = pc_model_W[mask]
                pc_model = pc_model[mask]
                img = img[mask]
                depth_vel = depth_vel[mask]
                modelPoints = modelPoints[mask]
                modelPointsGT = modelPointsGT[mask]
                rt = rt[mask]
                idx = idx[mask]
            
            if self.modalities == 0:
                RGBEnable = float(1)
                Depth1Enable = float(1)
                Depth2Enable = float(1)
                PC1Enable = float(1)
                PC2Enable = float(1)
            elif self.modalities == 1:
                RGBEnable = float(1)
                Depth1Enable = float(0)
                Depth2Enable = float(0)
                PC1Enable = float(0)
                PC2Enable = float(0)
            elif self.modalities == 2:
                RGBEnable = float(1)
                Depth1Enable = float(1)
                Depth2Enable = float(0)
                PC1Enable = float(0)
                PC2Enable = float(0)
            elif self.modalities == 3:
                RGBEnable = float(1)
                Depth1Enable = float(1)
                Depth2Enable = float(0)
                PC1Enable = float(1)
                PC2Enable = float(0)
            elif self.modalities == 4:
                RGBEnable = float(1)
                Depth1Enable = float(0)
                Depth2Enable = float(1)
                PC1Enable = float(0)
                PC2Enable = float(1)
            elif self.modalities == 5:
                RGBEnable = float(1)
                Depth1Enable = float(1)
                Depth2Enable = float(1)
                PC1Enable = float(0)
                PC2Enable = float(0)

            points = Variable(pc_depth).cuda()  # cam
            target = Variable(pc_depth_W).cuda()
            velodyne = Variable(pc_velodyne).cuda()
            velodyne_gt = Variable(pc_velodyne_W).cuda()
            model = Variable(pc_model).cuda()
            model_gt = Variable(pc_model_W).cuda()

            img = Variable(img).cuda()
            depth_vel = Variable(depth_vel).cuda()
            depth_vel = depth_vel.permute(0, 3, 1, 2).contiguous()

            choose = torch.LongTensor([0])
            choose = Variable(choose).cuda()
            idx = Variable(idx).cuda()

            modelPoints = Variable(modelPoints).cuda()
            modelPointsGT = Variable(modelPointsGT).cuda()

            img[:,0:3,:,:] = img[:,0:3,:,:] * RGBEnable
            img[:,3,:,:] = img[:,3,:,:] * Depth1Enable

            with torch.no_grad():
                if self.option == 1:
                    pred_r, pred_t, pred_c, _ = self.estimator(img, depth_vel*Depth2Enable, target*PC1Enable, model_gt*PC2Enable, choose, idx)
                elif self.option == 2:
                    pred_r, pred_t, pred_c, _ = self.estimator(img, depth_vel*Depth2Enable, target*PC1Enable, velodyne_gt*PC2Enable, choose, idx)
                elif self.option == 3:
                    pred_r, pred_t, pred_c, _ = self.estimator(img, depth_vel*Depth2Enable, model_gt*PC1Enable, velodyne_gt*PC2Enable, choose, idx)

            new_file_names = []
            for f in file_name:
                # Pegar o caminho relativo após "Anot Perdiz/results/"
                rel_path = os.path.relpath(f, '/home/goncalo/Manuel/DATASETS/Anot Perdiz/results')
                # Construir novo caminho na nova pasta
                new_f = os.path.join(new_base_folder, rel_path)
                # Garantir que a pasta existe
                os.makedirs(os.path.dirname(new_f), exist_ok=True)
                new_file_names.append(new_f)
        
            for i in range(rt.shape[0]):
                T = self.computeT(pred_r[i], pred_t[i])
                T = T.detach().cpu().numpy()

                # print(new_file_names[i], T)

                # guardar no ficheiro
                with open(new_file_names[i], 'w') as f:
                    for row in T:
                        formatted_row = ' '.join(f"{val:.5f}" for val in row)
                        f.write(formatted_row + " ; ")

    def computeT(self, pred_r, pred_t):
        bs = 1
        num_p = 1

        a = torch.norm(pred_r, dim=0)
        if a>0.001:
            pred_r = pred_r / a
        else:
            pred_r[3]=1
        
        x, y, z, w = pred_r
        rot = torch.tensor([
            [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)]
        ], device=pred_r.device)

        trans = pred_t.view(3, 1)

        upper = torch.cat([rot, trans], dim=1)
        bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=pred_r.device)
        transform = torch.cat([upper, bottom], dim=0)

        return transform

    def target_vs_pred(self):
        self.estimator.eval()

        for j, data in tqdm(enumerate(self.runloader, 0), total=len(self.runloader), unit='batch'):

            pc_depth_W, pc_depth, pc_velodyne_W, pc_velodyne, pc_model_W, pc_model, img, depth_vel, modelPoints, modelPointsGT, rt, idx, file_name = data
            print(file_name)

            if self.modalities == 0:
                RGBEnable = float(1)
                Depth1Enable = float(1)
                Depth2Enable = float(1)
                PC1Enable = float(1)
                PC2Enable = float(1)
            elif self.modalities == 1:
                RGBEnable = float(1)
                Depth1Enable = float(0)
                Depth2Enable = float(0)
                PC1Enable = float(0)
                PC2Enable = float(0)
            elif self.modalities == 2:
                RGBEnable = float(1)
                Depth1Enable = float(1)
                Depth2Enable = float(0)
                PC1Enable = float(0)
                PC2Enable = float(0)
            elif self.modalities == 3:
                RGBEnable = float(1)
                Depth1Enable = float(1)
                Depth2Enable = float(0)
                PC1Enable = float(1)
                PC2Enable = float(0)

            points = Variable(pc_depth).cuda()  # cam
            target = Variable(pc_depth_W).cuda()
            velodyne = Variable(pc_velodyne).cuda()
            velodyne_gt = Variable(pc_velodyne_W).cuda()
            model = Variable(pc_model).cuda()
            model_gt = Variable(pc_model_W).cuda()

            img = Variable(img).cuda()
            depth_vel = Variable(depth_vel).cuda()
            depth_vel = depth_vel.permute(0, 3, 1, 2).contiguous()

            choose = torch.LongTensor([0])
            choose = Variable(choose).cuda()        
            idx = Variable(idx).cuda()
            
            modelPoints = Variable(modelPoints).cuda()
            modelPointsGT = Variable(modelPointsGT).cuda()

            img[:,0:3,:,:] = img[:,0:3,:,:] * RGBEnable
            img[:,3,:,:] = img[:,3,:,:] * Depth1Enable

            with torch.no_grad():
                if self.option == 1:
                    pred_r, pred_t, pred_c, _ = self.estimator(img, depth_vel*Depth2Enable, target*PC1Enable, model_gt*PC2Enable, choose, idx)
                elif self.option == 2:
                    pred_r, pred_t, pred_c, _ = self.estimator(img, depth_vel*Depth2Enable, target*PC1Enable, velodyne_gt*PC2Enable, choose, idx)
                elif self.option == 3:
                    pred_r, pred_t, pred_c, _ = self.estimator(img, depth_vel*Depth2Enable, model_gt*PC1Enable, velodyne_gt*PC2Enable, choose, idx)

                loss_add, dis, new_points, new_target = self.criterion[0](pred_r, pred_t, pred_c, modelPoints, modelPointsGT, idx, velodyne_gt, self.opt.w, self.opt.refine_start)
                loss_centro, _, _, _ = self.criterion[0](pred_r, pred_t, pred_c, modelPoints, modelPointsGT, idx, velodyne_gt, self.opt.w, self.opt.refine_start, pontocentro=True)
                loss_geo = self.criterion[1](rt, pred_r)

                print(f"loss ADD: {loss_add:.4f} \t loss GEO: {loss_geo:.4f} \t loss centro: {loss_centro:.4f}")

            bs = pred_r.shape[0]
            for i in range(bs):
                T = self.computeT(pred_r[i], pred_t[i])
                print("Matriz Pred:")
                print(T)
                print("Matriz GT:")
                print(rt)

                rt_inv = np.linalg.inv(T.detach().cpu().numpy())
                
                rotation_inv = rt_inv[:3, :3]
                translation_inv = rt_inv[:3, 3]
                
                pred_pc_depth = (rotation_inv @ pc_depth_W[0].cpu().numpy().T).T + translation_inv
                pred_model_points = (rotation_inv @ modelPointsGT[0].cpu().numpy().T).T + translation_inv

                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')

                modelPointsGT = modelPointsGT.squeeze(0)
                modelPoints = modelPoints.squeeze(0)

                for i in range(1, len(modelPoints)):
                    x1 = [modelPointsGT.detach().cpu().numpy()[0, 0], modelPointsGT.detach().cpu().numpy()[i, 0]]
                    y1 = [modelPointsGT.detach().cpu().numpy()[0, 1], modelPointsGT.detach().cpu().numpy()[i, 1]]
                    z1 = [modelPointsGT.detach().cpu().numpy()[0, 2], modelPointsGT.detach().cpu().numpy()[i, 2]]

                    x_pred = [pred_model_points[0, 0], pred_model_points[i, 0]]
                    y_pred = [pred_model_points[0, 1], pred_model_points[i, 1]]
                    z_pred = [pred_model_points[0, 2], pred_model_points[i, 2]]

                    x1a = [modelPoints.detach().cpu().numpy()[0, 0], modelPoints.detach().cpu().numpy()[i, 0]]
                    y1a = [modelPoints.detach().cpu().numpy()[0, 1], modelPoints.detach().cpu().numpy()[i, 1]]
                    z1a = [modelPoints.detach().cpu().numpy()[0, 2], modelPoints.detach().cpu().numpy()[i, 2]]

                    ax.plot(x1, y1, z1, color='blue', linewidth=1)
                    ax.plot(x_pred, y_pred, z_pred, color='orange', linewidth=1)
                    ax.plot(x1a, y1a, z1a, color='red', linewidth=1)

                ax.text(x1[1], y1[1], z1[1], "World", color='blue', fontsize=7)
                ax.text(x_pred[1], y_pred[1], z_pred[1], "Pred", color='orange', fontsize=7)
                ax.text(x1a[1], y1a[1], z1a[1], "Origin", color='red', fontsize=7)

                pc_depth_W = pc_depth_W.squeeze(0)
                pc_depth = pc_depth.squeeze(0)

                """ax.scatter(
                    pc_depth_W.detach().cpu().numpy()[:, 0],
                    pc_depth_W.detach().cpu().numpy()[:, 1],
                    pc_depth_W.detach().cpu().numpy()[:, 2],
                    s=1,
                    alpha=0.5,
                    color="blue",
                    label='pointcloud depth'
                )

                ax.scatter(
                    pred_pc_depth[:, 0],
                    pred_pc_depth[:, 1],
                    pred_pc_depth[:, 2],
                    s=1,
                    alpha=0.5,
                    color="orange",
                    label='pointcloud depth pred'
                )

                ax.scatter(
                    pc_depth.detach().cpu().numpy()[:, 0],
                    pc_depth.detach().cpu().numpy()[:, 1],
                    pc_depth.detach().cpu().numpy()[:, 2],
                    s=1,
                    alpha=0.5,
                    color="red",
                    label='pointcloud depth do GT'
                )"""

                ax.set_title('PointCloud + Eixos do Modelo no mundo')
                ax.legend(loc='upper left')

                plt.show()

            break

    def main(self):
       # self.view_results()
       self.target_vs_pred()
