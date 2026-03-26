import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torchvision.models import resnet34
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "torchsparsemaster"))
import torchsparse
import torchsparse.nn as spnn
from torchsparse import SparseTensor

class PoseNet(nn.Module):
    def __init__(self, modelRGB, modelDepth, modelDepth2, num_points, num_obj, embchannels, noise=0.0025):
        super(PoseNet, self).__init__()
        self.num_points = num_points
        self.noise = noise

        self.cnn = RGBDFeatureExtractor2CUSTOM(modelRGB,modelDepth, modelDepth2,num_heads=4,emb=embchannels)
        
        self.compressr2 = torch.nn.Linear(4096, num_obj*4)
        self.compresst2 = torch.nn.Linear(4096, num_obj*3)
        self.num_obj = num_obj
        self.attn = TransformerEncoder(512, 4)
        self.net2 = Conv1DNetwork(50)
        self.net1 = Conv1DNetwork(3)
        self.net3 = Conv1DNetwork(3)
        self.netFinal = Conv1DNetwork(22)
        replace_batchnorm_with_instancenorm(self.cnn)
        replace_batchnorm_with_instancenorm(self.attn)

    def embed_fn(self, x, L_embed=6):
        rets = []
        rets.append(x[0])
        rets.append(x[1])
        rets.append(x[2])
        for i in range(L_embed):
            for fn in [np.sin, np.cos]:
                a=fn(2.**i * x)
                rets.append(a[0])
                rets.append(a[1])
                rets.append(a[2])                
        return rets 

    def forward(self, img, depth_vel, x,velodyne, choose, obj):
        # print("Dentro rede ------------------")
        batch = img.shape[0]
 
        rgb = img[:, 0:3, :, :]
        depth = img[:, 3, :, :].unsqueeze(0).permute(1, 0, 2, 3)
        
        # print("img.shape", rgb.shape)
        # print("depth.shape", depth.shape)
        # print("depth_vel", depth_vel.shape)

        out_img = self.cnn(rgb, depth, depth_vel)
        # print("out img", out_img.shape)

        bs, di, _ = out_img.size()

        emb = out_img.view(di, bs, -1)
        emb = F.adaptive_avg_pool2d(emb, (50,500))
        emb = emb.view(di, 50, 500)
        # print("emb.shape", emb.shape)

        x = x.transpose(2, 1).contiguous()
        velodyne = velodyne.transpose(2, 1).contiguous()

        ap_x = self.net1(x).contiguous()
        # print("net1: ", ap_x.shape)

        bs, di, _ = ap_x.size()

        ap_x = ap_x.view(di, bs, 512)
        ap_x = F.adaptive_avg_pool2d(ap_x, (15, 512))
        ap_x = ap_x.view(15, di, 512)
        # print("PC vector Adapt: ", ap_x.shape)

        ap_x2 = self.net3(velodyne).contiguous()
        # print("net3: ", ap_x2.shape)

        ap_x2 = ap_x2.view(di, bs, 512)
        ap_x2 = F.adaptive_avg_pool2d(ap_x2, (15,512))
        ap_x2 = ap_x2.view(15, di, 512)
        # print("PC2 vector Adapt: ",ap_x2.shape)

        ap_x = ap_x + ap_x2

        ap_y = self.net2(emb).contiguous()
        # print("net2", ap_y.shape)
        
        # MUDAR 20/25
        ap_x, ap_y = self.attn(F.dropout(ap_x, p=self.noise),F.dropout(ap_y, p=self.noise))

        ap_x = ap_x.permute(1, 0, 2) 
        ap_y = ap_y.permute(1, 0, 2) 

        ap_x = torch.cat([ap_x, ap_y], 1)
        ap_x = self.netFinal(F.dropout(ap_x, p=self.noise))
        ap_x = ap_x.permute(1, 0, 2)
        ap_x = ap_x.flatten(start_dim=1)
        rx = F.tanh(self.compressr2(F.dropout(ap_x, p=self.noise)))
        tx = (self.compresst2(F.dropout(ap_x, p=self.noise)))

        rx = rx.view(batch, self.num_obj, 4)
        tx = tx.view(batch, self.num_obj, 3)

        # print("rx.shape", rx.shape)
        # print("obj", obj)
        # print(rx)
        # print("tx.shape", tx.shape)

        if self.num_obj != 1:
            batch_indices = torch.arange(rx.size(0)).cuda()

            rx = rx[batch_indices, obj]
            tx = tx[batch_indices, obj]
        else:
            rx = rx.squeeze(1)
            tx = tx.squeeze(1)

        # print("rx.shape", rx.shape)
        # print(rx)

        # print("tx.shape", tx.shape)

        # print("Dentro rede ------------------")

        return rx, tx, None, emb.detach()
    
class PoseNetGarrote(nn.Module):
    def __init__(self, modelRGB, modelDepth, modelDepth2, num_points, num_obj, embchannels, noise=0.25):
        super(PoseNetGarrote, self).__init__()
        self.num_points = num_points
        self.noise = noise

        self.cnn = RGBDFeatureExtractor2CUSTOM(modelRGB,modelDepth, modelDepth2,num_heads=4,emb=embchannels)
        
        # self.compressr2 = torch.nn.Linear(4096, num_obj*4)
        # self.compresst2 = torch.nn.Linear(4096, num_obj*3)
        self.compressr2=nn.Sequential(
            nn.Linear(1536, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_obj*4)
        )


        self.compresst2=nn.Sequential(
            nn.Linear(1536, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_obj*3)
        )
        self.num_obj = num_obj
        self.attn = TransformerEncoder3(512,512,512, 4)
        self.net1 = PointResNet(3,128)
        self.net3 = PointResNet(3,128)
        replace_batchnorm_with_instancenorm(self.cnn)
        replace_batchnorm_with_instancenorm(self.attn)

    def embed_fn(self, x, L_embed=6):
        rets = []
        rets.append(x[0])
        rets.append(x[1])
        rets.append(x[2])
        for i in range(L_embed):
            for fn in [np.sin, np.cos]:
                a=fn(2.**i * x)
                rets.append(a[0])
                rets.append(a[1])
                rets.append(a[2])                
        return rets 

    def forward(self, img, depth_vel, x,velodyne, choose, obj):
        # print("Dentro rede ------------------")
        batch = img.shape[0]
 
        rgb = img[:, 0:3, :, :]
        depth = img[:, 3, :, :].unsqueeze(0).permute(1, 0, 2, 3)

        out_img = self.cnn(rgb, depth, depth_vel)
        bs, di, _ = out_img.size()
         #print(out_img.shape)

        emb = out_img.view(di, bs, -1)
        #emb = F.adaptive_avg_pool2d(emb, (7,512))
        #emb = emb.view(di,  7,512)
        #print("emb",emb.shape)

        x = x.transpose(2, 1).contiguous()
        velodyne = velodyne.transpose(2, 1).contiguous()
        #print("x",x.shape)

        ap_x = self.net1(x).contiguous()

        #bs, di, _ = ap_x.size()

        #ap_x = ap_x.view(di, bs, 512)
        #ap_x = F.adaptive_avg_pool2d(ap_x, (7, 512))
        #ap_x = ap_x.view(7, di, 512)

        ap_x2 = self.net3(velodyne).contiguous()

        #ap_x2 = ap_x2.view(di, bs, 512)
        #ap_x2 = F.adaptive_avg_pool2d(ap_x2, (7,512))
        #ap_x2 = ap_x2.view(7, di, 512)
        #print("ap_x",ap_x.shape,"ap_x2",ap_x2.shape)

        #ap_x = torch.cat([ap_x, ap_x2], 0)

        ap_x = F.adaptive_max_pool2d(ap_x, (128,512))
        ap_x = ap_x.view(-1,di, 512)
        ap_x2 = F.adaptive_max_pool2d(ap_x2, (128,512))
        ap_x2 = ap_x2.view(-1,di, 512)
        emb = emb.view(-1,di, 512)

        #ap_y = self.net2(emb).contiguous()
        ap_x, ap_y,emb = self.attn(F.dropout(ap_x, p=0.00001),F.dropout(ap_x2, p=0.00001),F.dropout(emb, p=0.000001))

        ap_x = ap_x.permute(1, 0, 2) 
        ap_x = F.adaptive_max_pool2d(ap_x, (1,512))

        ap_y = ap_y.permute(1, 0, 2) 
        ap_y = F.adaptive_max_pool2d(ap_y, (1,512))

        emb = emb.permute(1, 0, 2) 
        emb = F.adaptive_max_pool2d(emb, (1,512))
 

        #print(ap_x.shape,ap_y.shape,emb.shape)
        ap_x = torch.cat([ap_x.flatten(start_dim=1), ap_y.flatten(start_dim=1),emb.flatten(start_dim=1)], 1)
        #print(ap_x.shape,ap_y.shape,emb.shape)
        #ap_x = ap_x.permute(0, 2, 1)
        #ap_x = F.adaptive_avg_pool1d(ap_x, 8)       # [64, 512, 8]
        ap_x = ap_x.flatten(start_dim=1)

        rx = F.tanh(self.compressr2(F.dropout(ap_x, p=self.noise)))
        tx = (self.compresst2(F.dropout(ap_x, p=self.noise)))

        rx = rx.view(batch, self.num_obj, 4)
        tx = tx.view(batch, self.num_obj, 3)

        if self.num_obj != 1:
            batch_indices = torch.arange(rx.size(0)).cuda()

            rx = rx[batch_indices, obj]
            tx = tx[batch_indices, obj]
        else:
            rx = rx.squeeze(1)
            tx = tx.squeeze(1)

        return rx, tx, None, emb.detach()

class PoseNet_TorchSparse(nn.Module):
    def __init__(self, modelRGB, modelDepth, modelDepth2, num_points, num_obj, embchannels, noise=0.25):
        super(PoseNet_TorchSparse, self).__init__()
        self.num_points = num_points
        self.noise = noise

        self.cnn = RGBDFeatureExtractor2CUSTOM(modelRGB,modelDepth, modelDepth2,num_heads=4,emb=embchannels)
        
        # self.compressr2 = torch.nn.Linear(4096, num_obj*4)
        # self.compresst2 = torch.nn.Linear(4096, num_obj*3)
        self.compressr2=nn.Sequential(
            nn.Linear(1536, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_obj*4)
        )

        self.compresst2=nn.Sequential(
            nn.Linear(1536, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_obj*3)
        )

        self.num_obj = num_obj
        self.attn = TransformerEncoder3(512, 512, 512, 4)
        self.net1 = PointResNet_TorchSparse(3, 128)
        self.net3 = PointResNet_TorchSparse(3, 128)
        replace_batchnorm_with_instancenorm(self.cnn)
        replace_batchnorm_with_instancenorm(self.attn)

    def embed_fn(self, x, L_embed=6):
        rets = []
        rets.append(x[0])
        rets.append(x[1])
        rets.append(x[2])
        for i in range(L_embed):
            for fn in [np.sin, np.cos]:
                a=fn(2.**i * x)
                rets.append(a[0])
                rets.append(a[1])
                rets.append(a[2])                
        return rets 

    def forward(self, img, depth_vel, x, velodyne, choose, obj):
        # print("Dentro rede ------------------")
        batch = img.shape[0]
 
        rgb = img[:, 0:3, :, :]
        depth = img[:, 3, :, :].unsqueeze(0).permute(1, 0, 2, 3)

        out_img = self.cnn(rgb, depth, depth_vel)
        bs, di, _ = out_img.size()
         #print(out_img.shape)

        emb = out_img.view(di, bs, -1)
        #emb = F.adaptive_avg_pool2d(emb, (7,512))
        #emb = emb.view(di,  7,512)
        #print("emb",emb.shape)

        x = x.transpose(2, 1).contiguous()
        velodyne = velodyne.transpose(2, 1).contiguous()
        print("x", x.shape)
        print("velodyne", velodyne.shape)

        ap_x = self.net1(x).contiguous()

        #bs, di, _ = ap_x.size()

        #ap_x = ap_x.view(di, bs, 512)
        #ap_x = F.adaptive_avg_pool2d(ap_x, (7, 512))
        #ap_x = ap_x.view(7, di, 512)

        ap_x2 = self.net3(velodyne).contiguous()

        #ap_x2 = ap_x2.view(di, bs, 512)
        #ap_x2 = F.adaptive_avg_pool2d(ap_x2, (7,512))
        #ap_x2 = ap_x2.view(7, di, 512)
        #print("ap_x",ap_x.shape,"ap_x2",ap_x2.shape)

        #ap_x = torch.cat([ap_x, ap_x2], 0)

        ap_x = F.adaptive_max_pool2d(ap_x, (128,512))
        ap_x = ap_x.view(-1,di, 512)
        ap_x2 = F.adaptive_max_pool2d(ap_x2, (128,512))
        ap_x2 = ap_x2.view(-1,di, 512)
        emb = emb.view(-1,di, 512)

        #ap_y = self.net2(emb).contiguous()
        ap_x, ap_y,emb = self.attn(F.dropout(ap_x, p=0.00001),F.dropout(ap_x2, p=0.00001),F.dropout(emb, p=0.000001))

        ap_x = ap_x.permute(1, 0, 2) 
        ap_x = F.adaptive_max_pool2d(ap_x, (1,512))

        ap_y = ap_y.permute(1, 0, 2) 
        ap_y = F.adaptive_max_pool2d(ap_y, (1,512))

        emb = emb.permute(1, 0, 2) 
        emb = F.adaptive_max_pool2d(emb, (1,512))
 

        #print(ap_x.shape,ap_y.shape,emb.shape)
        ap_x = torch.cat([ap_x.flatten(start_dim=1), ap_y.flatten(start_dim=1),emb.flatten(start_dim=1)], 1)
        #print(ap_x.shape,ap_y.shape,emb.shape)
        #ap_x = ap_x.permute(0, 2, 1)
        #ap_x = F.adaptive_avg_pool1d(ap_x, 8)       # [64, 512, 8]
        ap_x = ap_x.flatten(start_dim=1)

        rx = F.tanh(self.compressr2(F.dropout(ap_x, p=self.noise)))
        tx = (self.compresst2(F.dropout(ap_x, p=self.noise)))

        rx = rx.view(batch, self.num_obj, 4)
        tx = tx.view(batch, self.num_obj, 3)

        if self.num_obj != 1:
            batch_indices = torch.arange(rx.size(0)).cuda()

            rx = rx[batch_indices, obj]
            tx = tx[batch_indices, obj]
        else:
            rx = rx.squeeze(1)
            tx = tx.squeeze(1)

        return rx, tx, None, emb.detach()
 

class RGBDFeatureExtractor2CUSTOM(nn.Module):
    def __init__(self, modelRGB,modelDepth, modelDepth2, num_heads=4, emb=512):
        super(RGBDFeatureExtractor2CUSTOM, self).__init__()
        self.encoder = RGBDEncoder2CUSTOM(modelRGB,modelDepth, modelDepth2)
        self.attention_fusion = MultiHeadAttentionFusion2(embed_dim=emb, num_heads=num_heads)

    def forward(self, rgb, depth, depth2):
        # Encode RGB and Depth images
        rgb_features,depth_features,depth2_features = self.encoder(rgb, depth,depth2)

        fused_features = self.attention_fusion(rgb_features, depth_features, depth2_features)

        return fused_features

class RGBDEncoder2CUSTOM(nn.Module):
    def __init__(self, modelRGB,modelDepth, modelDepth2):
        super(RGBDEncoder2CUSTOM, self).__init__()
        self.rgb_backbone = modelRGB
        self.depth_backbone = modelDepth
        self.depth_backbone2 = modelDepth2

    def forward(self, rgb, depth,depth2):
        rgb_features = self.rgb_backbone(rgb)
        depth_features = self.depth_backbone(depth)
        depth_features2 = self.depth_backbone2(depth2)
        # print("RGBDEncoder2", rgb_features.shape,depth_features.shape,depth_features2.shape)
                
        return rgb_features,depth_features,depth_features2

class MultiHeadAttentionFusion2(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionFusion2, self).__init__()
        self.attention = MultiheadAttention(embed_dim, num_heads)
        self.attention1 = MultiheadAttention(embed_dim, num_heads)
        self.attention2 = MultiheadAttention(embed_dim, num_heads)

    def forward(self, rgb_features, depth_features,depth2_features):
        # Flatten the spatial dimensions for multi-head attention
        # print(rgb_features.shape)
        rgb_flat = rgb_features.flatten(2).permute(2, 0, 1)
        depth_flat = depth_features.flatten(2).permute(2, 0, 1)
        depth2_flat = depth2_features.flatten(2).permute(2, 0, 1)
        fused_features1, _ = self.attention(rgb_flat, rgb_flat, rgb_flat)
        fused_features2, _ = self.attention1(depth_flat, depth_flat, depth_flat)
        fused_features3, _ = self.attention2(depth2_flat, depth2_flat, depth2_flat)

        return torch.cat([fused_features1, fused_features2, fused_features3], 0)

class TransformerEncoder(nn.Module):
    def __init__(self, n_features, n_heads):
        super(TransformerEncoder, self).__init__()

        self.norm       = nn.LayerNorm(n_features)
        self.norm2       = nn.LayerNorm(n_features)
        self.attention  = nn.MultiheadAttention(embed_dim= n_features, num_heads = n_heads)
        self.attention2  = nn.MultiheadAttention(embed_dim= n_features, num_heads = n_heads)
        self.fc         = nn.Linear(in_features = n_features, out_features = n_features, bias = True)
        self.fc2         = nn.Linear(in_features = n_features, out_features = n_features, bias = True)

    def forward(self, x,y):
        x1   = torch.clone(x)
        x    = self.norm(x)
        x, _ = self.attention(x,x,x) 
        x    = x + x1 
        x1   = torch.clone(x)
        x    = self.norm(x)
        x    = F.relu(self.fc(x))

        y1   = torch.clone(y)
        y    = self.norm2(y)
        y, _ = self.attention2(y,y,y) 
        y    = y + y1 
        y1   = torch.clone(y)
        y    = self.norm2(y)
        y    = F.relu(self.fc2(y))

        return x+x1,y+y1
    
class TransformerEncoder3(nn.Module):
    def __init__(self,n_featuresPC1,n_featuresPC2, n_features, n_heads):
        super(TransformerEncoder3, self).__init__()

        self.norm       = nn.LayerNorm(n_featuresPC1)
        self.norm2       = nn.LayerNorm(n_featuresPC2)
        self.norm3       = nn.LayerNorm(n_features)
        self.attention  = nn.MultiheadAttention(embed_dim= n_featuresPC1, num_heads = n_heads)
        self.attention2  = nn.MultiheadAttention(embed_dim= n_featuresPC2, num_heads = n_heads)
        self.attention3  = nn.MultiheadAttention(embed_dim= n_features, num_heads = n_heads)
        self.fc         = nn.Linear(in_features = n_featuresPC1, out_features = n_featuresPC1, bias = True)
        self.fc2         = nn.Linear(in_features = n_featuresPC2, out_features = n_featuresPC2, bias = True)
        self.fc3         = nn.Linear(in_features = n_features, out_features = n_features, bias = True)


    def forward(self, x,y,z):
        x1   = torch.clone(x)
        x    = self.norm(x)
        x, _ = self.attention(x,x,x) 
        x    = x + x1 
        x1   = torch.clone(x)
        x    = self.norm(x)
        x    = F.relu(self.fc(x))



        y1   = torch.clone(y)
        y    = self.norm2(y)
        y, _ = self.attention2(y,y,y) 
        y    = y + y1 
        y1   = torch.clone(y)
        y    = self.norm2(y)
        y    = F.relu(self.fc2(y))


        z1   = torch.clone(z)
        z    = self.norm2(z)
        z, _ = self.attention2(z,z,z) 
        z    = z + z1 
        z1   = torch.clone(z)
        z    = self.norm2(z)
        z    = F.relu(self.fc2(z))

        return x+x1,y+y1,z+z1

class RGBDEncoder(nn.Module):
    def __init__(self):
        super(RGBDEncoder, self).__init__()
        self.rgb_backbone = resnet34(pretrained=True)
        self.depth_backbone = resnet34(pretrained=True)
        
        # Remove the fully connected layers
        self.rgb_backbone = nn.Sequential(*list(self.rgb_backbone.children())[:-2])
        self.depth_backbone = nn.Sequential(*list(self.depth_backbone.children())[:-2])
        self.depth_backbone[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    def forward(self, rgb, depth):
        #print(rgb.shape,depth.shape)
        rgb_features = self.rgb_backbone(rgb)
        depth_features = self.depth_backbone(depth)
        
        #print(rgb_features.shape,depth_features.shape)
                
        return rgb_features,depth_features

class RGBDEncoder2(nn.Module):
    def __init__(self):
        super(RGBDEncoder2, self).__init__()
        self.rgb_backbone = resnet34(pretrained=True)
        self.depth_backbone = resnet34(pretrained=True)
        self.depth_backbone2 = resnet34(pretrained=True)

        # Remove the fully connected layers
        self.rgb_backbone = nn.Sequential(*list(self.rgb_backbone.children())[:-2])
        self.depth_backbone = nn.Sequential(*list(self.depth_backbone.children())[:-2])
        self.depth_backbone[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.depth_backbone2 = nn.Sequential(*list(self.depth_backbone2.children())[:-2])
        self.depth_backbone2[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    def forward(self, rgb, depth,depth2):
        rgb_features = self.rgb_backbone(rgb)
        depth_features = self.depth_backbone(depth)
        depth_features2 = self.depth_backbone2(depth2)
                
        return rgb_features,depth_features,depth_features2

class Conv1DNetwork(nn.Module):
    def __init__(self, in_channels_):
        super(Conv1DNetwork, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels_, out_channels=16, kernel_size=3,  padding=1)
        self.conv1b = nn.Conv1d(in_channels=in_channels_, out_channels=16, kernel_size=3, padding=1)
        self.conv1a = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm1d(16)
        self.bn1a = nn.InstanceNorm1d(16)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,  padding=1)
        self.conv2b = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,  padding=1)
        self.conv2a = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.InstanceNorm1d(32)
        self.bn2a = nn.InstanceNorm1d(32)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,  padding=1)
        self.conv3b = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3a = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.InstanceNorm1d(64)
        self.bn3a = nn.InstanceNorm1d(64)


        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4b = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4a = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.InstanceNorm1d(128)
        self.bn4a = nn.InstanceNorm1d(128)


        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3,  padding=1)
        self.conv5b = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv5a = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.InstanceNorm1d(256)
        self.bn5a = nn.InstanceNorm1d(256)


        self.conv6 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,  padding=1)
        self.conv6b = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,  padding=1)
        self.conv6a = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn6 = nn.InstanceNorm1d(512)
        self.bn6a = nn.InstanceNorm1d(512)
        
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        
#https://d2l.ai/chapter_convolutional-modern/resnet.html
    def forward(self, x):
        x1 = self.conv1b(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1a(x)
        x = self.bn1a(x)
        x=x+x1;
        x = F.relu(x)
        x = self.pool(x)


        x1 = self.conv2b(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2a(x)
        x = self.bn2a(x)
        x=x+x1;
        x = F.relu(x)
        x = self.pool(x)

        x1 = self.conv3b(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv3a(x)
        x = self.bn3a(x)
        x=x+x1;
        x = F.relu(x)
        x = self.pool(x)

        x1 = self.conv4b(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv4a(x)
        x = self.bn4a(x)
        x=x+x1;
        x = F.relu(x)
        x = self.pool(x)

        x1 = self.conv5b(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv5a(x)
        x = self.bn5a(x)
        x=x+x1;
        x = F.relu(x)
        x = self.pool(x)

        x1 = self.conv6b(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.conv6a(x)
        x = self.bn6a(x)
        x=x+x1;
        x = F.relu(x)
        x = self.pool(x)
        

        x = x.permute(2, 0, 1)  # Shape: [sequence_length, batch_size, embedding_dim]
        
        return x
    
class PointResNet(nn.Module):
    def __init__(self, input_dim=3, feature_dim=1024):
        super(PointResNet, self).__init__()
        self.layer1 = ResNet1DBlock(input_dim, 64)
        self.layer2 = ResNet1DBlock(64, 128)
        self.layer3 = ResNet1DBlock(128, 256)
        self.layer4 = ResNet1DBlock(256, feature_dim)

    def forward(self, x):
        # x: [B, N, 3]  (batch, num_points, features)
        #x = x.transpose(2, 1)     # → [B, 3, N]

        x = self.layer1(x)        # [B, 64, N]
        x = self.layer2(x)        # [B, 128, N]
        x = self.layer3(x)        # [B, 256, N]
        x = self.layer4(x)        # [B, feature_dim, N]

        # global feature (like PointNet)
        #x = torch.max(x, dim=2)[0]   # [B, feature_dim]
        return x
    
class ResNet1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet1DBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)

        # skip connection
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        return F.relu(out)
    
class PointResNet_TorchSparse(nn.Module):
    def __init__(self, input_dim=3, feature_dim=1024):
        super(PointResNet_TorchSparse, self).__init__()
        self.layer1 = SparseBlock(input_dim, 64)
        self.layer2 = SparseBlock(64, 128)
        self.layer3 = SparseBlock(128, 256)
        self.layer4 = SparseBlock(256, feature_dim)

    def forward(self, x):
        # x: [B, N, 3]  (batch, num_points, features)
        #x = x.transpose(2, 1)     # → [B, 3, N]

        x = self.layer1(x)        # [B, 64, N]
        x = self.layer2(x)        # [B, 128, N]
        x = self.layer3(x)        # [B, 256, N]
        x = self.layer4(x)        # [B, feature_dim, N]

        # global feature (like PointNet)
        # x = torch.max(x, dim=2)[0]   # [B, feature_dim]
        return x
    
class SparseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SparseBlock, self).__init__()
        self.conv = spnn.SubMConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # self.bn = spnn.BatchNorm(out_channels)
        self.bn = spnn.InstanceNorm(out_channels)
        self.act = spnn.ReLU()

    def forward(self, x):
        out = self.act(self.bn(self.conv(x)))
        return out

def replace_batchnorm_with_instancenorm(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            # Replace with InstanceNorm2d
            instance_norm = nn.InstanceNorm2d(module.num_features, affine=True)
            setattr(model, name, instance_norm)
        elif isinstance(module, nn.Module):
            replace_batchnorm_with_instancenorm(module)

class PoseNetMultiCUSTOM(nn.Module):
    def __init__(self, modelRGB, modelDepth, modelDepth2, num_points, num_obj,embchannels):
        super(PoseNetMultiCUSTOM, self).__init__()
        self.num_points = num_points

        self.cnn = RGBDFeatureExtractor2CUSTOM(modelRGB,modelDepth, modelDepth2,num_heads=4,emb=embchannels)
        
        self.compressr2=torch.nn.Linear(4096, num_obj*4)
        self.compresst2=torch.nn.Linear(4096, num_obj*3)
        self.num_obj = num_obj
        self.attn = TransformerEncoder(512, 4); 
        self.net2=Conv1DNetwork(50);
        self.net1=Conv1DNetwork(3);
        self.net3=Conv1DNetwork(3);
        self.netFinal=Conv1DNetwork(22);
        replace_batchnorm_with_instancenorm(self.cnn)
        replace_batchnorm_with_instancenorm(self.attn)

    def embed_fn(self, x, L_embed=6):
        rets = []
        rets.append(x[0])
        rets.append(x[1])
        rets.append(x[2])
        for i in range(L_embed):
            for fn in [np.sin, np.cos]:
                a=fn(2.**i * x)
                rets.append(a[0])
                rets.append(a[1])
                rets.append(a[2])                
        return rets 

    def forward(self, img, depth_vel, x,velodyne, choose, obj):
        #print(img)
        depth = img[:,3,:,:].unsqueeze(0)
        depth_vel = depth_vel.squeeze(3)
        depth_vel = depth_vel.unsqueeze(0)

        #print(depth.shape)
        #print(depth_vel.shape)

        out_img = self.cnn(img[:,0:3,:,:],depth,depth_vel)
        #print(out_img.shape)
        bs, di, _ = out_img.size()

        emb = out_img.view(di, bs, -1)
        emb=F.adaptive_avg_pool2d(emb, (50,500))
        emb = emb.view(1, 50, 500)

        x = x.transpose(2, 1).contiguous()
        velodyne = velodyne.transpose(2, 1).contiguous()

        ap_x = self.net1(x)
        #print("PC vector: ",ap_x.shape)
        ap_x2=self.net3(velodyne);
        #print("PC2 vector: ",ap_x.shape)
        ap_x = ap_x+ap_x2
        ap_y = self.net2(emb)
        ap_x,ap_y= self.attn(F.dropout(ap_x, p=0.0025),F.dropout(ap_y, p=0.0025))

        ap_x = ap_x.permute(1, 0, 2) 
        ap_y = ap_y.permute(1, 0, 2) 
        ap_x=torch.cat([ap_x, ap_y], 1)
        ap_x=self.netFinal(F.dropout(ap_x, p=0.0025))
        ap_x=ap_x.flatten()
        rx= F.tanh(self.compressr2(F.dropout(ap_x, p=0.005)))
        tx= (self.compresst2(F.dropout(ap_x, p=0.005)))

        rx = rx.view(1, self.num_obj, 4)
        tx = tx.view(1, self.num_obj, 3)
      
        b = 0
        rx = torch.index_select(rx[b], 0, obj[b])
        tx = torch.index_select(tx[b], 0, obj[b])

        return rx.flatten(), tx.flatten(), None, emb.detach()