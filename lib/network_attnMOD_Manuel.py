import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
from lib.pspnet_attn import PSPNet, PSPNetH, PSPNet4C, PSPNetH2b, PSPNetH2v2, PSPNetH2v2a, PSPNetH2v2b
from einops import rearrange

import inspect
__LINE__ = inspect.currentframe()

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


# psp_modelsH = {
#     'resnet18': lambda: PSPNetH(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
#     'resnet34': lambda: PSPNetH(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
#     'resnet50': lambda: PSPNetH(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
#     'resnet101': lambda: PSPNetH(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
#     'resnet152': lambda: PSPNetH(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
# }


#considering 1 epoch -> PSPNetH2a(0.027) better than PSPNetH2 (0.04)
#  PSPNetH2b ->0.02438354686590631                    p1 = rearrange(p1, 'd0 d1 d2 d3 -> d0 (d2 d3)  d1')

#  PSPNetH2b -> 0.05          p1 = rearrange(p1, 'd0 d1 d2 d3 ->  (d2 d3) d0 d1')
# PSPNetH2v2 -> 0.0279
# PSPNetH2v2a -> 0.029
# PSPNetH2v2b -> 0.03356962547355895


psp_modelsH = {
    'resnet18': lambda: PSPNetH2v2(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNetH2v2(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNetH2v2(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNetH2v2(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNetH2v2(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

psp_models4C = {
    'resnet18': lambda: PSPNet4C(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet4C(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet4C(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet4C(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet4C(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


class ModifiedResnet(nn.Module):

    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()

        self.model = psp_models['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x


class ModifiedResnet4C(nn.Module):

    def __init__(self, usegpu=True):
        #super(ModifiedResnet, self).__init__()
        super(ModifiedResnet4C, self).__init__()

        self.model = psp_models4C['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x


class ModifiedResnetWDepth(nn.Module):

    def __init__(self, usegpu=True):
        super(ModifiedResnetWDepth, self).__init__()

        self.model = psp_modelsH['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x,d):
        x = self.model(x,d)
        return x



class PoseNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(39, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(5, 64, 1,padding=122)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb): 

        # x -> points
        # y -> image emb
        
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        #print(x.shape,emb.shape)
        pointfeat_1 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv5(pointfeat_2))
        x = F.relu(self.conv6(x))


        #print(pointfeat_1.shape,pointfeat_2.shape,x.shape)


        return torch.cat([pointfeat_1, pointfeat_2, x], 1) #128 + 256 + 1024


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

class TransformerEncoderW3(nn.Module):
    def __init__(self, n_features, n_heads):
        super(TransformerEncoderW3, self).__init__()

        self.norm       = nn.LayerNorm(n_features)
        self.norm2       = nn.LayerNorm(n_features)
        self.norm3       = nn.LayerNorm(n_features)
        self.attention  = nn.MultiheadAttention(embed_dim= n_features, num_heads = n_heads)
        self.attention2  = nn.MultiheadAttention(embed_dim= n_features, num_heads = n_heads)
        self.attention3  = nn.MultiheadAttention(embed_dim= n_features, num_heads = n_heads)
        self.fc         = nn.Linear(in_features = n_features, out_features = n_features, bias = True)
        self.fc2         = nn.Linear(in_features = n_features, out_features = n_features, bias = True)
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
        z    = self.norm3(z)
        z, _ = self.attention3(z,z,z) 
        z    = z + z1 
        z1   = torch.clone(z)
        z    = self.norm3(z)
        z    = F.relu(self.fc3(z))

        return x+x1,y+y1,z+z1


class StdDevPool3d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=(0,0,0)):
        super(StdDevPool3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x):
        b, d, h, w = x.size()
        kd, kh, kw = self.kernel_size
        sd, sh, sw = self.stride
        pd, ph, pw = self.padding

        # Pad input if needed
        if pd > 0 or ph > 0 or pw > 0:
            x = F.pad(x, (pw, pw, ph, ph, pd, pd), mode='constant', value=0)

        # Calculate local mean
        mean = F.avg_pool3d(x, kernel_size=self.kernel_size, stride=self.stride, padding=0)

        # Calculate local variance
        variance = F.avg_pool3d(x**2, kernel_size=self.kernel_size, stride=self.stride, padding=0) - mean**2

        # Standard deviation
        std = torch.sqrt(variance)

        return std


class KurtosisPool3d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=(0,0,0)):
        super(KurtosisPool3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x):
        b, d, h, w = x.size()
        kd, kh, kw = self.kernel_size
        sd, sh, sw = self.stride
        pd, ph, pw = self.padding

        # Pad input if needed
        if pd > 0 or ph > 0 or pw > 0:
            x = F.pad(x, (pw, pw, ph, ph, pd, pd), mode='constant', value=0)

        # Calculate local mean
        mean = F.avg_pool3d(x, kernel_size=self.kernel_size, stride=self.stride, padding=0)

        # Calculate local fourth central moment (kurtosis)
        moment_4 = F.avg_pool3d(x ** 4, kernel_size=self.kernel_size, stride=self.stride, padding=0)
        variance = F.avg_pool3d(x**2, kernel_size=self.kernel_size, stride=self.stride, padding=0) - mean**2
        kurtosis = moment_4 / (0.0001+variance ** 2) - 3

        return kurtosis

class SkewnessPool3d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=(0,0,0)):
        super(SkewnessPool3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x):
        b, d, h, w = x.size()
        kd, kh, kw = self.kernel_size
        sd, sh, sw = self.stride
        pd, ph, pw = self.padding

        # Pad input if needed
        if pd > 0 or ph > 0 or pw > 0:
            x = F.pad(x, (pw, pw, ph, ph, pd, pd), mode='constant', value=0)

        # Calculate local mean
        mean = F.avg_pool3d(x, kernel_size=self.kernel_size, stride=self.stride, padding=0)

        # Calculate local third central moment (skewness)
        moment_3 = F.avg_pool3d(x ** 3, kernel_size=self.kernel_size, stride=self.stride, padding=0)
        variance = F.avg_pool3d(x**2, kernel_size=self.kernel_size, stride=self.stride, padding=0) - mean**2
        skewness = moment_3 / (variance ** (3/2))

        return skewness



class FeatureSelector(nn.Module):
    def __init__(self, num_channels, kernel_size=1, stride=1):
        super(FeatureSelector, self).__init__()
        # Learnable weights for each channel
        self.weights = nn.Parameter(torch.randn(num_channels))
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = nn.MaxPool3d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        # Normalize weights
        normalized_weights = F.softmax(self.weights, dim=0)
        
        # Scale feature maps
        scaled_feature_maps = x * normalized_weights.view(1, -1, 1, 1)
        
        # Optional: Apply pooling to scaled feature maps
        pooled_feature_maps = self.pool(scaled_feature_maps)
        
        return pooled_feature_maps


class FakeFeatureSelector(nn.Module):
    def __init__(self, num_channels, kernel_size=1, stride=1):
        super(FeatureSelector, self).__init__()
        # Learnable weights for each channel
        self.weights = nn.Parameter(torch.randn(num_channels))
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # Normalize weights
        normalized_weights = F.softmax(self.weights, dim=0)
        
        # Scale feature maps
        scaled_feature_maps = x * normalized_weights.view(1, -1, 1, 1)
                
        return scaled_feature_maps

class FakeFeatureSelectorFC(nn.Module):
    def __init__(self, num_channels, kernel_size=1, stride=1):
        super(FakeFeatureSelectorFC, self).__init__()
        # Learnable weights for each channel
        self.weights = nn.Parameter(torch.randn(num_channels))
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # Normalize weights
        normalized_weights = F.softmax(self.weights, dim=0)
        
        # Scale feature maps
        scaled_feature_maps = x * normalized_weights.view(1, -1)
                
        return scaled_feature_maps


class LearnableAvgPool3d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=1, stride=1):
        super(LearnableAvgPool3d, self).__init__()
        self.conv = nn.Conv3d(1, 1, kernel_size=(1,1,1))
        self.kernel_size=kernel_size
        self.stride=stride

    def forward(self, x):
        # Apply 1x1x1 convolution to input tensor
        weights = self.conv(x.unsqueeze(1))
        ##print(weights.shape)
        # Apply softmax to normalize the weights across channels
        weights = F.softmax(weights, dim=1)
        ##print(weights.shape)
        weights=weights.squeeze(1)
        # Element-wise multiplication of input tensor with normalized weights
        weighted_input = x * weights
        # Sum along the channel dimension to obtain the weighted average
        output = F.avg_pool3d(weighted_input, kernel_size=self.kernel_size, stride=self.stride, padding=0)
        ##print(output.shape)

        return output

 
class MedianPool3d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=(0,0,0)):
        super(MedianPool3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x):
        b, d, h, w = x.size()
        kd, kh, kw = self.kernel_size
        sd, sh, sw = self.stride
        pd, ph, pw = self.padding

        # Pad input if needed
        if pd > 0 or ph > 0 or pw > 0:
            x = F.pad(x, (pw, pw, ph, ph, pd, pd), mode='constant', value=0)

        # Initialize output tensor
        out_d = (d + 2 * pd - kd) // sd + 1
        out_h = (h + 2 * ph - kh) // sh + 1
        out_w = (w + 2 * pw - kw) // sw + 1
        output = torch.zeros(b, out_d, out_h, out_w, device=x.device)

        # Perform median pooling
        for i in range(out_d):
            for j in range(out_h):
                for k in range(out_w):
                    # Select the current pooling region
                    region = x[:, i * sd:i * sd + kd, j * sh:j * sh + kh, k * sw:k * sw + kw]

                    # Reshape and calculate median
                    median_value = torch.median(region.view(b, -1), dim=-1)[0]

                    # Set median value in output tensor
                    output[:, i, j, k] = median_value

        return output



class AvgStdDevSamplePool3d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=(0,0,0), num_samples=1):
        super(AvgStdDevSamplePool3d, self).__init__()
        self.avg_pool = nn.AvgPool3d(kernel_size, stride=stride, padding=padding)
        self.std_dev_pool = StdDevPool3d(kernel_size, stride=stride, padding=padding)
        self.num_samples = num_samples

    def forward(self, x):
        avg_output = self.avg_pool(x)
        std_dev_output = self.std_dev_pool(x)
        
        b, d, h, w = avg_output.size()


        ##print(avg_output.shape,std_dev_output.shape)
        
        # Reshape std_dev_output to match the shape of avg_output
        std_dev_output = std_dev_output.view(b, -1)
        avg_output = avg_output.view(b, -1)
        ##print(avg_output.shape,std_dev_output.shape)



        # Sample from Gaussian distribution parameterized by mean and standard deviation
        samples = torch.normal(avg_output, std_dev_output)

        ##print(samples.shape)
        # Take the average of the samples
        #combined_output = torch.mean(samples, dim=2)

        return samples

from functools import partial
from einops.layers.torch import Rearrange, Reduce


class FeatureMixingModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureMixingModule, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Depthwise convolution
        out = self.depthwise_conv(x)
        # Pointwise convolution
        out = self.pointwise_conv(out)
        return out



from torchvision.models import resnet34,resnet18
from torch.nn import MultiheadAttention





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





class RGBDEncoder2RESNET18(nn.Module):
    def __init__(self):
        super(RGBDEncoder2RESNET18, self).__init__()
        self.rgb_backbone = resnet18(pretrained=True)
        self.depth_backbone = resnet18(pretrained=True)
        self.depth_backbone2 = resnet18(pretrained=True)

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


class RGBEncoder(nn.Module):
    def __init__(self):
        super(RGBEncoder, self).__init__()
        self.rgb_backbone = resnet34(pretrained=True)

        # Remove the fully connected layers
        self.rgb_backbone = nn.Sequential(*list(self.rgb_backbone.children())[:-2])

    def forward(self, rgb):
        rgb_features = self.rgb_backbone(rgb)
      
                
        return rgb_features


class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionFusion, self).__init__()
        self.attentionRGB = MultiheadAttention(embed_dim, num_heads)
        self.attentionDepth = MultiheadAttention(embed_dim, num_heads)

    def forward(self, rgb_features, depth_features):
        # Flatten the spatial dimensions for multi-head attention
        rgb_flat = rgb_features.flatten(2).permute(2, 0, 1)
        depth_flat = depth_features.flatten(2).permute(2, 0, 1)
        fused_features, _ = self.attentionRGB(rgb_flat, rgb_flat, rgb_flat)
        fused_features2, _ = self.attentionDepth(depth_flat, depth_flat, depth_flat)

        #print(fused_features.shape,fused_features2.shape)
        return  torch.cat([fused_features, fused_features2], 0)


class MultiHeadAttentionFusionPlaceOlder(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionFusionPlaceOlder, self).__init__()
 

    def forward(self, rgb_features):
        # Flatten the spatial dimensions for multi-head attention
        rgb_flat = rgb_features.flatten(2).permute(2, 0, 1)


        return  rgb_flat


class MultiHeadAttentionFusion2(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionFusion2, self).__init__()
        self.attention = MultiheadAttention(embed_dim, num_heads)
        self.attention1 = MultiheadAttention(embed_dim, num_heads)
        self.attention2 = MultiheadAttention(embed_dim, num_heads)

    def forward(self, rgb_features, depth_features,depth2_features):
        # Flatten the spatial dimensions for multi-head attention
        #print(rgb_features.shape)
        rgb_flat = rgb_features.flatten(2).permute(2, 0, 1)
        depth_flat = depth_features.flatten(2).permute(2, 0, 1)
        depth2_flat = depth2_features.flatten(2).permute(2, 0, 1)
        fused_features1, _ = self.attention(rgb_flat, rgb_flat, rgb_flat)
        fused_features2, _ = self.attention1(depth_flat, depth_flat, depth_flat)
        fused_features3, _ = self.attention2(depth2_flat, depth2_flat, depth2_flat)

        return torch.cat([fused_features1,fused_features2, fused_features3], 0)


class RGBDFeatureExtractor(nn.Module):
    def __init__(self, num_heads=4):
        super(RGBDFeatureExtractor, self).__init__()
        self.encoder = RGBDEncoder()
        self.attention_fusion = MultiHeadAttentionFusion(embed_dim=512, num_heads=num_heads)

    def forward(self, rgb, depth):
        # Encode RGB and Depth images
        rgb_features,depth_features = self.encoder(rgb, depth)
        fused_features = self.attention_fusion(rgb_features, depth_features)

        return fused_features

class RGBFeatureExtractor(nn.Module):
    def __init__(self, num_heads=4):
        super(RGBFeatureExtractor, self).__init__()
        self.encoder = RGBEncoder()
        self.attention_fusion = MultiHeadAttentionFusionPlaceOlder(embed_dim=512, num_heads=num_heads)

    def forward(self, rgb):
        # Encode RGB and Depth images
        rgb_features= self.encoder(rgb)



       # ap_x=torch.cat([rgb_features, depth_features, depth2_features], 2)


        fused_features = self.attention_fusion(rgb_features)

        return fused_features

class RGBDFeatureExtractor2(nn.Module):
    def __init__(self, num_heads=4):
        super(RGBDFeatureExtractor2, self).__init__()
        self.encoder = RGBDEncoder2()
        self.attention_fusion = MultiHeadAttentionFusion2(embed_dim=512, num_heads=num_heads)

    def forward(self, rgb, depth, depth2):
        # Encode RGB and Depth images
        rgb_features,depth_features,depth2_features = self.encoder(rgb, depth,depth2)



       # ap_x=torch.cat([rgb_features, depth_features, depth2_features], 2)


        fused_features = self.attention_fusion(rgb_features, depth_features, depth2_features)

        return fused_features


class RGBDFeatureExtractor2CUSTOM(nn.Module):
    def __init__(self, modelRGB,modelDepth, modelDepth2, num_heads=4,emb=512):
        super(RGBDFeatureExtractor2CUSTOM, self).__init__()
        self.encoder = RGBDEncoder2CUSTOM(modelRGB,modelDepth, modelDepth2)
        self.attention_fusion = MultiHeadAttentionFusion2(embed_dim=emb, num_heads=num_heads)

    def forward(self, rgb, depth, depth2):
        # Encode RGB and Depth images
        rgb_features,depth_features,depth2_features = self.encoder(rgb, depth,depth2)



       # ap_x=torch.cat([rgb_features, depth_features, depth2_features], 2)


        fused_features = self.attention_fusion(rgb_features, depth_features, depth2_features)

        return fused_features


class RGBDFeatureExtractor2RESNET18(nn.Module):
    def __init__(self, num_heads=4):
        super(RGBDFeatureExtractor2RESNET18, self).__init__()
        self.encoder = RGBDEncoder2RESNET18()
        self.attention_fusion = MultiHeadAttentionFusion2(embed_dim=512, num_heads=num_heads)

    def forward(self, rgb, depth, depth2):
        # Encode RGB and Depth images
        rgb_features,depth_features,depth2_features = self.encoder(rgb, depth,depth2)



       # ap_x=torch.cat([rgb_features, depth_features, depth2_features], 2)


        fused_features = self.attention_fusion(rgb_features, depth_features, depth2_features)

        return fused_features


class StdDevPooling1D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(StdDevPooling1D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        x_unfold = F.unfold(x.unsqueeze(-1), (self.kernel_size, 1), stride=(self.stride, 1), padding=(self.padding, 0))
        x_unfold = x_unfold.view(x.size(0), x.size(1), self.kernel_size, -1)
        std_dev = torch.std(x_unfold, dim=2)
        return std_dev

class KurtosisPooling1D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(KurtosisPooling1D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        x_unfold = F.unfold(x.unsqueeze(-1), (self.kernel_size, 1), stride=(self.stride, 1), padding=(self.padding, 0))
        x_unfold = x_unfold.view(x.size(0), x.size(1), self.kernel_size, -1)
        mean = x_unfold.mean(dim=2, keepdim=True)
        variance = ((x_unfold - mean) ** 2).mean(dim=2, keepdim=True)
        fourth_moment = ((x_unfold - mean) ** 4).mean(dim=2)
        kurtosis = fourth_moment / (variance ** 2) - 3
        return kurtosis

class SkewnessPooling1D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(SkewnessPooling1D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        x_unfold = F.unfold(x.unsqueeze(-1), (self.kernel_size, 1), stride=(self.stride, 1), padding=(self.padding, 0))
        x_unfold = x_unfold.view(x.size(0), x.size(1), self.kernel_size, -1)
        mean = x_unfold.mean(dim=2, keepdim=True)
        variance = ((x_unfold - mean) ** 2).mean(dim=2, keepdim=True)
        skewness = (((x_unfold - mean) ** 3).mean(dim=2)) / (variance ** 1.5)
        return skewness

class MedianPooling1D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(MedianPooling1D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        x_unfold = F.unfold(x.unsqueeze(-1), (self.kernel_size, 1), stride=(self.stride, 1), padding=(self.padding, 0))
        x_unfold = x_unfold.view(x.size(0), x.size(1), self.kernel_size, -1)
        median = torch.median(x_unfold, dim=2)[0]
        return median



def weighted_average_quaternions(quaternions, weights):
    """
    Average multiple quaternions with specific weights

    :params quaternions: is a Nx4 numpy matrix and contains the quaternions
        to average in the rows.
        The quaternions are arranged as (w,x,y,z), with w being the scalar

    :params weights: The weight vector w must be of the same length as
        the number of rows in the

    :returns: the average quaternion of the input. Note that the signs
        of the output quaternion can be reversed, since q and -q
        describe the same orientation
    :raises: ValueError if all weights are zero
    """
    # Number of quaternions to average
    samples = quaternions.shape[0]
    mat_a = np.zeros(shape=(4, 4), dtype=np.float64)
    weight_sum = 0

    for i in range(0, samples):
        quat = quaternions[i, :]
        mat_a = weights[i] * np.outer(quat, quat) + mat_a
        weight_sum += weights[i]

    if weight_sum <= 0.0:
        raise ValueError("At least one weight must be greater than zero")

    # scale
    mat_a = (1.0/weight_sum) * mat_a

    # compute eigenvalues and -vectors
    eigen_values, eigen_vectors = np.linalg.eig(mat_a)

    # Sort by largest eigenvalue
    eigen_vectors = eigen_vectors[:, eigen_values.argsort()[::-1]]

    # return the real part of the largest eigenvector (has only real part)
    return np.real(np.ravel(eigen_vectors[:, 0]))


 





def replace_batchnorm_with_instancenorm(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            # Replace with InstanceNorm2d
            instance_norm = nn.InstanceNorm2d(module.num_features, affine=True)
            setattr(model, name, instance_norm)
        elif isinstance(module, nn.Module):
            replace_batchnorm_with_instancenorm(module)



 

 
import torch
import torch.nn as nn
import torch.nn.functional as F


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





    
class PoseNetMultiCUSTOMPointsX_old(nn.Module):
    def __init__(self, modelRGB, modelDepth, modelDepth2, num_points, num_obj, embchannels, noise=0.25):
        super(PoseNetMultiCUSTOMPointsX_old, self).__init__()
        self.num_points = num_points
        self.noise = noise

        self.cnn = RGBDFeatureExtractor2CUSTOM(modelRGB,modelDepth, modelDepth2,num_heads=4,emb=embchannels)
        
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
 