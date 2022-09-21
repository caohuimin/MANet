import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from CBAM import *
from densenet import *
from Resnet import *
from ASPP import *
from non_local_gaussian import *
from ssim import *

from unet_parts import *   #引入unet部分

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#卷积核大小为k, 卷积后尺寸大小不变，改变通道的卷积操作
def conv_k(in_channels, out_channels,k):
    out = nn.Conv2d(in_channels, out_channels, k, padding=1)
    return out

#卷积核为3x3，卷积后尺寸大小不变，改变通道大小的卷积操作
def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 3, padding=1)

#反卷积
def deconvkxk(in_channels, out_channels,k,p):
    return nn.ConvTranspose2d(in_channels, out_channels, k,
                              stride=2, padding=p, output_padding=1)

#不改变通道，图片尺寸减小一半的卷积操作
def conv(in_channels, out_channels,k):
    out = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    return out

#可改变通道大小，并将尺寸变大两倍的上采样（反卷积）操作
def deconv3x3(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, 3,
                              stride=2, padding=1, output_padding=1)

#扩张卷积
def dilatk_convkxk(in_channels,out_channels,k,p,d):
    return nn.Conv2d(in_channels,out_channels,k,padding=p,dilation=d)

#可改变通道大小，并将尺寸变大两倍的上采样（反卷积）操作
def deconv(in_channels, out_channels):
    out = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1,output_padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))
    return out

class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

class CompoundLoss(nn.Module):
    def __init__(self, alpha=0.8, normalize=True):
        super(CompoundLoss, self).__init__()
        self.alpha = alpha
        self.normalize = normalize
        self.charbonnier_loss = L1_Charbonnier_loss()

    def forward(self, prediction, target):
        return (F.mse_loss(prediction,target)+
                self.charbonnier_loss(prediction,target) + self.alpha * (1.0 - msssim(prediction, target, normalize=self.normalize)))


#多尺度机制
def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
    prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    bn = norm_layer(out_channels)
    relu = nn.ReLU(inplace=True)
    return nn.Sequential(prior, conv, bn, relu)


#空间注意力机制
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        #super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class conv_up(nn.Module):
    def __init__(self, channel):
        super(conv_up, self).__init__()
        self.conv1_up = nn.Conv2d(in_channels=channel, out_channels=channel*2, kernel_size=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear',align_corners=True)
        x = self.conv1_up(x)
        return x


# Landsat高分辨率网络
class LTHSNet(nn.Module):
    def __init__(self, channels, bilinear=True):
        super(LTHSNet, self).__init__()

        self.conv1_3x3 = dilatk_convkxk(1, 12, 3, 1, 1)  # 3x3卷积核
        self.conv2_3x3 = dilatk_convkxk(1, 12, 3, 2, 2)  # 3x3卷积核
        self.conv3_3x3 = dilatk_convkxk(1, 12, 3, 3, 3)  # 3x3卷积核

        self.de_conv1 = deconvkxk(36, 36, 3, 1)  #反卷积
        self.de_conv2 = deconvkxk(36, 36, 3, 1)
        self.de_conv3 = deconvkxk(36, 36, 3, 1)
        self.de_conv4 = deconvkxk(36, 36, 3, 1)

        self.conv1_5x5 = dilatk_convkxk(1, 12, 5, 2, 1)
        self.conv2_5x5 = dilatk_convkxk(12, 12, 5, 2, 1)
        self.conv3_5x5 = dilatk_convkxk(12, 36, 5, 2, 1)

        self.conv3x3 = conv3x3(36,36)

        self.non_local1 = NONLocalBlock2D(1, sub_sample=True, bn_layer=False)
        self.poolMax1 = conv(12, 36, 3)
        self.channel_conv1 = conv_k(12,36,3)
        self.poolMax2 = conv(36, 36, 3)
        self.poolMax3 = conv(36, 36, 3)

        #self.channelAttention1 = ChannelAttention(36, 1)
        self.senet = SELayer(36)
        #self.sk_conv = SKConv(36, WH=1, M=2, G=1, r=2)
        self.aff = AFF()


        # 改变通道
        self.channel_conv = conv_k(1, 36, 3)

        self.apnb = APNB(48,48,3,3,0.05)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, coarse, fine):
        #Modis
        x = coarse[1]
        y = fine - coarse[0]  #y是modis图像，x是landsat图像

        out5 = self.relu(self.conv1_3x3(y))
        out6 = self.relu(self.conv2_3x3(y))
        out7 = self.relu(self.conv3_3x3(y))

        #合并低层信息和卷积后的信息
        out8 = torch.cat([out5, out6, out7], dim=1)  #(4,12,10,10) #这里不确定要不要加y



        x1 = self.relu(self.conv1_5x5(x))  # 64
        x2 = self.relu(self.conv2_5x5(x1))  # 128
        x3 = self.conv3_5x5(x2)



        M1 = self.channel_conv1(self.poolMax1(x3))
        M2 = self.poolMax2(M1)
        M3 = self.poolMax3(M2)

        out9 = self.relu(self.de_conv1(out8))
        out10 = self.conv3x3(out9)
        out11 = self.senet(out10) + out9
        out11 = out11 + M3
        out12 = self.relu(self.de_conv2(out11))
        out13 = self.conv3x3(out12)
        out14 = self.senet(out13) + out12
        out14 = out14 + M2
        out15 = self.relu(self.de_conv3(out14))
        out16 = self.conv3x3(out15)
        out17 = self.senet(out16) + out15
        out17 = out17 + M1
        out18 = self.relu(self.de_conv4(out17))
        out19 = self.conv3x3(out18)
        out20 = self.senet(out19) + out18
        out20 = self.conv3x3(out20)


        out = torch.cat([out20,x3],dim=1)
        out = self.apnb(out)

        #out = self.aff1(x5,out12)

        return out



# Reconstruction重建图像网络（去掉全连接）
class ReconstructNet(nn.Module):
    def __init__(self, channels):
        super(ReconstructNet, self).__init__()
        self.dense1 = nn.Linear(48,24)
        self.dense2 = nn.Linear(24, 1)  # 全连接操作，改变通道大小
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        out = x
        out = out.permute(0, 2, 3, 1)
        out = self.dense1(out)
        out = self.relu(out)
        out = self.dense2(out)
        out = out.permute(0, 3, 1, 2)
        return out


class FusionNet(nn.Module):
    def __init__(self, channels):
        super(FusionNet, self).__init__()
        #self.coarse_net = HTLSNet(channels)
        self.fine_net = LTHSNet(channels)
        self.reconstruct_net = ReconstructNet(36)

    def forward(self, coarse_input, reference_inputs):
        #assert len(reference_inputs) == 1
        #coarse_in = self.coarse_net(coarse_input)

        #coarse = self.coarse_net(reference_inputs[0]) #M1-M2的残差图像训练
        fine = self.fine_net(reference_inputs, coarse_input)   #L1的精细图像训练
        #print("coarse", coarse.shape)
        #print("fine", fine.shape)
        #result = torch.cat((coarse, fine), dim=1)   #这里要进行concat操作
        #print(result.shape)   (32,96,160,160)

        result = self.reconstruct_net(fine)
        return result


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out += identity

        return out


class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction):
        super(ChannelAttention, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.module(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

#SK MOdule
class SKConv(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1, L=32):
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            # 使用不同kernel size的卷积
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(features,
                              features,
                              kernel_size=3 + i * 2,
                              stride=stride,
                              padding=1 + i,
                              groups=G), nn.BatchNorm2d(features),
                    nn.ReLU(inplace=False)))

        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, features))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v



class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


    #IAFF
class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo


class MS_CAM(nn.Module):
    '''
    单特征 进行通道加权,作用类似SE模块
    '''

    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei

class APNB(nn.Module):
    """
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=([1]), norm_type=None,psp_size=(1,3,6,8)):
        super(APNB, self).__init__()
        self.stages = []
        self.norm_type = norm_type
        self.psp_size=psp_size
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, out_channels, key_channels, value_channels, size) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, padding=0),
            #ModuleHelper.BNReLU(out_channels, norm_type=norm_type),
            nn.Dropout2d(dropout)
        )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock2D(in_channels,
                                    key_channels,
                                    value_channels,
                                    output_channels,
                                    size,
                                    self.norm_type,
                                    self.psp_size)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center


class _SelfAttentionBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1, norm_type=None,psp_size=(1,3,6,8)):
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            #ModuleHelper.BNReLU(self.key_channels, norm_type=norm_type),
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)

        self.psp = PSPModule(psp_size)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        value = self.psp(self.f_value(x))

        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x)
        # value=self.psp(value)#.view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        key = self.psp(key)  # .view(batch_size, self.key_channels, -1)
        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        return context


class SelfAttentionBlock2D(_SelfAttentionBlock):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1, norm_type=None,psp_size=(1,3,6,8)):
        super(SelfAttentionBlock2D, self).__init__(in_channels,
                                                   key_channels,
                                                   value_channels,
                                                   out_channels,
                                                   scale,
                                                   norm_type,
                                                   psp_size=psp_size)

