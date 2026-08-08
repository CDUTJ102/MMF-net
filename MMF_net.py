import torch.nn as nn
import torch
from torch.nn import functional as F
from timm.layers import DropPath
#import numpy as np
#from einops import rearrange

def _init_weights(module):
    """Kaiming 初始化"""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, dim, drop_path_rate=0.):
        super().__init__()
        # 深度可分离 7×7
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.bn = nn.BatchNorm2d(dim)          # 可替换成 LayerNorm，但官方先用 BN

        # 1×1 卷积（等价 Linear 当通道在最后一维）
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()


    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.bn(x)
        x = x.permute(0, 2, 3, 1)          # (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        x = shortcut + self.drop_path(x)   # 残差 + 随机深度
        return x


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.layer = nn.Conv2d(in_ch, in_ch // 2, 1, 1)
        self.double_conv = double_conv(in_ch, out_ch)

    def forward(self, x, y):
        x = self.up(x)
        x = self.layer(x)
        x = torch.cat([x, y], dim=1)
        x = self.double_conv(x)
        return x



class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.D_layer = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.D_layer(x)



class FeatureFusion(nn.Module):
    def __init__(self, channels, r=4):
        super(FeatureFusion, self).__init__()
        inter_channels = int(channels // r)

        self.fuse = nn.Conv2d(3 * channels, channels, 1, bias=False)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, ct, oar, ptv):
        f0 = torch.cat([ct, oar, ptv], dim=1)  # (B,3C,H,W)
        f0 = self.fuse(f0)  # (B,C,H,W)
        #print("Inside AFF xa: ", xa.size())
        xl = self.local_att(f0)
        #print("Inside AFF xl: ", xl.size())
        xg = self.global_att(f0)
        #print("Inside AFF xg: ", xg.size())
        xlg = xl + xg + f0
        #print("Inside AFF xlg: ", xlg.size())
        wei = self.sigmoid(xlg)

        f_fused = ptv * wei + ct * wei * 0.7 + oar * wei
        #print("Inside AFF xo: ", wei.size())
        return f_fused


class InfoGate(nn.Module):
    """动态加权：‖x‖₂² - γ‖x‖₁ → softmax 权重"""
    def __init__(self, C, reduction=16, gamma=0.5):
        super().__init__()
        self.gamma = gamma
        self.fc = nn.Sequential(
            nn.Linear(2, C // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(C // reduction, 2)
        )

    def info_est(self, x):
        l2 = x.pow(2).mean()
        l1 = x.abs().mean()
        return l2 - self.gamma * l1

    def forward(self, x_ch, x_sp):
        e1 = self.info_est(x_ch)
        e2 = self.info_est(x_sp)
        logits = self.fc(torch.stack([e1, e2], dim=0).unsqueeze(0)).squeeze(0)
        w = logits.softmax(dim=0)               # [α, β]
        out = w[0] * x_ch + w[1] * x_sp
        return out, w.detach().cpu().numpy()

class CBAM(nn.Module):
    """
    原 CBAM 的通道+空间并行，但用 InfoGate 动态加权
    """
    def __init__(self, c1, r=16, gamma=0.5):
        super().__init__()
        c_ = max(c1 // r, 1)

        # 通道分支
        self.mlp = nn.Sequential(
            nn.Conv2d(c1, c_, 1, bias=False), nn.ReLU(),
            nn.Conv2d(c_, c1, 1, bias=False)
        )
        # 空间分支
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

        # 动态门控
        self.gate = InfoGate(c1, reduction=r, gamma=gamma)

    def forward(self, x):
        # 1. 通道注意力分支 → 得到加权特征 x_ch
        avg_out = torch.mean(x, dim=(2, 3), keepdim=True)
        max_out, _ = torch.max(x.view(x.size(0), x.size(1), -1), dim=2, keepdim=True)
        max_out = max_out.view(x.size(0), x.size(1), 1, 1)
        channel_att = torch.sigmoid(self.mlp(avg_out) + self.mlp(max_out))
        x_ch = x * channel_att

        # 2. 空间注意力分支 → 得到加权特征 x_sp
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        x_sp = x * spatial_att

        # 3. 动态信息量加权融合
        out, w = self.gate(x_ch, x_sp)          # w 可可视化
        return out

class AM(nn.Module):
    """
    Self-Enhanced Attention Module
    输入: 单尺度特征图  Ci  (B,C,H,W)
    输出: 增强后的特征图      (B,C,H,W)
    """
    def __init__(self, in_channels, reduction=16):
        super(AM, self).__init__()
        # 1. 并行空洞卷积
        self.dconv = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, 3,
                      padding=dil, dilation=dil, bias=False)
            for dil in [1, 2, 3, 4]
        ])
        # 2. 拼接后的融合卷积
        self.fuse = nn.Conv2d(in_channels * 4, in_channels, 3, padding=1, bias=False)
        # 3. 融合
        self.guide_conv = nn.Conv2d(in_channels, in_channels, 1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        #self.cp = CpAttentionLite(in_channels)
        self.cp = CBAM(in_channels)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels*2, in_channels, 3, padding=1)
        )

    def forward(self, x, y):
        y = self.up(y)

        cross = self.sigmoid(self.guide_conv(x))*y

        input = x
        x = x + cross

        # 1. 多 dilation 卷积 + 拼接
        multi = torch.cat([d(x) for d in self.dconv], dim=1)

        fused = self.fuse(multi)          # (B,C,H,W)

        o = self.cp(fused)

        return o + fused + input



class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel//2, 1, 1)

    def forward(self, x):
        up = F.interpolate(x, scale_factor=2, mode='bilinear')
        out = self.layer(up)
        return out


class DoseRefineMLP(nn.Module):
    """
    输入: coarse 剂量 + PTV + OARs
    输出: 精修剂量图 (residual)
    """
    def __init__(self, in_ch, hidden=64, out_ch=1, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1, bias=True),
            nn.LeakyReLU(),
            nn.Dropout2d(drop),
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=True),
            nn.LeakyReLU(),
            nn.Dropout2d(drop),
            nn.Conv2d(hidden, out_ch, kernel_size=1, bias=True)
        )

    def forward(self, coarse_dose, oar, ptv):
        # 通道拼接
        x = torch.cat((coarse_dose, oar, ptv), dim=1)  # B,C,H,W,D
        residual = self.net(x)
        return coarse_dose + residual


class MMF_net(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(MMF_net, self).__init__()
        self.spatial_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1_1 = double_conv(3, 32)
        self.conv1_2 = double_conv(3, 32)
        self.conv1_3 = double_conv(3, 32)

        self.share_conv1 = ConvBlock(32)
        self.conv2_1 = double_conv(32, 64)
        self.conv2_2 = double_conv(32, 64)
        self.conv2_3 = double_conv(32, 64)

        self.share_conv2 = ConvBlock(64)
        self.conv3_1 = double_conv(64, 128)
        self.conv3_2 = double_conv(64, 128)
        self.conv3_3 = double_conv(64, 128)

        self.share_conv3 = ConvBlock(128)
        self.conv4_1 = double_conv(128, 256)
        self.conv4_2 = double_conv(128, 256)
        self.conv4_3 = double_conv(128, 256)

        self.share_conv4 = ConvBlock(256)
        self.conv5_1 = double_conv(256, 512)
        self.conv5_2 = double_conv(256, 512)
        self.conv5_3 = double_conv(256, 512)

        self.down1_1 = DownSample(32)
        self.down1_2 = DownSample(32)
        self.down1_3 = DownSample(32)

        self.down2_1 = DownSample(64)
        self.down2_2 = DownSample(64)
        self.down2_3 = DownSample(64)

        self.down3_0 = DownSample(128)
        self.down3_1 = DownSample(128)
        self.down3_2 = DownSample(128)
        self.down3_3 = DownSample(128)

        self.down4_0 = DownSample(256)
        self.down4_1 = DownSample(256)
        self.down4_2 = DownSample(256)
        self.down4_3 = DownSample(256)


        self.up1 = UpConv(512, 256)
        self.up2 = UpConv(256, 128)
        self.up3 = UpConv(128, 64)
        self.up4 = UpConv(64, 32)

        self.feature_fusion0 = FeatureFusion(32)
        self.feature_fusion1 = FeatureFusion(64)
        self.feature_fusion2 = FeatureFusion(128)
        self.feature_fusion3 = FeatureFusion(256)
        self.feature_fusion4 = FeatureFusion(512)

        self.ia0 = AM(256)
        self.ia1 = AM(128)
        self.ia2 = AM(64)
        self.ia3 = AM(32)

        self.u1 = UpSample(512)
        self.u2 = UpSample(256)
        self.u3 = UpSample(128)
        self.u4 = UpSample(64)


        self.out = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.DoseRefineMLP = DoseRefineMLP(in_ch=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        x_p = x2[:,1].unsqueeze(1)
        x_o = x3[:,1].unsqueeze(1)
        x1 = self.conv1_1(x1)
        x2 = self.conv1_2(x2)
        x3 = self.conv1_3(x3)
        x0_0 = self.feature_fusion0(x1, x2, x3)
        x1 = self.down1_1(x1)
        x2 = self.down1_2(x2)
        x3 = self.down1_3(x3)

        x1 = self.share_conv1(x1)
        x2 = self.share_conv1(x2)
        x3 = self.share_conv1(x3)

        x1 = self.conv2_1(x1)
        x2 = self.conv2_2(x2)
        x3 = self.conv2_3(x3)
        x0_1 = self.feature_fusion1(x1, x2, x3)



        x1 = self.down2_1(x1)
        x2 = self.down2_2(x2)
        x3 = self.down2_3(x3)

        x1 = self.share_conv2(x1)
        x2 = self.share_conv2(x2)
        x3 = self.share_conv2(x3)

        x1 = self.conv3_1(x1)
        x2 = self.conv3_2(x2)
        x3 = self.conv3_3(x3)
        x0_2 = self.feature_fusion2(x1, x2, x3)



        x1 = self.down3_1(x1)
        x2 = self.down3_2(x2)
        x3 = self.down3_3(x3)

        x1 = self.share_conv3(x1)
        x2 = self.share_conv3(x2)
        x3 = self.share_conv3(x3)

        x1 = self.conv4_1(x1)
        x2 = self.conv4_2(x2)
        x3 = self.conv4_3(x3)
        x0_3 = self.feature_fusion3(x1, x2, x3)

        x1 = self.down4_1(x1)
        x2 = self.down4_2(x2)
        x3 = self.down4_3(x3)

        x1 = self.share_conv4(x1)
        x2 = self.share_conv4(x2)
        x3 = self.share_conv4(x3)
        x1 = self.conv5_1(x1)
        x2 = self.conv5_2(x2)
        x3 = self.conv5_3(x3)
        x0_4 = self.feature_fusion4(x1, x2, x3)



        o1 = self.up1(x0_4, self.ia0(x0_3, x0_4))
        o2 = self.up2(o1, self.ia1(x0_2, x0_3))
        o3 = self.up3(o2, self.ia2(x0_1, x0_2))
        o4 = self.up4(o3, self.ia3(x0_0, x0_1))

        out = self.out(o4)
        outs = self.DoseRefineMLP(out, x_p, x_o)
        return self.sigmoid(outs)
if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis

    net = MMF_net()
    x = torch.randn(1, 3, 3, 256, 256)
    device = torch.device("cpu")

    model = net.to(device).eval()


    flop_analysis = FlopCountAnalysis(model, x)
    flops = flop_analysis.total()

    params = sum(p.numel() for p in model.parameters())

    print(f"FLOPs: {flops / 1e9:.3f} GFLOPs")
    print(f"Params: {params / 1e6:.3f} M")
    y = net(x)
    # for i, o in enumerate(y):
    #     print(f"pred{i+1}:", o.shape)
    print(y.shape)
