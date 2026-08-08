import torch
import torch.nn as nn
from torch.nn import functional as F



def rank_loss(gt: torch.Tensor, pred: torch.Tensor, eps=1e-6):
    """
    Rank loss 实现（对应论文公式 20-22）
    :param pred: 预测剂量，任意形状 (B, ...)
    :param gt:   真值剂量，与 pred 同形
    :return:     rank loss 标量
    """
    # 展平并排序
    B = pred.size(0)
    pred = pred.view(B, -1)          # (B, N)
    gt   = gt.view(B, -1)            # (B, N)

    # 按 gt 值升序排序
    gt_sorted, indices = torch.sort(gt, dim=1)   # (B, N)
    pred_sorted = torch.gather(pred, 1, indices) # (B, N)

    # 相邻差值
    delta_gt = gt_sorted[:, 1:] - gt_sorted[:, :-1]   # (B, N-1)
    delta_pr = pred_sorted[:, 1:] - pred_sorted[:, :-1]

    # 标签 ω：gt 后>前为1，相等0.5，否则0
    omega = (delta_gt > 0).float() + (delta_gt == 0).float() * 0.5

    # 概率 ρ（公式20）
    rho = torch.sigmoid(delta_pr)

    # 负对数似然（公式22）
    loss_pos = -omega * torch.log(rho + eps)
    loss_neg = -(1 - omega) * torch.log(1 - rho + eps)
    rank_loss = (loss_pos + loss_neg).mean()

    return rank_loss


class MS_SSIM_L1_LOSS(nn.Module):
    """
    Have to use cuda, otherwise the speed is too slow.
    Both the group and shape of input image should be attention on.
    I set 255 and 1 for gray image as default.
    """

    def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
                 data_range=1,
                 K=(0.01, 0.03),  # c1,c2
                 alpha=0.025,  # weight of ssim and l1 loss
                 compensation=200.0,  # final factor for total loss
                 cuda_dev=0,  # cuda device choice
                 channel=1):  # RGB image should set to 3 and Gray image should be set to 1
        super(MS_SSIM_L1_LOSS, self).__init__()
        self.channel = channel
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation = compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros(
            (self.channel * len(gaussian_sigmas), 1, filter_size, filter_size))  # 创建了(3*5, 1, 33, 33)个masks
        for idx, sigma in enumerate(gaussian_sigmas):
            if self.channel == 1:
                # only gray layer
                g_masks[idx, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            elif self.channel == 3:
                # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
                g_masks[self.channel * idx + 0, 0, :, :] = self._fspecial_gauss_2d(filter_size,
                                                                                   sigma)  # 每层mask对应不同的sigma
                g_masks[self.channel * idx + 1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
                g_masks[self.channel * idx + 2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            else:
                raise ValueError
        self.g_masks = g_masks.cuda(cuda_dev)  # 转换为cuda数据类型

    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)
        # Outer product of input and vec2. If input is a vector of size nn and vec2 is a vector of size mm,
        # then out must be a matrix of size (n \times m)(n×m).

    def forward(self, x, y):
        b, c, h, w = x.shape
        assert c == self.channel

        mux = F.conv2d(x, self.g_masks, groups=c, padding=self.pad)  # 图像为96*96，和33*33卷积，出来的是64*64，加上pad=16,出来的是96*96
        muy = F.conv2d(y, self.g_masks, groups=c, padding=self.pad)  # groups 是分组卷积，为了加快卷积的速度

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=c, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=c, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=c, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l = (2 * muxy + self.C1) / (mux2 + muy2 + self.C1)  # [B, 15, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)
        if self.channel == 3:
            lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]  # 亮度对比因子
            PIcs = cs.prod(dim=1)
        elif self.channel == 1:
            lM = l[:, -1, :, :]
            PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM * PIcs  # [B, H, W]

        loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, C, H, W]
        # average l1 loss in num channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-self.channel, length=self.channel),
                               groups=c, padding=self.pad).mean(1)  # [B, H, W]

        # loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        # loss_mix = self.compensation * loss_mix
        loss_mix = self.alpha * loss_ms_ssim

        return loss_mix.mean()
# def rank_loss_mae_scale(pred: torch.Tensor, gt: torch.Tensor, delta: float = 2.0):
#     """
#     与 MAE 同尺度的 rank loss
#     pred, gt: 同形 tensor，已归一化到同一剂量范围（如 0~70 Gy）
#     delta:    Huber 阈值，可设为 2 Gy 或 1 Gy
#     """
#     # 1. 展平并升序排序（与 PRT-Net 一致）
#     B = pred.size(0)
#     pred = pred.view(B, -1)
#     gt   = gt.view(B, -1)
#     gt_sorted, idx = torch.sort(gt, dim=1)
#     pred_sorted = torch.gather(pred, 1, idx)
#
#     # 2. 相邻差值
#     diff_gt = gt_sorted[:, 1:] - gt_sorted[:, :-1]          # (B, N-1)
#     diff_pr = pred_sorted[:, 1:] - pred_sorted[:, :-1]
#
#     # 3. 标签 ω：升序方向
#     omega = (diff_gt > 0).float() + (diff_gt == 0).float() * 0.5
#
#     # 4. 把“排序误差”变成剂量尺度的误差
#     #    我们希望 diff_pr 与 diff_gt 同号且同量级
#     err = diff_pr - diff_gt                                 # 单位：Gy
#     err = torch.where(omega > 0.5, err, -err)              # 符号惩罚
#
#     # 5. Huber 范数（与 MAE 同梯度行为）
#     huber = torch.where(err.abs() <= delta,
#                         0.5 * err ** 2,
#                         delta * (err.abs() - 0.5 * delta))
#     return huber.mean()
#
# def rank_loss_linear(gt, pred):
#     """
#     无 sigmoid、与 MAE 同尺度的 rank loss
#     pred, gt: 同形 tensor，单位 Gy
#     """
#     B = pred.size(0)
#     pred = pred.view(B, -1)
#     gt   = gt.view(B, -1)
#
#     # 升序排序
#     gt_sorted, idx = torch.sort(gt, dim=1)
#     pred_sorted = torch.gather(pred, 1, idx)
#
#     # 相邻差值
#     delta_gt = gt_sorted[:, 1:] - gt_sorted[:, :-1]
#     delta_pr = pred_sorted[:, 1:] - pred_sorted[:, :-1]
#
#     # L1 误差
#     return (delta_pr - delta_gt).abs().mean()

