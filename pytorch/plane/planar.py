import torch
import torch.nn as nn


class plane_loss(nn.Module):
    def __init__(self):
        super(plane_loss, self).__init__()

    def compute_loss(self, norm_lse, norm_est, norms, masks, device):
        """
        :param norm_lse: tensor with size (3, h, w)
        :param norm_est: tensor with size (3, h, w)
        :param norms: tensor with size (20, 3)
        :param masks: tensor with size (20, h, w)
        """
        assert norm_est.shape[1:] == masks.shape[1:]
        param_loss, cosine_loss = 0., 0.
        count = 0
        for i in range(masks.shape[0]):
            valid = (masks[i] > 0)
            if valid.sum() > 0:
                count += 1
                lse_valid = norm_lse[:, valid]  # (3, n)
                est_valid = norm_est[:, valid]  # (3, n)
                gt_valid = norms[i].view(-1, 1)
                _loss = torch.mean(torch.sum(torch.abs(est_valid - gt_valid), dim=0))
                param_loss += _loss
                lse_sim = torch.cosine_similarity(lse_valid, gt_valid, dim=0).mean()
                est_sim = torch.cosine_similarity(est_valid, gt_valid, dim=0).mean()
                sim_loss = lse_sim - est_sim if lse_sim > est_sim else torch.tensor(0.).to(device)
                cosine_loss += sim_loss
        if count != 0:
            return param_loss / count, cosine_loss / count
        else:
            return torch.tensor(0.).to(device), torch.tensor(0.).to(device)

    def forward(self, plane_lse, plane_est, plane_gt, ins_mask):
        n, c, h, w = ins_mask.shape
        device = ins_mask.device

        param_loss, cosine_loss = 0., 0.
        for i in range(n):
            l1_loss, cos_loss = self.compute_loss(plane_lse[i], plane_est[i], plane_gt[i], ins_mask[i], device)
            param_loss += l1_loss
            cosine_loss += cos_loss
        return param_loss / n, cosine_loss / n


class atrous_conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, apply_bn_first=True):
        super(atrous_conv, self).__init__()
        self.atrous_conv = nn.Sequential()
        if apply_bn_first:
            self.atrous_conv.add_module('first_bn', nn.BatchNorm2d(in_channels, momentum=0.01, affine=True,
                                                                   track_running_stats=True, eps=1.1e-5))

        self.atrous_conv.add_module('aconv_sequence', nn.Sequential(nn.ReLU(),
                                                                    nn.Conv2d(in_channels=in_channels,
                                                                              out_channels=out_channels * 2, bias=False,
                                                                              kernel_size=1, stride=1, padding=0),
                                                                    nn.BatchNorm2d(out_channels * 2, momentum=0.01,
                                                                                   affine=True,
                                                                                   track_running_stats=True),
                                                                    nn.ReLU(),
                                                                    nn.Conv2d(in_channels=out_channels * 2,
                                                                              out_channels=out_channels, bias=False,
                                                                              kernel_size=3, stride=1,
                                                                              padding=(dilation, dilation),
                                                                              dilation=dilation)))
    def forward(self, x):
        return self.atrous_conv.forward(x)


class NormalEstimator(nn.Module):
    def __init__(self):
        super(NormalEstimator, self).__init__()

        self.conv0 = nn.Conv2d(3, 32, 3, padding=1)
        self.daspp_3 = atrous_conv(32, 32, 3, apply_bn_first=False)
        self.daspp_6 = atrous_conv(32 * 2, 32, 6)
        self.daspp_12 = atrous_conv(32 * 3, 32, 12)
        self.daspp_18 = atrous_conv(32 * 4, 32, 18)
        self.daspp_24 = atrous_conv(32 * 5, 32, 24)
        self.daspp_conv = nn.Sequential(nn.Conv2d(32 * 6, 32, 1, bias=False),
                                        nn.ELU())
        self.get_plane = nn.Conv2d(32, 3, 1, bias=False)

    def least_square_estimator(self, k_inv_dot_xy1, seg_map, depth_map):
        plane_normals = torch.zeros_like(k_inv_dot_xy1)  # (3, h, w)
        ins_mask = []
        if len(seg_map.shape) == 3:
            for i in range(seg_map.shape[0]):
                mask = seg_map[i] > 0
                if mask.sum() != 0:
                    ins_mask.append(mask)
        else:
            for ins in torch.unique(seg_map):
                if ins != 0:
                    mask = seg_map == ins
                    ins_mask.append(mask)

        for mask in ins_mask:
            matrix_a = k_inv_dot_xy1[:, mask]  # (3, N)
            matrix_a = torch.mul(matrix_a, depth_map[mask])
            matrix_a_trans = matrix_a.permute(1, 0)  # (N, 3)
            matrix_mul = torch.matmul(torch.inverse(torch.matmul(matrix_a, matrix_a_trans)), matrix_a).sum(1)
            ins_normal = matrix_mul / torch.norm(matrix_mul)

            plane_normals[:, mask] = ins_normal.repeat(mask.sum(), 1).transpose(1, 0)
        return plane_normals

    def forward(self, depth_map, k_inv_dot_xy1, seg_map):
        n, c, h, w = depth_map.shape
        rough_normals = []
        for i in range(n):
            normals = self.least_square_estimator(k_inv_dot_xy1, seg_map[i], depth_map[i].squeeze())
            rough_normals.append(normals)
        rough_normals = torch.stack(rough_normals, dim=0)  # (n, 3, h, w)

        x = self.conv0(rough_normals)
        daspp_3 = self.daspp_3(x)
        concat4_2 = torch.cat([x, daspp_3], dim=1)
        daspp_6 = self.daspp_6(concat4_2)
        concat4_3 = torch.cat([concat4_2, daspp_6], dim=1)
        daspp_12 = self.daspp_12(concat4_3)
        concat4_4 = torch.cat([concat4_3, daspp_12], dim=1)
        daspp_18 = self.daspp_18(concat4_4)
        concat4_5 = torch.cat([concat4_4, daspp_18], dim=1)
        daspp_24 = self.daspp_24(concat4_5)
        concat4_daspp = torch.cat([x, daspp_3, daspp_6, daspp_12, daspp_18, daspp_24], dim=1)
        daspp_feat = self.daspp_conv(concat4_daspp)
        refine_normals = self.get_plane(daspp_feat)
        return rough_normals, refine_normals
