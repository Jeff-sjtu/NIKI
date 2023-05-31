import torch
import torch.nn as nn
import math

# from .builder import LOSS
from pytorch3d.transforms import matrix_to_rotation_6d, axis_angle_to_matrix

# from hybrik.models.layers.smpl.lbs import batch_rodrigues


def MMD_multiscale(x, y):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx
    dyy = ry.t() + ry - 2. * yy
    dxy = rx.t() + ry - 2. * zz

    XX, YY, XY = (torch.zeros(xx.shape).to(x.device),
                  torch.zeros(xx.shape).to(x.device),
                  torch.zeros(xx.shape).to(x.device))

    for a in [0.05, 0.2, 0.9]:
        XX += a**2 * (a**2 + dxx)**-1
        YY += a**2 * (a**2 + dyy)**-1
        XY += a**2 * (a**2 + dxy)**-1

    return torch.mean(XX + YY - 2. * XY)


def weighted_l1_loss(input, target, weights, size_average):
    input = input * 64
    target = target * 64
    out = torch.abs(input - target)
    out = out * weights
    if size_average and weights.sum() > 0:
        return out.sum() / weights.sum()
    else:
        return out.sum()


class LossTS(nn.Module):
    def __init__(self, smpl_layer):
        super(LossTS, self).__init__()
        self.smpl_layer = smpl_layer
        self.amp = 1 / math.sqrt(2 * math.pi)

        self.mmd = MMD_multiscale
        self.use_std = False

    def forward(self, output, labels, **kwargs):

        batch_size, seq_len = output['inv_pred_29joints'].shape[:2]

        w_smpl = labels['valid_smpl'].reshape(batch_size, seq_len)
        # ######=== pred_xyz -> inv -> rot ===######
        # === loss rot6d ===
        gt_rot_aa = labels['gt_theta'].reshape(batch_size * seq_len, 24, 3)
        gt_rot_mat = axis_angle_to_matrix(gt_rot_aa).reshape(batch_size * seq_len, 24, 3, 3)
        gt_rot_6d = matrix_to_rotation_6d(gt_rot_mat).reshape(batch_size * seq_len, 24, 6)

        gt_rot_weight = labels['theta_weight'].reshape(batch_size * seq_len, 24, 3)
        gt_rot_weight = gt_rot_weight[:, :, [0]] * gt_rot_weight[:, :, [1]] * gt_rot_weight[:, :, [2]]
        # gt_rot_mat = gt_rot_mat.reshape(batch_size * seq_len, 24, 9)
        # gt_rot9_weight = gt_rot_weight.expand_as(gt_rot_mat)
        gt_rot_weight = gt_rot_weight.expand_as(gt_rot_6d)

        # inv_pred2rotmat = output['inv_pred2rotmat'].reshape(-1, 24, 9)
        # loss_rot = torch.abs(inv_pred2rotmat - gt_rot_mat) ** 2 * gt_rot9_weight
        # loss_rot = loss_rot.sum() / gt_rot9_weight.sum()
        inv_pred2rot6d = output['inv_pred2rot6d'].reshape(-1, 24, 6)
        loss_rot = torch.abs(inv_pred2rot6d - gt_rot_6d) ** 2 * gt_rot_weight
        loss_rot = loss_rot.sum() / gt_rot_weight.sum()

        # === loss swing rot6d ===
        gt_rot_swing_6d = labels['gt_rot_swing_6d'].reshape(-1, 24, 6)
        inv_pred2swingrot6d = output['inv_pred2swingrot6d'].reshape(-1, 24, 6)

        loss_swing_rot6d = torch.abs(gt_rot_swing_6d - inv_pred2swingrot6d) ** 2 * gt_rot_weight
        loss_swing_rot6d = loss_swing_rot6d.sum() / gt_rot_weight.sum()

        # === loss regularize ze ===
        if not self.use_std:
            inv_pred2zes = output['inv_pred2zes'].reshape(batch_size * seq_len, -1)
            loss_inv_zes_mmd = self.mmd(inv_pred2zes, torch.randn_like(inv_pred2zes))

            loss_inv_zes_mmd = loss_inv_zes_mmd.mean()

            inv_pred2zet = output['inv_pred2zet'].reshape(batch_size * seq_len, -1)
            loss_inv_zet_mmd = self.mmd(inv_pred2zet, torch.randn_like(inv_pred2zet))

            loss_inv_zet_mmd = loss_inv_zet_mmd.mean()
        else:
            loss_inv_zes_mmd = 0
            loss_inv_zet_mmd = 0

        # === loss beta ===
        if 'inv_pred2beta' in output.keys():
            gt_beta = labels['gt_betas'].reshape(batch_size * seq_len, 10)
            beta_weight = labels['betas_weight'].reshape(batch_size * seq_len, 10)
            inv_pred2beta = output['inv_pred2beta'].reshape(batch_size * seq_len, 10)

            loss_inv_pred2beta = (gt_beta - inv_pred2beta) ** 2 * beta_weight
            loss_inv_pred2beta = loss_inv_pred2beta.sum() / beta_weight.sum()
        else:
            loss_inv_pred2beta = 0

        # === loss phi ===
        if 'inv_pred2phi' in output.keys():
            gt_phi = labels['gt_phi'].reshape(batch_size, seq_len, 23, 2)
            phi_weight = labels['phi_weight'].reshape(batch_size, seq_len, 23, 2)

            pred_phi = output['inv_pred2phi'].reshape(batch_size, seq_len, 23, 2)

            loss_inv_pred2phi = (gt_phi - pred_phi) ** 2 * phi_weight
            loss_inv_pred2phi = loss_inv_pred2phi.sum() / phi_weight.sum()
        else:
            loss_inv_pred2phi = 0

        # === loss jts ===
        gt_xyz_29 = labels['gt_xyz_29'].reshape(batch_size, seq_len, 29, 3)
        gt_xyz_29 = align_root(gt_xyz_29)
        xyz29_weight = labels['xyz_29_weight'].reshape(batch_size, seq_len, 29, -1)
        pred_xyz_29 = output['inv_pred_29joints'].reshape(batch_size, seq_len, 29, 3)
        pred_xyz_29 = align_root(pred_xyz_29)
        if self.use_std:
            std_t = output['inv_pred2zet'].reshape(batch_size, seq_len, -1)
            std_t = torch.std(std_t, dim=2, keepdim=True).unsqueeze(-1)

            loss_inv_pred2jts = torch.abs(gt_xyz_29 - pred_xyz_29) / (math.sqrt(2) * std_t + 1e-9) + torch.log(std_t / self.amp)
            loss_inv_pred2jts = (loss_inv_pred2jts * xyz29_weight)
            loss_inv_pred2jts = loss_inv_pred2jts.sum() / xyz29_weight.sum()
        else:
            loss_inv_pred2jts = torch.abs(pred_xyz_29 - gt_xyz_29) * xyz29_weight
            loss_inv_pred2jts = loss_inv_pred2jts.sum() / xyz29_weight.sum()

        # === loss independent ===
        if not self.use_std:
            mask_rot = gt_rot_weight.reshape(batch_size * seq_len, 24, 6)[:, :, 0]
            mask_rot = (mask_rot.sum(dim=1) >= 24).float()

            gt_qs = torch.cat((
                gt_rot_swing_6d.reshape(batch_size * seq_len, -1)[mask_rot > 0],
                torch.randn_like(inv_pred2zes[mask_rot > 0])), dim=1)
            pred_ps = torch.cat((
                inv_pred2swingrot6d.reshape(batch_size * seq_len, -1).detach()[mask_rot > 0],
                inv_pred2zes[mask_rot > 0]), dim=1)
            loss_inv_idp = self.mmd(gt_qs, pred_ps)
            loss_inv_idp = loss_inv_idp.mean()

            gt_qt = torch.cat((
                gt_rot_6d.reshape(batch_size * seq_len, -1)[mask_rot > 0],
                torch.randn_like(inv_pred2zet[mask_rot > 0])), dim=1)
            pred_pt = torch.cat((
                inv_pred2rot6d.reshape(batch_size * seq_len, -1).detach()[mask_rot > 0],
                inv_pred2zet[mask_rot > 0]), dim=1)
            loss_inv_idp += self.mmd(gt_qt, pred_pt)
        else:
            loss_inv_idp = 0

        # ######=== gt_xyz -> inv -> rot ===######
        # === loss swing rot6d ===
        inv_gt2swingrot6d = output['inv_gt2swingrot6d'].reshape(-1, 24, 6)
        assert tuple(gt_rot_weight.shape) == (batch_size * seq_len, 24, 6)
        gt_rot_weight = gt_rot_weight * w_smpl.reshape(batch_size * seq_len, 1, 1)

        loss_gt2swingrot6d = torch.abs(inv_gt2swingrot6d - gt_rot_swing_6d) ** 2 * gt_rot_weight
        loss_gt2swingrot6d = loss_gt2swingrot6d * w_smpl.reshape(batch_size * seq_len, 1, 1)
        loss_gt2swingrot6d = loss_gt2swingrot6d.sum() / gt_rot_weight.sum()

        # === loss rot6d ===
        inv_gt2rot6d = output['inv_gt2rot6d'].reshape(-1, 24, 6)
        loss_gt2rot6d = torch.abs(inv_gt2rot6d - gt_rot_6d) ** 2 * gt_rot_weight
        loss_gt2rot6d = loss_gt2rot6d * w_smpl.reshape(batch_size * seq_len, 1, 1)
        loss_gt2rot6d = loss_gt2rot6d.sum() / gt_rot_weight.sum()

        # === loss regularize ze ===
        inv_gt2zes = output['inv_gt2zes']
        loss_inv_zero_gt_zes = (inv_gt2zes - torch.zeros_like(inv_gt2zes)) ** 2
        loss_inv_zero_gt_zes = loss_inv_zero_gt_zes.reshape(batch_size * seq_len, 32)
        loss_inv_zero_gt_zes = loss_inv_zero_gt_zes * w_smpl.reshape(batch_size * seq_len, 1)
        loss_inv_zero_gt_zes = loss_inv_zero_gt_zes.mean()

        inv_gt2zet = output['inv_gt2zet']
        loss_inv_zero_gt_zet = (inv_gt2zet - torch.zeros_like(inv_gt2zet)) ** 2
        loss_inv_zero_gt_zet = loss_inv_zero_gt_zet.reshape(batch_size * seq_len, 46)
        loss_inv_zero_gt_zet = loss_inv_zero_gt_zet * w_smpl.reshape(batch_size * seq_len, 1)
        loss_inv_zero_gt_zet = loss_inv_zero_gt_zet.mean()

        # === loss beta ===
        if 'inv_gt2beta' in output.keys():
            inv_gt2beta = output['inv_gt2beta'].reshape(batch_size * seq_len, 10)
            loss_inv_gt2beta = (gt_beta - inv_gt2beta) ** 2 * beta_weight
            loss_inv_gt2beta = loss_inv_gt2beta.sum() / beta_weight.sum()
        else:
            loss_inv_gt2beta = 0

        # === loss phi ===
        if 'inv_gt2phi' in output.keys():
            inv_gt2phi = output['inv_gt2phi'].reshape(batch_size, seq_len, 23, 2)
            loss_inv_gt2phi = (gt_phi - inv_gt2phi) ** 2 * phi_weight
            loss_inv_gt2phi = loss_inv_gt2phi.sum() / phi_weight.sum()
        else:
            loss_inv_gt2phi = 0

        # === loss jts ===
        gt_xyz_29 = labels['gt_xyz_29'].reshape(batch_size, seq_len, 29, 3)
        gt_xyz_29 = align_root(gt_xyz_29)
        xyz29_weight = labels['xyz_29_weight'].reshape(batch_size, seq_len, 29, -1)
        xyz29_weight = xyz29_weight * w_smpl.reshape(batch_size, seq_len, 1, 1)
        inv_gt2joints = output['inv_gt2joints'].reshape(batch_size, seq_len, 29, 3)
        inv_gt2joints = align_root(inv_gt2joints)
        loss_inv_gt2jts = torch.abs(inv_gt2joints - gt_xyz_29) * xyz29_weight
        loss_inv_gt2jts = loss_inv_gt2jts.sum() / xyz29_weight.sum()

        # ######=== gt_rot -> fwd -> xyz ===######
        # === loss xyz 29 ===
        gt_xyz_29 = labels['gt_xyz_29'].reshape(batch_size, seq_len, 29, -1)
        gt_xyz_29 = align_root(gt_xyz_29)
        xyz29_weight = labels['xyz_29_weight'].reshape(batch_size, seq_len, 29, -1)
        xyz29_weight = xyz29_weight * w_smpl.reshape(batch_size, seq_len, 1, 1)
        fwd_gt_2_29joints = output['fwd_gt_2_29joints'].reshape(batch_size, seq_len, 29, -1)
        # pred_sigma = output['pred_sigma'].reshape(batch_size, seq_len, 29, -1)

        loss_xyz29 = torch.abs(gt_xyz_29 - fwd_gt_2_29joints)
        loss_xyz29 = (loss_xyz29 * xyz29_weight)
        loss_xyz29 = loss_xyz29.sum() / xyz29_weight.sum()

        # === loss beta ===
        if 'fwd_gt_2_betas' in output.keys():
            gt_beta = labels['gt_betas'].reshape(batch_size * seq_len, 10)
            beta_weight = labels['betas_weight'].reshape(batch_size * seq_len, 10)
            fwd_gt2betas = output['fwd_gt_2_betas'].reshape(batch_size * seq_len, 10)

            loss_fwd_gt2beta = (gt_beta - fwd_gt2betas) ** 2 * beta_weight
            loss_fwd_gt2beta = loss_fwd_gt2beta.sum() / beta_weight.sum()
        else:
            loss_fwd_gt2beta = 0
        # === loss swing6d ===
        fwd_gt2swing6d = output['fwd_gt_2_swing6d'].reshape(batch_size * seq_len, 24, 6)
        loss_fwd_gt2swingrot6d = torch.abs(fwd_gt2swing6d - gt_rot_swing_6d) ** 2 * gt_rot_weight
        loss_fwd_gt2swingrot6d = loss_fwd_gt2swingrot6d.sum() / gt_rot_weight.sum()

        # === loss phi ===
        gt_phi = labels['gt_phi'].reshape(batch_size * seq_len, 23, 2)
        phi_weight = labels['phi_weight'].reshape(batch_size * seq_len, 23, 2)
        phi_weight = phi_weight * w_smpl.reshape(batch_size * seq_len, 1, 1)
        fwd_gt2phis = output['fwd_gt_2_phis'].reshape(batch_size * seq_len, 23, 2)
        loss_fwd_gt2phi = (gt_phi - fwd_gt2phis) ** 2 * phi_weight
        # loss_fwd_gt2phi = torch.abs(gt_phi - fwd_gt2phis) * phi_weight
        loss_fwd_gt2phi = loss_fwd_gt2phi.sum() / phi_weight.sum()

        # ######=== zx losses ===######
        if 'fwd_gt_2_zx1' in output.keys():
            zx1 = output['fwd_gt_2_zx1']
            zx2 = output['fwd_gt_2_zx2']
            loss_zx = torch.mean(zx1 ** 2) + torch.mean(zx2 ** 2)
        else:
            zx = output['fwd_gt_2_zx']
            loss_zx = torch.mean(zx ** 2)

        # ######=== add all losses ===######
        # === inv pred ===
        loss_total = loss_rot * 1.0 + loss_swing_rot6d * 1.0
        loss_total += (loss_inv_zes_mmd + loss_inv_zet_mmd) * 1.0 + loss_inv_pred2beta * 5.0 + loss_inv_pred2phi * 1.0
        # loss_total += loss_inv_pred2beta * 5.0 + loss_inv_pred2phi * 1.0
        loss_total += loss_inv_pred2jts * 1.0
        loss_total += loss_inv_idp * 1
        # === inv gt ===
        loss_total += loss_gt2swingrot6d * 1.0 + loss_gt2rot6d * 1.0
        loss_total += (loss_inv_zero_gt_zes + loss_inv_zero_gt_zet) * 0.1 + loss_inv_gt2beta * 5.0 + loss_inv_gt2phi * 1.0
        loss_total += loss_inv_gt2jts * 1.0
        # === fwd gt ===
        loss_total += loss_xyz29 * 1.0 + loss_fwd_gt2beta * 1.0
        loss_total += loss_fwd_gt2swingrot6d * 1.0 + loss_fwd_gt2phi * 1.0

        loss_total += loss_zx

        loss_dict = {
            'l_i_rot': loss_rot.detach().cpu().numpy(),
            # 'l_i_beta': loss_inv_pred2beta.detach().cpu().numpy(),
            'l_f_xyz29': loss_xyz29.detach().cpu().numpy(),
            'l_i_0_ze': (loss_inv_zero_gt_zes + loss_inv_zero_gt_zet).detach().cpu().numpy()
        }

        return loss_total, loss_dict

    def projection(self, pred_xyz_29, pred_cam, focal=1000.0):
        # pred_joints: [B, 29, 3]
        pred_xyz_29 = pred_xyz_29 - pred_xyz_29[:, [1, 2], :].mean(dim=1, keepdim=True)
        cam_depth = focal / (256.0 * pred_cam[:, :, [0]] + 1e-9)  # [B, T, 1]
        transl = torch.cat((pred_cam[:, :, 1:], cam_depth), dim=2)
        pred_xyz_29_cam = pred_xyz_29 + transl[:, :, None, :]  # [B, T, K, 3]

        pred_uv_29 = pred_xyz_29_cam[:, :, :, :2] * focal / (pred_xyz_29_cam[:, :, :, [2]] * 256.0)
        return pred_uv_29


def align_root(xyz_29):
    shape = xyz_29.shape
    xyz_29 = xyz_29.reshape(-1, 29, 3)
    xyz_29 = xyz_29 - xyz_29[:, [0], :]
    xyz_29 = xyz_29.reshape(*shape)
    return xyz_29
