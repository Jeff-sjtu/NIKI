import numpy as np
import torch
import torch.nn as nn
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d, axis_angle_to_matrix

from .layers.real_nvp import RealNVP
from niki.models.layers.smpl.SMPL import SMPL_layer
from easydict import EasyDict as edict


def get_nets(inp_dim):
    def nets():
        return nn.Sequential(nn.Linear(inp_dim, 512), nn.LeakyReLU(), nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, inp_dim), nn.Tanh())
    
    return nets


def get_nett(inp_dim):
    def nett():
        return nn.Sequential(nn.Linear(inp_dim, 512), nn.LeakyReLU(), nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, inp_dim))

    return nett


class FlowIK_camnet(nn.Module):
    def __init__(
        self,
        opt,
        **cfg
    ):

        super(FlowIK_camnet, self).__init__()

        self.seqlen = opt.seq_len

        self.regressor = FlowRegressor(opt)

    def forward(self, inp, **kwargs):

        smpl_output = self.regressor(inp, **kwargs)

        return smpl_output
    
    def forward_getcam(self, inp, **kwargs):
        return self.regressor.forward_getcam(inp, **kwargs)


class FlowRegressor(nn.Module):
    def __init__(self, opt, dtype=torch.float32):
        super(FlowRegressor, self).__init__()

        h36m_jregressor = np.load('./model_files/J_regressor_h36m.npy')
        self.smpl_layer = SMPL_layer(
            './model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
            h36m_jregressor=h36m_jregressor,
            dtype=torch.float32
        )

        self.inp_dim = 29 * 3 + 29 * 1
        self.out_dim = 24 * 6
        self.ze_dim = 32
        self.total_dim = self.out_dim + self.ze_dim
        self.zx_dim = self.total_dim - self.inp_dim

        masks = torch.from_numpy(np.array([
            [0, 1] * (self.total_dim // 2),
            [1, 0] * (self.total_dim // 2)
        ] * 8).astype(np.float32))

        self.flow = RealNVP(get_nets(self.total_dim), get_nett(self.total_dim), masks)
        self.alpha = 0.9
        self.running_mean_ze = None

        self.camnet = CameraNet()

    def forward(self, inp, target=None, **kwargs):
        input_joints = inp['pred_xyz_29']
        pred_beta = inp['pred_beta']
        input_sigma = inp['pred_sigma']

        # input size NTF
        batch_size, seqlen = input_joints.shape[:2]

        input_joints = input_joints.reshape(batch_size * seqlen, 29 * 3)
        input_joints = align_root(input_joints)
        input_sigma = input_sigma.reshape(batch_size * seqlen, 29 * 1)

        zx = 1e-3 * torch.randn((batch_size * seqlen, self.zx_dim), device=input_joints.device, dtype=input_joints.dtype)

        input_joints = torch.cat((input_joints, input_sigma, zx), dim=1)
        # forward pass: [-1, 29 * 6]
        out, _ = self.flow.inverse_p(input_joints)
        out = out.reshape(batch_size * seqlen, self.total_dim)
        inv_pred2rot6d = out[:, :24 * 6].contiguous().reshape(batch_size * seqlen, 24, 6)
        # z_e for error, z_s for shape
        inv_pred2ze = out[:, -self.ze_dim:]

        if target is not None:
            # only for training
            # ====== inverse pass: gt joints ======
            gt_jts29 = target['gt_xyz_29'].reshape(batch_size * seqlen, 29 * 3)
            gt_betas = target['gt_betas'].reshape(batch_size * seqlen, 10)
            gt_jts29 = align_root(gt_jts29)
            gt_sigma29 = 1e-6 * torch.randn((batch_size * seqlen, 29 * 1), device=input_joints.device, dtype=input_joints.dtype).clamp_min(min=0)

            zx = 1e-3 * torch.randn((batch_size * seqlen, self.zx_dim), device=input_joints.device, dtype=input_joints.dtype)
            gt_jts29 = torch.cat((gt_jts29, gt_sigma29, zx), dim=1)

            inv_gt_out, _ = self.flow.inverse_p(gt_jts29)
            inv_gt2rot6d = inv_gt_out[:, :24 * 6].contiguous().reshape(batch_size * seqlen, 24, 6)
            # z_e for error, z_s for shape
            inv_gt2ze = inv_gt_out[:, -self.ze_dim:]
            inv_gt2rotmat = rotation_6d_to_matrix(inv_gt2rot6d).reshape(batch_size * seqlen, 24, 3, 3)

            inv_gt_output = self.smpl_layer(
                pose_axis_angle=inv_gt2rotmat.reshape(batch_size * seqlen, 24, 9),
                betas=gt_betas.reshape(batch_size * seqlen, 10),
                global_orient=None,
                pose2rot=False,
                return_29_jts=True
            )
            inv_gt_29joints = inv_gt_output['joints']

            # ====== forward pass: gt rotmat ======
            gt_rot_aa = target['gt_theta'].reshape(batch_size * seqlen, 24, 3)
            gt_rot_mat = axis_angle_to_matrix(gt_rot_aa)
            gt_rot_6d = matrix_to_rotation_6d(gt_rot_mat).reshape(batch_size * seqlen, 24 * 6)

            zero_ze = torch.zeros_like(inv_gt2ze)
            latent = torch.cat((gt_rot_6d, zero_ze), dim=1)

            # [B, 29 * 6], [B,]
            fwd_gt2jts, log_det_J = self.flow.forward_p(latent)
            fwd_gt2zx = fwd_gt2jts[:, -self.zx_dim:]
            fwd_gt2sigma = fwd_gt2jts[:, 29 * 3:29 * 4]

            fwd_gt2jts = fwd_gt2jts[:, :29 * 3]

        assert torch.isnan(inv_pred2rot6d).sum() == 0, inv_pred2rot6d

        inv_pred_rot = rotation_6d_to_matrix(inv_pred2rot6d).reshape(batch_size * seqlen, 24, 9)

        assert torch.isnan(inv_pred_rot).sum() == 0, inv_pred_rot

        inv_pred_output = self.smpl_layer(
            pose_axis_angle=inv_pred_rot.reshape(batch_size * seqlen, 24, 9),
            betas=pred_beta.reshape(batch_size * seqlen, 10),
            global_orient=None,
            pose2rot=False,
            return_29_jts=True
        )

        pred_shape = pred_beta.reshape(batch_size * seqlen, 10)

        inv_pred_vertices = inv_pred_output['vertices']

        inv_pred_29joints = inv_pred_output['joints']
        pred_xyz_17 = inv_pred_output['joints_from_verts']

        output = edict(
            rotmat=inv_pred_output['rot_mats'].reshape(batch_size, seqlen, 24, 3, 3),
            inv_pred2rot6d=inv_pred2rot6d.reshape(batch_size, seqlen, 24, 6),
            inv_pred2rotmat=inv_pred_rot.reshape(batch_size, seqlen, 24, 3, 3),
            pred_shape=pred_shape,
            verts=inv_pred_vertices.reshape(batch_size, seqlen, -1, 3),
            inv_pred_29joints=inv_pred_29joints.reshape(batch_size, seqlen, 29, 3),
            pred_xyz_29=inv_pred_29joints.reshape(batch_size, seqlen, 29, 3),
            pred_xyz_29_struct=inv_pred_29joints.reshape(batch_size, seqlen, 29, 3),
            pred_xyz_17=pred_xyz_17.reshape(batch_size, seqlen, 17, 3),
            pred_beta=pred_beta,
            pred_phi=inp['pred_phi'],
            pred_cam=inp['pred_cam'],
            # pred_24joints=input_joints.reshape(batch_size, seqlen, 29, 3)[:, :, :24, :],
            inv_pred2ze=inv_pred2ze.reshape(batch_size, seqlen, self.ze_dim),
        )
        if target is not None:
            output['fwd_gt_2_29joints'] = fwd_gt2jts.reshape(batch_size, seqlen, 29, 3)
            output['fwd_gt_2_sigma'] = fwd_gt2sigma.reshape(batch_size, seqlen, 29, 1)
            output['fwd_gt_2_zx'] = fwd_gt2zx.reshape(batch_size, seqlen, self.zx_dim)
            output['log_det_J'] = log_det_J.reshape(batch_size, seqlen, 1)
            output['inv_gt2rot6d'] = inv_gt2rot6d.reshape(batch_size, seqlen, 24, 6)
            output['inv_gt2rotmat'] = inv_gt2rotmat.reshape(batch_size, seqlen, 24, 3, 3)
            output['inv_gt2ze'] = inv_gt2ze.reshape(batch_size, seqlen, self.ze_dim)
            output['inv_gt2joints'] = inv_gt_29joints.reshape(batch_size, seqlen, 29, 3)

        return output

    def forward_getcam(self, inp, target=None, **kwargs):
        pred_uv = inp['pred_uv']

        batch_size, seq_len = pred_uv.shape[:2]
        img_center = inp['img_center'].reshape(batch_size * seq_len, 2)

        with torch.no_grad():
            niki_out = self.forward(inp, **kwargs)
        
        pred_xyz_29 = niki_out.pred_xyz_29.reshape(batch_size*seq_len, 29, 3)
        inv_pred2cam = self.camnet(
            pred_uv.reshape(batch_size*seq_len, 29, 2),
            pred_xyz_29,
            inp['pred_sigma'].reshape(batch_size*seq_len, 29, 1),
            inp['pred_cam'].reshape(batch_size*seq_len, 3),
        )

        niki_out['inv_pred2cam'] = inv_pred2cam
        # print(inv_pred2cam[0], inp['pred_cam'][0])
        inv_pred2uv = self.projection2uv(inv_pred2cam, pred_xyz_29, img_center)
        niki_out['inv_pred2uv'] = inv_pred2uv.reshape(batch_size, seq_len, 29, 2)

        inv_pred_vertices = niki_out.verts.reshape(batch_size*seq_len, 6890, 3)
        inv_pred_29joints = pred_xyz_29.clone()
        input_sigma = inp['pred_sigma'].reshape(batch_size*seq_len, 29, 1)
        transl_imgcenter = 0
        transl = 0
        if not self.training:
            inv_pred_vertices = inv_pred_vertices - inv_pred_29joints[:, [0]]
            inv_pred_29joints = inv_pred_29joints - inv_pred_29joints[:, [0]]
            
            transl = inv_pred2cam.clone()
            transl[:, :2] = inv_pred2cam[:, 1:]
            transl[:, 2] = 1000 / (256.0 * inv_pred2cam[:, 0])

            img_center_bbox = (img_center / 256.0).unsqueeze(1)
            dxy = (inv_pred2uv[:, :24] - img_center_bbox) * \
                        (inv_pred_29joints[:, :24, [2]] + transl[:, [2]].unsqueeze(1)) * 256.0 / 1000.0

            dxy = dxy - inv_pred_29joints[:, :24, :2]
            
            score = torch.clamp(1-5*input_sigma[:, :24], min=0, max=1)
            dxy = (dxy * score).sum(dim=1) / score.sum(dim=1)

            transl_imgcenter = transl.clone()
            transl_imgcenter[:, :2] = dxy

        niki_out['transl'] = transl_imgcenter.reshape(batch_size, seq_len, 3)
        niki_out['verts'] = inv_pred_vertices.reshape(batch_size, seq_len, 6890, 3)
        return niki_out
    
    def projection2uv(self, cam_param, pred_xyz, img_center):
        img_center = img_center / 256.0
        img_center = img_center.unsqueeze(1)
        img_center[:] = 0.0

        trans_xy = cam_param[:, 1:].unsqueeze(1)
        s = cam_param[:, [0]].unsqueeze(1)

        pred_xyz = pred_xyz - pred_xyz[:, [1, 2]].mean(dim=1, keepdim=True)

        # (xy + d_xy) = (uv - c) * (z * 256 / 1000 + 1/s)
        pred_uv = (pred_xyz[:, :, :2] + trans_xy) / (pred_xyz[:, :, [2]] *256.0/1000 + 1/s) + img_center

        return pred_uv


class CameraNet(nn.Module):
    def __init__(self):
        super(CameraNet, self).__init__()
        self.inp_dim = 29 * 3 + 29 * 3 + 3 # uv + sigma + refined_xyz + init scale
        hidden_dim = 1024
        self.camera_net = nn.Sequential(
            nn.Linear(self.inp_dim, hidden_dim),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

        init_cam = torch.tensor([0.8, 0, 0])

        self.register_buffer('init_cam', init_cam.float())
    
    def forward(self, pred_uv_raw, pred_xyz, pred_sigma, pred_cam_raw):
        batch_size = pred_uv_raw.shape[0]
        inp = torch.cat([
            pred_uv_raw.reshape(batch_size, -1), 
            pred_xyz.reshape(batch_size, -1), 
            pred_sigma.reshape(batch_size, -1), 
            pred_cam_raw.reshape(batch_size, -1)
        ], dim=1)

        out = self.camera_net(inp) + self.init_cam

        return out



def align_root(xyz_29):
    shape = xyz_29.shape
    xyz_29 = xyz_29.reshape(-1, 29, 3)
    xyz_29 = xyz_29 - xyz_29[:, [0], :]
    xyz_29 = xyz_29.reshape(*shape)
    return xyz_29
