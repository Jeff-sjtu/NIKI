import math
import os
import os.path as osp
import random

import cv2
import joblib
import numpy as np
import torch
import torch.utils.data as data
from pytorch3d.transforms import axis_angle_to_matrix

from niki.models.layers.smpl.SMPL import SMPL_layer
from niki.utils.data_utils import split_into_chunks, split_into_chunks_filtered
from niki.utils.occlusion_utils import translate_img_path
from niki.utils.pose_utils import (calc_cam_scale_trans,
                                   cam2pixel_temporal, normalize_uv_temporal,
                                   reconstruction_error)

from .amass_dataset_temporal import amass_dataset_temporal
from .coco_temporal import coco_temporal
from .hp3d_dataset_temporal import hp3d_dataset_temporal


def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / (norm_quat.norm(p=2, dim=1, keepdim=True) + 1e-8)
    w, x, y, z = norm_quat[:, 0], norm_quat[:,
                                            1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def batch_rodrigues(axisang):
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle + 1e-8)

    rx, ry, rz = torch.split(axisang_normalized, 1, dim=1)
    zeros = torch.zeros_like(rx)

    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((axisang.shape[0], 3, 3))
    ident = torch.eye(3, dtype=axisang.dtype,
                      device=axisang.device).unsqueeze(dim=0)

    angle = angle[:, :, None]
    rot_mat = ident + torch.sin(angle) * K + \
        (1 - torch.cos(angle)) * torch.bmm(K, K)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat


class mix_temporal_dataset_wamass(data.Dataset):
    def __init__(self, gt_paths, pred_paths, train=True, usage='xyz', occlusion=True):
        self.h36m_dataset = naive_dataset_temporal(
            gt_paths[0], pred_paths[0], 'h36m', train=train, usage=usage, use_amass=True, occlusion=occlusion)
        self.pw3d_dataset = naive_dataset_temporal(
            gt_paths[1], pred_paths[1], 'pw3d', train=train, usage=usage, use_amass=True, occlusion=occlusion)
        # self.hp3d_dataset = hp3d_dataset_temporal(gt_paths[2], pred_paths[2], 'hp3d', train=train, usage=usage, use_amass=True)
        self.amass_dataset = amass_dataset_temporal(
            usage=usage, occlusion=occlusion)

    def __len__(self):
        return len(self.h36m_dataset)

    def __getitem__(self, idx):
        p = random.random()
        if p < 0.6:
            new_idx = random.randint(0, len(self.h36m_dataset) - 1)
            return self.h36m_dataset[new_idx]
        elif p < 0.8:
            new_idx = random.randint(0, len(self.pw3d_dataset) - 1)
            return self.pw3d_dataset[new_idx]
        else:
            new_idx = random.randint(0, len(self.amass_dataset) - 1)
            return self.amass_dataset[new_idx]


class mix_temporal_dataset_full_woamass(data.Dataset):
    def __init__(self, gt_paths, pred_paths, opt, train=True, usage='xyz', occlusion=True, use_pretrained_feat=True, img_feat_size=1024, load_img=False):
        seq_len = opt.seq_len
        self.use_pretrained_feat = use_pretrained_feat
        self.use_flip = ('feature_flipped' in gt_paths[0])
        self.h36m_dataset = naive_dataset_temporal(
            gt_paths[0], pred_paths[0], 'h36m', train=train,
            usage=usage, use_amass=True, occlusion=occlusion,
            use_pretrained_feat=use_pretrained_feat, seq_len=seq_len,
            load_img=load_img)
        self.pw3d_dataset = naive_dataset_temporal(
            gt_paths[1], pred_paths[1], 'pw3d', train=train,
            usage=usage, use_amass=True, occlusion=occlusion,
            use_pretrained_feat=use_pretrained_feat, seq_len=seq_len,
            load_img=load_img)
        self.hp3d_dataset = hp3d_dataset_temporal(
            gt_paths[2], pred_paths[2], 'hp3d', train=train,
            usage=usage, use_amass=True, use_pretrained_feat=use_pretrained_feat,
            seq_len=seq_len, load_img=load_img)

    def __len__(self):
        data_len = len(self.h36m_dataset) // 3
        if not self.use_pretrained_feat:
            data_len = data_len // 4
        return data_len

    def __getitem__(self, idx):
        p = random.random()
        if p < 0.4:
            new_idx = random.randint(0, len(self.h36m_dataset) - 1)
            return self.h36m_dataset[new_idx]
        elif p < 0.8:
            new_idx = random.randint(0, len(self.pw3d_dataset) - 1)
            return self.pw3d_dataset[new_idx]
        else:
            new_idx = random.randint(0, len(self.hp3d_dataset) - 1)
            return self.hp3d_dataset[new_idx]
        # else:
        #     new_idx = random.randint(0, len(self.amass_dataset) - 1)
        #     return self.amass_dataset[new_idx]


class mix_temporal_dataset_full_wocc(data.Dataset):
    def __init__(self, gt_paths, pred_paths, opt, train=True, usage='xyz', occlusion=True, use_pretrained_feat=True,
                 wrong_flip_aug=False, thres_ps=None, simulated_amass=False, use_joint_part_seg=False, img_feat_size=1024):
        seq_len = opt.seq_len
        self.use_pretrained_feat = use_pretrained_feat
        self.use_flip = ('feature_flipped' in gt_paths[0])
        self.h36m_dataset = naive_dataset_temporal(
            gt_paths[0], pred_paths[0], 'h36m', train=train, usage=usage, use_amass=True, use_pretrained_feat=use_pretrained_feat, wrong_flip_aug=wrong_flip_aug, seq_len=seq_len)
        self.pw3d_dataset = naive_dataset_temporal(
            gt_paths[1], pred_paths[1], 'pw3d', train=train, usage=usage, use_amass=True, use_pretrained_feat=use_pretrained_feat, wrong_flip_aug=wrong_flip_aug, seq_len=seq_len)
        self.hp3d_dataset = hp3d_dataset_temporal(
            gt_paths[2], pred_paths[2], 'hp3d', train=train, usage=usage, use_amass=True, use_pretrained_feat=use_pretrained_feat, seq_len=seq_len)
        if not simulated_amass:
            self.amass_dataset = amass_dataset_temporal(
                usage=usage, occlusion=occlusion, use_pretrained_feat=use_pretrained_feat, use_flip=self.use_flip, wrong_flip_aug=wrong_flip_aug, seq_len=seq_len, img_feat_size=img_feat_size)
        else:
            raise NotImplementedError
            # self.amass_dataset = amass_dataset_temporal_simulated(
            #     usage=usage, occlusion=occlusion, use_pretrained_feat=use_pretrained_feat, use_flip=self.use_flip, wrong_flip_aug=wrong_flip_aug, use_joint_part_seg=use_joint_part_seg)
        # occ_path = 'exp/video_predict/feat_flipped/3dpw_train_occ_db_xyz17_uv24new_addcam_bbox_pred_amb_add_cam.pt'
        if len(gt_paths) == 3:
            occ_path = gt_paths[1]
        else:
            occ_path = gt_paths[3]

        self.pw3d_dataset_occ = naive_dataset_temporal(
            occ_path, '', 'pw3d', train=train, usage=usage, use_amass=True, occlusion=True, use_pretrained_feat=use_pretrained_feat, seq_len=seq_len)
        # occ_path = 'exp/video_predict/feat_flipped/h36m_train_25fps_occ_smpl_db_xyz17new_add_cam_bbox_pred_amb.pt'
        # self.h36m_dataset_occ = naive_dataset_temporal(
        #     occ_path, '', 'h36m', train=train, usage=usage, use_amass=True, occlusion=True, use_pretrained_feat=use_pretrained_feat)
        if thres_ps is None:
            self.thres_ps = [0.4, 0.65, 0.8, 0.95]
        else:
            self.thres_ps = thres_ps

    def __len__(self):
        data_len = len(self.h36m_dataset) // 3
        if not self.use_pretrained_feat:
            data_len = data_len // 4
        return data_len

    def __getitem__(self, idx):
        p = random.random()

        if False:
            new_idx = random.randint(0, len(self.h36m_dataset_occ) - 1)
            return self.h36m_dataset_occ[new_idx]
        elif p < self.thres_ps[0]:
            new_idx = random.randint(0, len(self.h36m_dataset) - 1)
            return self.h36m_dataset[new_idx]
        elif p < self.thres_ps[1]:
            new_idx = random.randint(0, len(self.pw3d_dataset) - 1)
            return self.pw3d_dataset[new_idx]
        elif p < self.thres_ps[2]:
            new_idx = random.randint(0, len(self.hp3d_dataset) - 1)
            return self.hp3d_dataset[new_idx]
        elif p < self.thres_ps[3]:
            new_idx = random.randint(0, len(self.amass_dataset) - 1)
            return self.amass_dataset[new_idx]
        else:
            new_idx = random.randint(0, len(self.pw3d_dataset_occ) - 1)
            return self.pw3d_dataset_occ[new_idx]


class mix_temporal_dataset_full_wcoco(data.Dataset):
    def __init__(self, gt_paths, pred_paths, opt, train=True, usage='xyz', occlusion=True, use_pretrained_feat=True, img_feat_size=1024, load_img=False):
        seq_len = opt.seq_len
        self.use_pretrained_feat = use_pretrained_feat
        self.use_flip = ('feature_flipped' in gt_paths[0])
        self.h36m_dataset = naive_dataset_temporal(
            gt_paths[0], pred_paths[0], 'h36m', train=train,
            usage=usage, use_amass=True, occlusion=occlusion,
            use_pretrained_feat=use_pretrained_feat, seq_len=seq_len,
            load_img=load_img)
        self.pw3d_dataset = naive_dataset_temporal(
            gt_paths[1], pred_paths[1], 'pw3d', train=train,
            usage=usage, use_amass=True, occlusion=occlusion,
            use_pretrained_feat=use_pretrained_feat, seq_len=seq_len,
            load_img=load_img)
        self.coco_eft_dataset = coco_temporal(
            # 'exp/pt_files_3doh/person_keypoints_train2017_pred.pt', 'coco_eft', train=True, seq_len=seq_len
            'new_coco.pt', 'coco_eft', train=True, seq_len=seq_len
        )

    def __len__(self):
        data_len = len(self.h36m_dataset) // 3
        if not self.use_pretrained_feat:
            data_len = data_len // 4
        return data_len

    def __getitem__(self, idx):
        p = random.random()
        if p < 0.3:
            new_idx = random.randint(0, len(self.h36m_dataset) - 1)
            return self.h36m_dataset[new_idx]
        elif p < 0.3:
            new_idx = random.randint(0, len(self.pw3d_dataset) - 1)
            return self.pw3d_dataset[new_idx]
        else:
            new_idx = random.randint(0, len(self.coco_eft_dataset) - 1)
            return self.coco_eft_dataset[new_idx]
        # else:
        #     new_idx = random.randint(0, len(self.amass_dataset) - 1)
        #     return self.amass_dataset[new_idx]



class naive_dataset_temporal(data.Dataset):
    EVAL_JOINTS = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]

    joint_pairs_29 = ((1, 2), (4, 5), (7, 8),
                      (10, 11), (13, 14), (16, 17), (18, 19), (20, 21),
                      (22, 23), (25, 26), (27, 28))

    joint_pairs_24 = ((1, 2), (4, 5), (7, 8),
                      (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))

    joint_pairs_17 = ((1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16))

    skeleton_29jts = [
        [0, 1], [0, 2], [0, 3],  # 2
        [1, 4], [2, 5], [3, 6],  # 5
        [4, 7], [5, 8], [6, 9],  # 8
        [7, 10], [8, 11], [9, 12], [9, 13], [9, 14],  # 13
        [12, 15], [13, 16], [14, 17],  # 16
        [16, 18], [17, 19],  # 18
        [18, 20], [19, 21],  # 20
        [20, 22], [21, 23],  # 22
        [15, 24], [22, 25], [23, 26], [10, 27], [11, 28]  # 27
    ]

    def __init__(self, gt_path, pred_path, dataset_name, train=False, usage='phi', only_read=True, return_jts2d=False,
                 use_amass=False, occlusion=True, use_pretrained_feat=True, wrong_flip_aug=False, seq_len=16, get_gt_uv=True):
        self.root_idx_17 = 0
        self.root_idx_smpl = 0

        self.occlusion = occlusion
        self.dataset_name = dataset_name
        self.train = train
        self.db_gt = self.load_db(gt_path, only_read)
        if len(pred_path) > 0:
            self.db_pred = self.load_db(pred_path, only_read)
        else:
            self.db_pred = self.db_gt

        self.seq_len = seq_len
        if self.train:
            overlap = (self.seq_len - 1) / float(self.seq_len)
        else:
            overlap = 0

        self.stride = int(self.seq_len * (1 - overlap) + 0.5)

        if self.dataset_name == 'pw3d' or self.dataset_name == '3dpw':
            self.vid_indices, wrong_indices = split_into_chunks_filtered(
                self.db_gt, self.seq_len, self.stride)
        else:
            self.vid_indices = split_into_chunks(
                self.db_gt, self.seq_len, self.stride, filtered=False)
            wrong_indices = []

        self.wrong_start_indices = [item[0] for item in wrong_indices]

        self.usage = usage

        downsample = False
        if downsample and (not train):
            if self.dataset_name == 'h36m':
                self.vid_indices = self.vid_indices[::16]
            elif self.dataset_name == 'pw3d' or self.dataset_name == '3dpw':
                self.vid_indices = self.vid_indices[::8]

        self.return_jts2d = True
        self.get_gt_uv = get_gt_uv

        self.cam_record = []
        self.use_amass = use_amass

        self.use_pretrained_feat = use_pretrained_feat
        self.input_size = 256

        self.wrong_flip_aug = wrong_flip_aug
        print('wrong_flip_aug', wrong_flip_aug)

        h36m_jregressor = np.load('./model_files/J_regressor_h36m.npy')
        self.smpl_layer = SMPL_layer(
            './model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
            h36m_jregressor=h36m_jregressor,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, idx):
        start_index, end_index = self.vid_indices[idx]
        images = 0

        if self.train:
            target = self._get_item_xyz_train(idx)
        else:
            target = self._get_item_xyz_val(idx)

        target['images'] = images
        return target

    def _get_item_xyz_train(self, idx):
        start_index, end_index = self.vid_indices[idx]
        kp_3d_29 = self.get_sequence(start_index, end_index, self.db_gt['xyz_29'])
        kp_3d_29_pred = self.get_sequence(start_index, end_index, self.db_pred['pred_xyz_29']) * 2.2
        kp_3d_24_struct_pred = self.get_sequence(start_index, end_index, self.db_pred['pred_xyz_24_struct']) * 2.2
        pred_score = self.get_sequence(start_index, end_index, self.db_pred['pred_scores']) * 0.1
        pred_score = pred_score.reshape(self.seq_len, 29, 1)
        pred_sigma = self.get_sequence(start_index, end_index, self.db_pred['pred_sigma'])
        pred_sigma = pred_sigma.reshape(self.seq_len, 29, 1)

        kp_3d_29 = kp_3d_29 - kp_3d_29[:, [1, 2], :].mean(axis=1, keepdims=True)
        kp_3d_29_pred = kp_3d_29_pred - kp_3d_29_pred[:, [1, 2], :].mean(axis=1, keepdims=True)
        kp_3d_24_struct_pred = kp_3d_24_struct_pred - kp_3d_24_struct_pred[:, [1, 2], :].mean(axis=1, keepdims=True)

        kp_3d_17 = self.get_sequence(start_index, end_index, self.db_gt['xyz_17'])
        kp_3d_17 = kp_3d_17 - kp_3d_17[:, [1, 4], :].mean(axis=1, keepdims=True)

        xyz_29_weight = np.ones_like(kp_3d_29)
        xyz_17_weight = np.ones_like(kp_3d_17)

        pred_betas = self.get_sequence(start_index, end_index, self.db_pred['pred_betas']).reshape(self.seq_len, -1)[:, :10]
        gt_betas = self.get_sequence(start_index, end_index, self.db_gt['shape']).reshape(self.seq_len, 10)
        beta_rand_mask = np.random.rand(self.seq_len) > 0.5
        pred_betas[beta_rand_mask] = gt_betas[beta_rand_mask]

        add_beta_noise = True
        if add_beta_noise:
            noise_scale = 0
            noise = np.random.randn(self.seq_len, 10) * noise_scale
            pred_betas = gt_betas + noise

        beta_weight = np.ones_like(gt_betas)

        pose = self.get_sequence(start_index, end_index, self.db_gt['pose'])
        theta = pose.reshape(self.seq_len, 24, 3)
        angle = np.linalg.norm(theta, axis=2, keepdims=True)
        axis = theta / (angle + 1e-10)
        mask = angle > math.pi
        angle[mask] = angle[mask] - 2 * math.pi
        new_theta = axis * angle
        pose = new_theta.reshape(self.seq_len, 72)

        pred_phi = self.get_sequence(start_index, end_index, self.db_pred['pred_phi'])
        if self.dataset_name != '3doh':
            phi_angle = self.get_sequence(start_index, end_index, self.db_gt['twist_angle']).reshape(self.seq_len, 23)
        else:
            pose_matrix = axis_angle_to_matrix(torch.from_numpy(pose).reshape(self.seq_len, 24, 3))
            rotmat_swing, rotmat_twist, angle_twist = self.smpl_layer.twist_swing_decompose_rot(
                pose_matrix.reshape(self.seq_len, 24, 3, 3).float(), 
                torch.from_numpy(gt_betas).float())

            phi_angle = angle_twist.numpy().reshape(self.seq_len, 23)
            
        phi = np.zeros((self.seq_len, 23, 2))
        phi_weight = np.zeros((self.seq_len, 23, 2))
        phi[:, :, 0] = np.cos(phi_angle)
        phi[:, :, 1] = np.sin(phi_angle)

        phi_weight[:, :, 0] = (phi_angle > -10) * 1.0
        phi_weight[:, :, 1] = (phi_angle > -10) * 1.0

        phi = phi.reshape(-1, 2)
        phi_angle = phi_angle.reshape(-1)
        phi[phi_angle < -10, 0] = 1
        phi[phi_angle < -10, 1] = 0
        phi = phi.reshape(self.seq_len, 23, 2)

        # is_continuous = (start_index not in self.wrong_start_indices) * 1.0
        is_continuous = (start_index not in self.wrong_start_indices) * 1.0
        valid_smpl = torch.ones(self.seq_len, 1).float()
        phi_mask = (phi_angle.reshape(self.seq_len, 23) > -10) * 1.0
        phi_mask = torch.from_numpy((phi_mask.sum(axis=1) > 22) * 1.0).float()
        valid_smpl = valid_smpl * phi_mask.reshape(self.seq_len, 1)
        
        pred_thetas  = self.get_sequence(start_index, end_index, self.db_pred['pred_thetas']).reshape(self.seq_len, 24*9)

        target = {
            'pred_betas': torch.from_numpy(pred_betas).float(),
            'gt_betas': torch.from_numpy(gt_betas).float(),
            'betas_weight': torch.from_numpy(beta_weight).float(),
            'pred_phi': torch.from_numpy(pred_phi).float(),
            'gt_phi': torch.from_numpy(phi).float(),
            'phi_weight': torch.from_numpy(phi_weight).float(),
            'gt_xyz_29': torch.from_numpy(kp_3d_29).float(),
            'pred_xyz_29': torch.from_numpy(kp_3d_29_pred).float(),
            'pred_xyz_24_struct': torch.from_numpy(kp_3d_24_struct_pred).float(),
            'pred_score': torch.from_numpy(pred_score).float(),
            'pred_sigma': torch.from_numpy(pred_sigma).float(),
            'xyz_29_weight': torch.from_numpy(xyz_29_weight).float(),
            'gt_xyz_17': torch.from_numpy(kp_3d_17).float(),
            'xyz_17_weight': torch.from_numpy(xyz_17_weight).float(),
            'is_amass': float(0.0),
            'is_continuous': float(is_continuous),
            'gt_theta': torch.from_numpy(pose).float(),
            'theta_weight': torch.ones(self.seq_len, 72).float(),
            'pred_theta': torch.from_numpy(pred_thetas).float(),
            'is_occlusion': float(self.occlusion),
            'is_3dhp': float(0.0),
            'valid_smpl': valid_smpl.float()
        }

        if self.return_jts2d:
            if self.occlusion:
                bbox = self.get_sequence(start_index, end_index, self.db_pred['amb_center_scale'])
                scale = 1
            else:
                bbox = self.get_sequence(start_index, end_index, self.db_pred['bbox']).reshape(self.seq_len, 4)
                if self.dataset_name == 'h36m' and self.train:
                    scale = 1
                else:
                    scale = 1.25

                bbox[:, 2:] = bbox[:, 2:] * scale

            uv_29, uv_29_normalized, scale_trans = self.get_uv24_cam(
                self.db_gt, start_index, end_index, bbox, scale=1.0
            )

            target['bbox'] = torch.from_numpy(bbox).float()
            target['gt_uv_29'] = torch.from_numpy(uv_29_normalized).float()
            target['gt_scale_trans'] = torch.from_numpy(scale_trans).float()
            target['uv_29_weight'] = torch.ones_like(target['gt_uv_29'])
            target['rand_xyz_29'] = torch.from_numpy(kp_3d_29_pred).float()

            gt_valid_mask = ((target['gt_uv_29'] < 0.5) & (target['gt_uv_29'] > -0.5)) * 1.0
            gt_valid_mask = (gt_valid_mask.sum(dim=-1) > 1.5) * 1.0
            target['gt_valid_mask'] = gt_valid_mask  # which uv is within bbox

            pred_uv_29 = self.get_sequence(start_index, end_index, self.db_pred['pred_uvd']).reshape(self.seq_len, 29, 3)[:, :, :2]
            pred_uvd_29 = self.get_sequence(start_index, end_index, self.db_pred['pred_uvd']).reshape(self.seq_len, 29, 3)
            pred_uvd_29[:, :, 2] = pred_uvd_29[:, :, 2] * 2.2
            target['pred_uv_29'] = torch.from_numpy(pred_uv_29).float()
            target['pred_uvd_29'] = torch.from_numpy(pred_uvd_29).float()

            pred_cam_scale = self.get_sequence(start_index, end_index, self.db_pred['pred_camera']).reshape(self.seq_len, 1)
            pred_cam = np.concatenate((
                pred_cam_scale,
                np.zeros_like(pred_cam_scale),
                np.zeros_like(pred_cam_scale)
            ), axis=1)
            target['pred_cam'] = torch.from_numpy(pred_cam).float()
            target['pred_cam_scale'] = torch.from_numpy(pred_cam_scale).float()

            pred_cam_0center = self.recover_cam_root(
                torch.from_numpy(kp_3d_29_pred).float(),
                torch.from_numpy(pred_uvd_29).float(),
                pred_cam_scale
            )

            target['pred_cam_0center'] = pred_cam_0center.float()

            img_width = self.get_sequence(start_index, end_index, self.db_pred['width']).reshape(self.seq_len, 1)  # img_ann['width'], img_ann['height']
            img_height = self.get_sequence(start_index, end_index, self.db_pred['height']).reshape(self.seq_len, 1)  # img_ann['width'], img_ann['height']
            img_sizes = np.concatenate([img_width, img_height], axis=1)
            img_center = img_sizes * 0.5
            img_center_bbox_coord = (img_center - (bbox[:, :2] - bbox[:, 2:] * 0.5)) / bbox[:, 2:]  # 0-1
            img_center_bbox_coord = (img_center_bbox_coord - 0.5) * 256.0
            target['img_center'] = torch.from_numpy(img_center_bbox_coord).float()
            target['img_sizes'] = torch.from_numpy(img_sizes).float()

            uv_error = np.sqrt(np.sum((uv_29_normalized - pred_uv_29)**2, axis=-1))
            acc_flag = (uv_error < 0.2) * 1.0
            target['is_accurate'] = torch.from_numpy(acc_flag).unsqueeze(-1).float()

            img_paths = []
            for idx in range(start_index, end_index + 1):
                vid_name = self.db_pred['vid_name'][idx]
                vid_name = '_'.join(vid_name.split('_')[:-1])
                if self.dataset_name == 'h36m':
                    img_paths.append(self.db_pred['img_name'][idx])
                elif self.dataset_name == '3doh':
                    img_paths.append(self.db_pred['img_name'][idx])
                else:
                    folder = 'data/pw3d'
                    f = osp.join(folder, 'imageFiles', vid_name)
                    video_file_list = [osp.join(f, x) for x in sorted(os.listdir(f)) if x.endswith('.jpg')]
                    frame_idx = self.db_pred['frame_id'][idx]

                    img_paths.append(video_file_list[frame_idx])

            target['img_path'] = img_paths

        return target

    def _get_item_xyz_val(self, idx):
        start_index, end_index = self.vid_indices[idx]
        kp_3d_29 = self.get_sequence(start_index, end_index, self.db_gt['xyz_29']).reshape(self.seq_len, 29, 3)
        kp_3d_29_pred = self.get_sequence(start_index, end_index, self.db_pred['pred_xyz_29']).reshape(self.seq_len, 29, 3) * 2.2
        kp_3d_24_struct_pred = self.get_sequence(start_index, end_index, self.db_pred['pred_xyz_24_struct']).reshape(self.seq_len, 24, 3) * 2.2

        kp_3d_17_pred = self.get_sequence(start_index, end_index, self.db_pred['pred_xyz_17']).reshape(self.seq_len, 17, 3) * 2.2
        pred_score = self.get_sequence(start_index, end_index, self.db_pred['pred_scores']) * 0.1
        pred_sigma = self.get_sequence(start_index, end_index, self.db_pred['pred_sigma'])

        if 'img_name' in self.db_gt:
            img_name = self.get_sequence(start_index, end_index, self.db_gt['img_name'])
        else:
            img_name = self.get_sequence(start_index, end_index, self.db_gt['img_path'])
        
        img_name = [str(item) for item in img_name]

        kp_3d_29 = kp_3d_29 - kp_3d_29[:, [1, 2], :].mean(axis=1, keepdims=True)
        kp_3d_29_pred = kp_3d_29_pred - kp_3d_29_pred[:, [1, 2], :].mean(axis=1, keepdims=True)
        kp_3d_24_struct_pred = kp_3d_24_struct_pred - kp_3d_24_struct_pred[:, [1, 2], :].mean(axis=1, keepdims=True)
        xyz_29_weight = np.ones_like(kp_3d_29)

        kp_3d_17 = self.get_sequence(start_index, end_index, self.db_gt['xyz_17']).reshape(self.seq_len, 17, 3)
        kp_3d_17 = kp_3d_17 - kp_3d_17[:, [1, 4], :].mean(axis=1, keepdims=True)

        pred_phi = self.get_sequence(start_index, end_index, self.db_pred['pred_phi'])

        phi_angle = self.get_sequence(start_index, end_index, self.db_gt['twist_angle']).reshape(self.seq_len, 23)
        phi = np.zeros((self.seq_len, 23, 2))
        phi_weight = np.zeros((self.seq_len, 23, 2))
        phi[:, :, 0] = np.cos(phi_angle)
        phi[:, :, 1] = np.sin(phi_angle)

        phi_weight[:, :, 0] = (phi_angle > -10) * 1.0
        phi_weight[:, :, 1] = (phi_angle > -10) * 1.0

        phi = phi.reshape(-1, 2)
        phi_angle = phi_angle.reshape(-1)
        phi[phi_angle < -10, :] = pred_phi.reshape(-1, 2)[phi_angle < -10, :]
        phi = phi.reshape(self.seq_len, 23, 2)

        pred_theta = self.get_sequence(start_index, end_index, self.db_pred['pred_thetas'])
        pred_betas = self.get_sequence(start_index, end_index, self.db_pred['pred_betas'])[:, :10]
        gt_betas = self.get_sequence(start_index, end_index, self.db_gt['shape'])

        is_continuous = (start_index not in self.wrong_start_indices) * 1.0

        pose = self.get_sequence(start_index, end_index, self.db_gt['pose'])

        theta = pose.reshape(self.seq_len, 24, 3)
        angle = np.linalg.norm(theta, axis=2, keepdims=True)
        axis = theta / (angle + 1e-6)
        mask = angle > math.pi
        angle[mask] = angle[mask] - 2 * math.pi
        new_theta = axis * angle
        pose = new_theta.reshape(self.seq_len, 72)

        pred_thetas  = self.get_sequence(start_index, end_index, self.db_pred['pred_thetas']).reshape(self.seq_len, 24*9)

        target = {
            'img_name': img_name,
            'gt_xyz_29': torch.from_numpy(kp_3d_29).float(),
            'pred_xyz_29': torch.from_numpy(kp_3d_29_pred).float(),
            'pred_xyz_24_struct': torch.from_numpy(kp_3d_24_struct_pred).float(),
            'pred_xyz_17': torch.from_numpy(kp_3d_17_pred).float(),
            'pred_score': torch.from_numpy(pred_score).float(),
            'pred_sigma': torch.from_numpy(pred_sigma).float(),
            'gt_xyz_17': torch.from_numpy(kp_3d_17).float(),
            'xyz_29_weight': torch.from_numpy(xyz_29_weight).float(),
            'pred_phi': torch.from_numpy(pred_phi).float(),
            'gt_phi': torch.from_numpy(phi).float(),
            'pred_theta': torch.from_numpy(pred_theta).float(),
            'pred_betas': torch.from_numpy(pred_betas).float(),
            'gt_betas': torch.from_numpy(gt_betas).float(),
            'is_continuous': float(is_continuous),
            'gt_theta': torch.from_numpy(pose).float(),
            'theta_weight': torch.zeros(self.seq_len, 72).float(),
            'pred_theta': torch.from_numpy(pred_thetas).float(),
        }

        if self.return_jts2d:
            pred_cam_scale = self.get_sequence(start_index, end_index, self.db_pred['pred_camera']).reshape(self.seq_len, 1)
            pred_cam = np.concatenate((
                pred_cam_scale,
                np.zeros_like(pred_cam_scale),
                np.zeros_like(pred_cam_scale)
            ), axis=1)
            target['pred_cam'] = torch.from_numpy(pred_cam).float()
            target['pred_cam_scale'] = torch.from_numpy(pred_cam_scale).float()

            if self.occlusion:
                bbox = self.get_sequence(start_index, end_index, self.db_pred['amb_center_scale'])
                syn_size = self.get_sequence(start_index, end_index, self.db_pred['amb_synth_size'])
                target['amb_synth_size'] = torch.from_numpy(syn_size).type(torch.int32)

                img_paths = self.get_sequence(start_index, end_index, self.db_pred['img_paths'], to_np=False)
                target['img_path'] = list(img_paths)
                scale = 1
            else:
                bbox = self.get_sequence(start_index, end_index, self.db_pred['bbox']).reshape(self.seq_len, 4)
                if self.dataset_name == 'h36m' and self.train:
                    scale = 1
                elif self.dataset_name == '3doh':
                    scale = 1.3
                else:
                    scale = 1.25

                bbox[:, 2:] = bbox[:, 2:] * scale

                img_names = self.get_sequence(start_index, end_index, self.db_pred['img_name'], to_np=False)
                img_paths = [translate_img_path(img_name, self.dataset_name) for img_name in img_names]

                target['img_path'] = img_paths

            target['bbox'] = torch.from_numpy(bbox).float()

            pred_uv_29 = self.get_sequence(start_index, end_index, self.db_pred['pred_uvd']).reshape(self.seq_len, 29, 3)[:, :, :2]
            pred_uvd_29 = self.get_sequence(start_index, end_index, self.db_pred['pred_uvd']).reshape(self.seq_len, 29, 3)
            pred_uvd_29[:, :, 2] = pred_uvd_29[:, :, 2] * 2.2
            target['pred_uv_29'] = torch.from_numpy(pred_uv_29).float()
            target['pred_uvd_29'] = torch.from_numpy(pred_uvd_29).float()
            target['rand_xyz_29'] = torch.from_numpy(kp_3d_29_pred).float()

            pred_cam_0center = self.recover_cam_root(
                torch.from_numpy(kp_3d_29_pred).float(),
                torch.from_numpy(pred_uvd_29).float(),
                pred_cam_scale
            )

            target['pred_cam_0center'] = pred_cam_0center.float()

            if self.get_gt_uv:
                uv_29, uv_29_normalized, scale_trans = self.get_uv24_cam(
                    self.db_gt, start_index, end_index, bbox, scale=1.0
                )
                target['gt_scale_trans'] = torch.from_numpy(scale_trans).float()
                target['gt_uv_29'] = torch.from_numpy(uv_29_normalized).float()
                target['uv_29_weight'] = torch.ones_like(target['gt_uv_29'])

            img_width = self.get_sequence(start_index, end_index, self.db_pred['width']).reshape(self.seq_len, 1)  # img_ann['width'], img_ann['height']
            img_height = self.get_sequence(start_index, end_index, self.db_pred['height']).reshape(self.seq_len, 1)  # img_ann['width'], img_ann['height']
            img_sizes = np.concatenate([img_width, img_height], axis=1)
            img_center = img_sizes * 0.5
            img_center_bbox_coord = (img_center - (bbox[:, :2] - bbox[:, 2:] * 0.5)) / bbox[:, 2:]  # 0-1
            img_center_bbox_coord = (img_center_bbox_coord - 0.5) * 256.0
            target['img_center'] = torch.from_numpy(img_center_bbox_coord).float()
            target['img_sizes'] = torch.from_numpy(img_sizes).float()

            if 'annot_idx' in self.db_pred:
                annot_idx = self.get_sequence(start_index, end_index, self.db_pred['annot_idx'], to_np=False)
                target['annot_idx'] = torch.from_numpy(annot_idx)

        return target

    def load_db(self, ann_file, only_read=True):
        if osp.isfile(ann_file) and only_read:
            db = joblib.load(ann_file, 'r')
        elif osp.isfile(ann_file):
            db = joblib.load(ann_file)
        else:
            raise ValueError(f'{ann_file} do not exists.')

        print(f'Loaded {ann_file}.')
        return db

    def get_sequence(self, start_index, end_index, data, to_np=True):
        if start_index != end_index:
            v = data[start_index:end_index + 1]
        else:
            v = data[start_index:start_index + 1]
            # v = data[start_index:start_index + 1].repeat(self.seqlen, axis=0)

        if to_np:
            return np.array(v)
        else:
            return v

    def evaluate_xyz_17(self, preds, gts):
        # print('Evaluation start...')

        assert len(gts) == len(preds)
        sample_num = len(gts)

        # pred_save = []
        error = np.zeros((sample_num, len(self.EVAL_JOINTS)))  # joint error
        error_align = np.zeros((sample_num, len(self.EVAL_JOINTS)))  # joint error
        error_x = np.zeros((sample_num, len(self.EVAL_JOINTS)))  # joint error
        error_y = np.zeros((sample_num, len(self.EVAL_JOINTS)))  # joint error
        error_z = np.zeros((sample_num, len(self.EVAL_JOINTS)))  # joint error
        # error for each sequence
        # error_action = [[] for _ in range(len(self.action_name))]
        for n in range(sample_num):
            gt_3d_kpt = gts[n].copy()
            pred_3d_kpt = preds[n].copy()

            gt_3d_kpt = gt_3d_kpt
            pred_3d_kpt = pred_3d_kpt

            # root joint alignment
            pred_3d_kpt = pred_3d_kpt - pred_3d_kpt[self.root_idx_17]
            gt_3d_kpt = gt_3d_kpt - gt_3d_kpt[self.root_idx_17]

            pred_3d_kpt_align = reconstruction_error(pred_3d_kpt.copy(), gt_3d_kpt.copy())

            # select eval 14 joints
            pred_3d_kpt = np.take(pred_3d_kpt, self.EVAL_JOINTS, axis=0) * 1000.0
            gt_3d_kpt = np.take(gt_3d_kpt, self.EVAL_JOINTS, axis=0) * 1000.0
            pred_3d_kpt_align = np.take(pred_3d_kpt_align, self.EVAL_JOINTS, axis=0) * 1000.0

            # error calculate
            error[n] = np.sqrt(np.sum((pred_3d_kpt - gt_3d_kpt)**2, 1))
            error_align[n] = np.sqrt(np.sum((pred_3d_kpt_align - gt_3d_kpt)**2, 1))
            error_x[n] = np.abs(pred_3d_kpt[:, 0] - gt_3d_kpt[:, 0])
            error_y[n] = np.abs(pred_3d_kpt[:, 1] - gt_3d_kpt[:, 1])
            error_z[n] = np.abs(pred_3d_kpt[:, 2] - gt_3d_kpt[:, 2])

        # total error
        tot_err = np.mean(error)
        tot_err_align = np.mean(error_align)
        tot_err_x = np.mean(error_x)
        tot_err_y = np.mean(error_y)
        tot_err_z = np.mean(error_z)

        eval_summary = f'PRED XYZ_17 tot: {tot_err:2f}, tot_pa: {tot_err_align:2f}, x: {tot_err_x:2f}, y: {tot_err_y:.2f}, z: {tot_err_z:2f}'

        print(eval_summary)

        return tot_err_align, tot_err

    def evaluate_xyz_29(self, preds, gts):
        # print('Evaluation start...')

        assert len(gts) == len(preds), f'{len(gts)}, {len(preds)}'
        sample_num = len(gts)

        error = np.zeros((sample_num, 29))  # joint error
        error_align = np.zeros((sample_num, 29))  # joint error
        error_x = np.zeros((sample_num, 29))  # joint error
        error_y = np.zeros((sample_num, 29))  # joint error
        error_z = np.zeros((sample_num, 29))  # joint error

        for n in range(sample_num):
            gt_3d_kpt = gts[n].copy()

            # restore coordinates to original space
            pred_3d_kpt = preds[n].copy() * 1000.0
            gt_3d_kpt = gt_3d_kpt * 1000.0

            # print(pred_3d_kpt, gt_3d_kpt)

            # root joint alignment
            pred_3d_kpt = pred_3d_kpt - pred_3d_kpt[self.root_idx_smpl]
            gt_3d_kpt = gt_3d_kpt - gt_3d_kpt[self.root_idx_smpl]

            # rigid alignment for PA MPJPE
            pred_3d_kpt_align = reconstruction_error(pred_3d_kpt.copy(), gt_3d_kpt.copy())

            # error calculate
            error[n] = np.sqrt(np.sum((pred_3d_kpt - gt_3d_kpt)**2, 1))
            error_align[n] = np.sqrt(np.sum((pred_3d_kpt_align - gt_3d_kpt)**2, 1))
            error_x[n] = np.abs(pred_3d_kpt[:, 0] - gt_3d_kpt[:, 0])
            error_y[n] = np.abs(pred_3d_kpt[:, 1] - gt_3d_kpt[:, 1])
            error_z[n] = np.abs(pred_3d_kpt[:, 2] - gt_3d_kpt[:, 2])

        # total error
        tot_err = np.mean(error)
        tot_err_align = np.mean(error_align)
        tot_err_x = np.mean(error_x)
        tot_err_y = np.mean(error_y)
        tot_err_z = np.mean(error_z)

        eval_summary = f'XYZ_29 tot: {tot_err:2f}, tot_pa: {tot_err_align:2f}, x: {tot_err_x:2f}, y: {tot_err_y:.2f}, z: {tot_err_z:2f}'
        print(eval_summary)

        return tot_err_align, tot_err

    def get_uv24_cam(self, db, start_index, end_index, bbox, scale=1):
        if self.dataset_name == 'pw3d':
            uv_24 = self.get_sequence(start_index, end_index, db['uv_24'])

            joint_cam_origin = self.get_sequence(start_index, end_index, db['kpt_cam_origin'])
            joint_cam_origin = joint_cam_origin.reshape(self.seq_len, -1, 3)
            joint_cam_xyz29 = self.get_sequence(start_index, end_index, db['xyz_29'])
            joint_cam_xyz29 = joint_cam_xyz29.reshape(self.seq_len, 29, 3)

            joint_cam_origin = joint_cam_origin * 1000
            joint_cam_xyz29 = joint_cam_xyz29 * 1000
            joint_cam_xyz29 = (joint_cam_xyz29 - joint_cam_xyz29[:, [0], :]) + joint_cam_origin[:, [0], :]

            cam_params = self.get_sequence(start_index, end_index, db['cam_param'], to_np=False)
            fs = np.array([cam_para['focal'].copy() for cam_para in cam_params])
            cs = np.array([cam_para['princpt'].copy() for cam_para in cam_params])

            gt_uv_29 = cam2pixel_temporal(joint_cam_xyz29, fs, cs)[:, :, :2]
            if self.train:
                assert (np.absolute(gt_uv_29[:, :24, :] - uv_24) < 1e-1).all(), f'{gt_uv_29[0]}, {uv_24[0]}, {np.absolute(gt_uv_29[:, :24, :] - uv_24).max()}'
            uv_24 = gt_uv_29[:, :24, :2]
        
        elif self.dataset_name == 'h36m':
            cam_params = self.get_sequence(start_index, end_index, db['cam_param'], to_np=False)
            f = np.array([cam_para['f'] for cam_para in cam_params])
            c = np.array([cam_para['c'] for cam_para in cam_params])

            joint_cam_origin = self.get_sequence(start_index, end_index, db['kpt_cam_origin']).reshape(self.seq_len, 17, 3)
            joint_cam_xyz29 = self.get_sequence(start_index, end_index, self.db_gt['xyz_29']).reshape(self.seq_len, 29, 3)
            joint_cam_xyz29 = (joint_cam_xyz29 - joint_cam_xyz29[:, [0]]) * 1000.0 + joint_cam_origin[:, [0]]

            joint_img_29 = cam2pixel_temporal(joint_cam_xyz29, f, c)
            gt_uv_29 = joint_img_29[:, :, :2]
            uv_24 = joint_img_29[:, :24, :2]
        
        else: raise NotImplementedError
    
        # bbox = self.get_sequence(start_index, end_index, db['bbox']) # center-scale
        xyz_29 = self.get_sequence(start_index, end_index, db['xyz_29']).reshape(self.seq_len, 29, 3)
        xyz_29 = xyz_29 - xyz_29[:, [1, 2], :].mean(axis=1, keepdims=True)
        xyz_24 = xyz_29[:, :24]

        xyz_24 = xyz_24 - xyz_24[:, [1, 2], :].mean(axis=1, keepdims=True)

        scale_trans = np.zeros((self.seq_len, 4))

        bbox[:, 2:] = bbox[:, 2:] * scale
        normed_uv = normalize_uv_temporal(uv_24, bbox, scale=1.0)
        normed_uv_29 = normalize_uv_temporal(gt_uv_29, bbox, scale=1.0)

        img_width = self.get_sequence(start_index, end_index, db['width']).reshape(self.seq_len, 1)  # img_ann['width'], img_ann['height']
        img_height = self.get_sequence(start_index, end_index, db['height']).reshape(self.seq_len, 1)  # img_ann['width'], img_ann['height']
        img_sizes = np.concatenate([img_width, img_height], axis=1)
        img_center = img_sizes * 0.5
        img_center_bbox_coord = (img_center - (bbox[:, :2] - bbox[:, 2:] * 0.5)) / bbox[:, 2:]  # 0-1
        img_center_bbox_coord = (img_center_bbox_coord - 0.5) * 256.0

        for idx in range(self.seq_len):
            new_kp_2d_i = normed_uv[idx]

            cam_scale_trans, cam_valid, diff = calc_cam_scale_trans(
                                        xyz_24[idx].reshape(-1, 3).copy(),
                                        new_kp_2d_i.reshape(-1, 2).copy(),
                                        np.ones_like(xyz_24[idx]))

            scale_trans[idx, :3] = cam_scale_trans
            scale_trans[idx, 3] = cam_valid

        return gt_uv_29, normed_uv_29, scale_trans

    def recover_cam_root(self, pred_xyz, pred_uv, init_scale):  
        # xyz, uv: seq_len x 24/29 x3/2

        # (xy + d_xy) = uv * (z * 256 / 1000 + 1/s)
        batch_size = self.seq_len

        Ax = torch.zeros((batch_size, 24, 3), device=pred_xyz.device, dtype=torch.float32)
        Ay = torch.zeros((batch_size, 24, 3), device=pred_xyz.device, dtype=torch.float32)

        Ax[:, :, 0], Ax[:, :, 2] = 1, -pred_uv[:, :24, 0]
        Ay[:, :, 1], Ay[:, :, 2] = 1, -pred_uv[:, :24, 1]

        bx = pred_uv[:, :24, 0] * pred_xyz[:, :24, 2] * 256 / 1000 - pred_xyz[:, :24, 0]
        by = pred_uv[:, :24, 1] * pred_xyz[:, :24, 2] * 256 / 1000 - pred_xyz[:, :24, 1]

        A = torch.cat([Ax, Ay], dim=1) # b x 48 x 3
        b = torch.cat([bx, by], dim=1) # b x 48

        A_T = A.transpose(1, 2) # b x 3 x 48
        A_T_A = torch.bmm(A_T, A)

        res = torch.inverse(A_T_A).bmm(torch.bmm(A_T, b.unsqueeze(-1)))
        res = res[:, :, 0]

        scale = 1 / res[:, 2]

        transl = res.clone()
        transl[:, 2] = 1000.0 / (256.0 * scale + 1e-9)

        pred_cam = res.clone()
        pred_cam[:, 0] = scale
        pred_cam[:, 1:] = res[:, :2]

        return pred_cam