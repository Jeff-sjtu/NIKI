import math
import os
import pickle as pk
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data

from niki.models.layers.smpl.SMPL import SMPL_layer
from niki.utils.pose_utils import Error_Score_Evaluator, back_projection_batch


def batch_rodrigues(axisang):
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle + 1e-8)

    rx, ry, rz = torch.split(axisang_normalized, 1, dim=1)
    zeros = torch.zeros_like(rx)

    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((axisang.shape[0], 3, 3))
    ident = torch.eye(3, dtype=axisang.dtype, device=axisang.device).unsqueeze(dim=0)

    angle = angle[:, :, None]
    rot_mat = ident + torch.sin(angle) * K + (1 - torch.cos(angle)) * torch.bmm(K, K)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat


class amass_dataset_temporal(data.Dataset):
    dataset_names = [
        'ACCAD', 'BMLhandball', 'BMLmovi', 'BioMotionLab_NTroje', 'CMU',
        'DFaust_67', 'DanceDB', 'EKUT', 'Eyes_Japan_Dataset', 'HUMAN4D', 'HumanEva', 'KIT', 'MPI_mosh',
        'MPI_HDM05', 'MPI_Limits', 'SFU', 'SSM_synced', 'TCD_handMocap', 'TotalCapture', 'Transitions_mocap']

    parents = torch.tensor([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
                            16, 17, 18, 19, 20, 21, 15, 22, 23, 10, 11], dtype=torch.long)

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

    def __init__(self, seq_len=16, amass_path='data/amass', usage='phi', occlusion=True, use_pretrained_feat=True,
                 use_flip=False, wrong_flip_aug=False, use_joint_part_seg=False, img_feat_size=2048):
        # len_file = len()
        # self._lazy_load_data()
        self.pkl_dir = 'data/amass/processed_pose_25fps'
        self.files = [os.path.join(self.pkl_dir, f) for f in os.listdir(self.pkl_dir) if f.split('.')[-1] == 'pkl']
        self.files.sort()
        self.seq_len = seq_len

        self.usage = usage
        self.occlusion = occlusion
        self.use_flip = use_flip
        self.wrong_flip_aug = wrong_flip_aug
        self.use_joint_part_seg = use_joint_part_seg
        print('wrong_flip_aug', wrong_flip_aug)

        self.kpt29_error = np.ones(29) * 3
        self.kpt29_error[[1, 2, 3]] = 15
        self.kpt29_error[[4, 5, 6, 9]] = 40
        self.kpt29_error[[7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]] = 65
        self.kpt29_error[[20, 21, 22, 23, 24, 27, 28]] = 80
        self.kpt29_error[[25, 26]] = 120
        self.kpt29_error = self.kpt29_error.reshape(-1, 1)

        self.xyz_ratio = np.array([0.3, 0.4, 0.6])

        h36m_jregressor = np.load('./model_files/J_regressor_h36m.npy')
        self.smpl_layer = SMPL_layer(
            './model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
            h36m_jregressor=h36m_jregressor,
            dtype=torch.float32
        )

        if self.occlusion:
            self.valid_score_mean = np.array([0.107235, 0.47090596])
            self.valid_score_cov = np.array(
                [[0.02559005, -0.01198124],
                 [-0.01198124, 0.02639371]]
            )

            self.invalid_score_mean = np.array([0.36691626, 0.3448269])
            self.invalid_score_cov = np.array(
                [[0.07646557, -0.04886626],
                 [-0.04886626, 0.09448104]]
            )

            self.cam_mean = np.array([0.85, 0.05, 0.12])
            self.cam_std = np.array([0.36, 0.12, 0.25])
            self.scale_min = 0.2
        else:
            self.cam_mean = np.array([0.63473389 * 0.9, -0.00110506, 0.01689055])
            self.cam_std = np.array([0.12386466, 0.0778528, 0.09586132])
            self.cam_std[1:] = self.cam_std[1:] * 3
            self.scale_min = 0.3

            self.valid_score_mean = np.array([0.09131249, 0.46515207])
            self.valid_score_cov = np.array(
                [[0.0072631, -0.00508146],
                 [-0.00508146, 0.01609746]]
            )

            self.invalid_score_mean = self.valid_score_mean
            self.invalid_score_cov = self.valid_score_cov

        self.es_eva_valid = Error_Score_Evaluator()
        self.es_eva_invalid = Error_Score_Evaluator()

        self.es_eva_valid.update_params(self.valid_score_mean, self.valid_score_cov)
        self.es_eva_invalid.update_params(self.invalid_score_mean, self.invalid_score_cov)

        self.use_pretrained_feat = use_pretrained_feat
        self.img_feat_size = img_feat_size

    def __getitem__(self, idx):
        return self._get_item_xyz(idx)

    def __len__(self):
        return len(self.files)

    def _get_item_xyz(self, idx):
        '''
        'beta': item['beta'][0].numpy(),
        'pose': item['pose'][0].numpy(),
        'twist_angle': item['twist_angle'][0].numpy(),
        'gt_xyz_17': item['gt_xyz_17'][0].numpy(),
        'gt_xyz_29': item['gt_xyz_29'][0].numpy(),
        'gender': item['gender'][0]
        '''
        with open(self.files[idx], 'rb') as f:
            file_db = pk.load(f)

        file_db_len = len(file_db)

        if file_db_len < self.seq_len:
            targets = [file_db[0] for _ in range(self.seq_len)]
            start_index, end_idx = 0, self.seq_len
        else:
            start_index = random.randint(0, file_db_len - self.seq_len)
            end_idx = start_index + self.seq_len
            targets = file_db[start_index:end_idx]

        # gt_xyz_29 = [target['gt_xyz_29'].reshape(1, 29, 3) for target in targets]
        # gt_xyz_17 = [target['gt_xyz_17'].reshape(1, 17, 3) for target in targets]
        # pose = [target['pose'].reshape(1, 24, 3) for target in targets]
        # phi_angle = [target['twist_angle'].reshape(1, 23) for target in targets]
        # beta = [target['beta'].reshape(1, 10) for target in targets]

        # gt_xyz_29 = np.concatenate(gt_xyz_29, axis=0)
        # gt_xyz_17 = np.concatenate(gt_xyz_17, axis=0)
        # pose = np.concatenate(pose, axis=0)
        # phi_angle = np.concatenate(phi_angle, axis=0)
        # beta = np.concatenate(beta, axis=0)
        # beta_weight = np.ones_like(beta)

        # gt_xyz_29 = gt_xyz_29 - gt_xyz_29[:, [1, 2]].mean(axis=1, keepdims=True)
        # gt_xyz_17 = gt_xyz_17 - gt_xyz_17[:, [1, 4]].mean(axis=1, keepdims=True)
        pose = [target['pose'].reshape(1, 24, 3) for target in targets]
        phi_angle = [target['twist_angle'].reshape(1, 23) for target in targets]

        new_pose = []
        for single_pose in pose:
            new_pose.append(rectify_pose(single_pose.reshape(72)).reshape(1, 24, 3).copy())

        pose = np.concatenate(new_pose, axis=0)
        # pose[:, 22:] = 0

        beta = [target['beta'].reshape(1, 10) for target in targets]
        beta = np.concatenate(beta, axis=0)

        pose_torch = torch.from_numpy(pose).float()
        beta_torch = torch.from_numpy(beta).float()
        with torch.no_grad():
            smpl_out = self.smpl_layer(
                pose_axis_angle=pose_torch,
                betas=beta_torch,
                global_orient=None,
                return_29_jts=True
            )

            gt_xyz_29 = smpl_out.joints.numpy()
            gt_xyz_17 = smpl_out.joints_from_verts.numpy()

        # gt_xyz_29 = np.concatenate(gt_xyz_29, axis=0)
        # gt_xyz_17 = np.concatenate(gt_xyz_17, axis=0)
        phi_angle = np.concatenate(phi_angle, axis=0)
        # beta = np.concatenate(beta, axis=0)
        beta_weight = np.ones_like(beta)

        gt_xyz_29 = gt_xyz_29 - gt_xyz_29[:, [1, 2]].mean(axis=1, keepdims=True)
        gt_xyz_17 = gt_xyz_17 - gt_xyz_17[:, [1, 4]].mean(axis=1, keepdims=True)

        xyz_29_weight = np.ones_like(gt_xyz_29)
        xyz_17_weight = np.ones_like(gt_xyz_17)
        xyz_29_weight[:, [25, 26]] = 0

        elem_rand_scale = np.random.randn(self.seq_len, gt_xyz_29.shape[1], 3)
        elem_rand_scale = elem_rand_scale * (0.5 + np.random.rand(self.seq_len, gt_xyz_29.shape[1], 3))
        elem_rand_scale = elem_rand_scale * self.kpt29_error * 1e-3
        elem_rand_scale = elem_rand_scale * (self.xyz_ratio + np.random.rand(3) * 0.1 - 0.05)
        rand_xyz_29 = gt_xyz_29 + elem_rand_scale

        rand_scale = np.random.rand(self.seq_len).reshape(self.seq_len, 1, 1)
        rand_xyz_29[:, :, :2] = rand_xyz_29[:, :, :2] * (1 + 0.1 * rand_scale - 0.05)
        rand_xyz_29 = rand_xyz_29 - rand_xyz_29[:, [1, 2]].mean(axis=1, keepdims=True)

        rand_phi_angle = phi_angle + np.random.randn(*phi_angle.shape) * 10 * 180 / math.pi

        gt_phi = np.zeros((self.seq_len, 23, 2))
        phi_weight = np.zeros((self.seq_len, 23, 2))
        gt_phi[:, :, 0] = np.cos(phi_angle)
        gt_phi[:, :, 1] = np.sin(phi_angle)

        phi_weight[:, :, 0] = (phi_angle > -10) * 1.0
        phi_weight[:, :, 1] = (phi_angle > -10) * 1.0

        gt_phi = gt_phi.reshape(-1, 2)
        phi_angle = phi_angle.reshape(-1)
        gt_phi[phi_angle < -10, 0] = 1
        gt_phi[phi_angle < -10, 1] = 0
        gt_phi = gt_phi.reshape(self.seq_len, -1, 2)

        if np.isnan(gt_phi).any():
            gt_phi[:, :, 0] = 1
            gt_phi[:, :, 1] = 0
            phi_weight[:] = 0

        pred_phi = np.zeros((self.seq_len, 23, 2))
        pred_phi[:, :, 0] = np.cos(rand_phi_angle)
        pred_phi[:, :, 1] = np.sin(rand_phi_angle)
        pred_phi = pred_phi.reshape(-1, 2)
        rand_phi_angle = rand_phi_angle.reshape(-1)
        pred_phi[rand_phi_angle < -10, 0] = 1
        pred_phi[rand_phi_angle < -10, 1] = 0
        pred_phi = pred_phi.reshape(self.seq_len, -1, 2)

        if np.isnan(pred_phi).any():
            pred_phi[:, :, 0] = 1
            pred_phi[:, :, 1] = 0

        theta = pose.reshape(self.seq_len, 24, 3)
        theta_weight = np.zeros_like(theta)
        theta_weight[:, :20] = 1

        target = {
            'betas': torch.from_numpy(beta).float(),
            'betas_weight': torch.from_numpy(beta_weight).float(),
            'gt_phi': torch.from_numpy(gt_phi).float(),
            'pred_phi': torch.from_numpy(pred_phi).float(),
            'phi_weight': torch.from_numpy(phi_weight).float(),
            'gt_xyz_29': torch.from_numpy(gt_xyz_29).float(),
            # 'pred_xyz_29': torch.from_numpy(rand_xyz_29).float(),
            # 'pred_score': torch.from_numpy(pred_score).float(),
            'xyz_29_weight': torch.from_numpy(xyz_29_weight).float(),
            'gt_xyz_17': torch.from_numpy(gt_xyz_17).float(),
            'xyz_17_weight': torch.from_numpy(xyz_17_weight).float(),
            'is_continuous': float(1.0),
            'theta': torch.from_numpy(theta.reshape(self.seq_len, 72)).float(),
            'theta_weight': torch.from_numpy(theta_weight.reshape(self.seq_len, 72)).float(),
            'is_occlusion': float(self.occlusion),
            'is_3dhp': float(0.0),
            'img_path': [f'{self.files[idx]}_{i}.png' for i in range(start_index, end_idx)]
        }

        self.return_jts2d = True
        if self.return_jts2d:
            # generate camera param, uv29, scores by random
            rand_scale_trans = np.ones((self.seq_len, 4))
            pred_rand_scale_trans = np.ones((self.seq_len, 4))
            base_scale_trans = np.random.randn(3) * self.cam_std + self.cam_mean
            rand_scale_trans[:, :3] = base_scale_trans + np.random.randn(self.seq_len, 3) * self.cam_std * 0.2
            rand_scale_trans[:, 0] = np.maximum(rand_scale_trans[:, 0], self.scale_min)

            pred_rand_scale_trans[:, :3] = rand_scale_trans[:, :3] + np.random.randn(self.seq_len, 3) * self.cam_std * 0.2
            pred_rand_scale_trans[:, 0] = np.maximum(pred_rand_scale_trans[:, 0], self.scale_min)

            pred_uv_29 = self.projection(
                rand_xyz_29.copy(),
                rand_scale_trans.copy()
            )  # roughly accurate
            gt_uv_29 = self.projection(gt_xyz_29.copy(), rand_scale_trans[:, :3].copy())

            gt_valid_mask_29 = ((gt_uv_29 < 0.5) & (gt_uv_29 > -0.5)) * 1.0
            gt_valid_mask_29 = (gt_valid_mask_29.sum(axis=-1) > 1.5) * 1.0
            gt_valid_mask = gt_valid_mask_29

            pred_valid_mask_29 = ((pred_uv_29 < 0.5) & (pred_uv_29 > -0.5)) * 1.0
            pred_valid_mask_29 = (pred_valid_mask_29.sum(axis=-1) > 1.5) * 1.0
            pred_valid_mask_29 = pred_valid_mask_29.reshape(-1)
            invalid_num = int((1 - pred_valid_mask_29).sum())

            if self.occlusion:
                pred_uv_29 = pred_uv_29.reshape(-1)
                pred_uv_29_new = pred_uv_29.copy()
                pred_uv_29_new[pred_uv_29 < -0.5] = -0.5 + np.random.random(int((pred_uv_29 < -0.5).sum())) * 0.3
                pred_uv_29_new[pred_uv_29 > 0.5] = 0.5 - np.random.random(int((pred_uv_29 > 0.5).sum())) * 0.3

                pred_uv_29 = pred_uv_29_new.reshape(self.seq_len, 29, 2)
                assert (pred_uv_29 >= -0.5).all() and (pred_uv_29 <= 0.5).all(), pred_uv_29

                rand_xyz_29_from_uv = back_projection_batch(
                    pred_uv_29.copy(),
                    rand_xyz_29.copy(),
                    pred_rand_scale_trans,
                    focal_length=1000.0).reshape(-1, 3)  # not accurate at all

                rand_xyz_29_tmp = rand_xyz_29.reshape(-1, 3).copy()
                rand_xyz_29_tmp[pred_valid_mask_29 < 0.5, :2] = rand_xyz_29_from_uv[pred_valid_mask_29 < 0.5, :2]
                rand_xyz_29_tmp[pred_valid_mask_29 < 0.5, 2] += np.random.randn(invalid_num) * 0.1

                rand_xyz_29_tmp = rand_xyz_29_tmp.reshape(self.seq_len, 29, 3)
            else:
                rand_xyz_29_tmp = rand_xyz_29.copy()

            xyz_diff = rand_xyz_29_tmp - gt_xyz_29

            score_valid = self.es_eva_valid.sample(xyz_diff, pred_valid_mask_29 > 0.5)
            score_invalid = self.es_eva_invalid.sample(xyz_diff, pred_valid_mask_29 < 0.5)

            pred_valid_mask_29 = pred_valid_mask_29.reshape(self.seq_len, -1)
            pred_score = score_valid * pred_valid_mask_29 + (1 - pred_valid_mask_29) * score_invalid
            pred_score = pred_score.reshape(self.seq_len, 29, 1)

            if self.wrong_flip_aug:
                # flip upper body / lower body
                rand_xyz_29, rand_xyz_29_tmp, pred_uv_29, pred_score = self.wrong_flip_augment(rand_xyz_29, rand_xyz_29_tmp, pred_uv_29, pred_score)

            rand_xyz_29_tmp = rand_xyz_29_tmp - rand_xyz_29_tmp[:, [1, 2]].mean(axis=1, keepdims=True)

            with torch.no_grad():
                pred_phi = torch.from_numpy(pred_phi).float()
                pred_shape = torch.from_numpy(beta).float()
                pred_xyz_jts_29 = rand_xyz_29_tmp - rand_xyz_29_tmp[:, [0]]

                smpl_out = self.smpl_layer.hybrik(
                    pose_skeleton=torch.from_numpy(pred_xyz_jts_29).float(),
                    betas=pred_shape,
                    phis=pred_phi,
                    global_orient=None,
                    return_verts=False
                )
                pred_xyz_jts_24_struct = smpl_out.joints.float().numpy()
                pred_xyz_jts_24_struct = pred_xyz_jts_24_struct - pred_xyz_jts_24_struct[:, [1, 2]].mean(axis=1, keepdims=True)

            target['rand_xyz_29'] = torch.from_numpy(rand_xyz_29).float()  # predict aftfer bert
            target['pred_xyz_29'] = torch.from_numpy(rand_xyz_29_tmp).float()
            target['pred_xyz_24_struct'] = torch.from_numpy(pred_xyz_jts_24_struct).float()
            target['pred_score'] = torch.from_numpy(pred_score).float()
            target['pred_sigma'] = torch.from_numpy(1 - pred_score * 0.1).float()
            target['gt_uv_29'] = torch.from_numpy(gt_uv_29).float()
            target['gt_scale_trans'] = torch.from_numpy(rand_scale_trans).float()
            target['uv_29_weight'] = torch.ones_like(target['gt_uv_29']).float()
            target['gt_valid_mask'] = torch.from_numpy(gt_valid_mask).float()
            target['pred_uv_29'] = torch.from_numpy(pred_uv_29).float()
            target['is_amass'] = float(1.0)
            # target['pred_cam'] = torch.from_numpy(pred_rand_scale_trans[:, :3]).float()
            if self.img_feat_size != 2048:
                target['pred_cam'] = torch.from_numpy(pred_rand_scale_trans[:, :3]).float()
            else:
                target['pred_cam'] = torch.from_numpy(pred_rand_scale_trans[:, :1]).float()
            target['img_center'] = torch.zeros(self.seq_len, 2).float()

            uv_error = np.sqrt(np.sum((gt_uv_29 - pred_uv_29)**2, axis=-1))
            acc_flag = (uv_error < 0.2) * 1.0
            # acc_flag = np.zeros_like(uv_error) + 0.5
            # acc_flag[uv_error < 0.1] = 1
            # acc_flag[uv_error > 0.2] = 0
            target['is_accurate'] = torch.from_numpy(acc_flag).unsqueeze(-1).float()

            if self.use_pretrained_feat:
                target['img_feat'] = torch.zeros(self.seq_len, self.img_feat_size).float()

                if self.use_flip:
                    target['img_feat_flip'] = torch.zeros(self.seq_len, self.img_feat_size).float()

                masked_indices = np.ones((self.seq_len, 29))
                for i in range(self.seq_len):
                    rand_indices = np.random.choice(np.arange(29), replace=False, size=5)
                    masked_indices[i, rand_indices] = 0

                target['masked_indices'] = torch.from_numpy(masked_indices).float()
            else:
                rand_img = np.random.rand(self.seq_len, 3, 256, 256)
                img_mean = np.array([-0.406, -0.457, -0.480]).reshape(3, 1, 1)
                img_std = np.array([0.225, 0.224, 0.229]).reshape(3, 1, 1)
                rand_img = (rand_img - img_mean) / img_std / 2
                target['video'] = torch.from_numpy(rand_img).float()

            # if self.use_joint_part_seg:
            #     target['joint_part_seg'] = torch.from_numpy(draw_joints4body_parts_batch(64, pred_uv_29)).float()
        return target

    def calc_bone_length_29(self, xyz):
        # xyz: batch x 29 x 3
        bone_pair_diff = [xyz[:, item1, :] - xyz[:, item2, :] for item1, item2 in self.skeleton_29jts]  # list of (seq_len, 3)
        bone_length = [np.sqrt((item**2).sum(axis=-1)).reshape(self.seq_len, 1) for item in bone_pair_diff]

        return np.concatenate(bone_length, axis=1)

    def projection(self, pred_joints, pred_camera, focal_length=1000.0):
        camDepth = focal_length / (256.0 * pred_camera[:, [0]] + 1e-9)  # batch x 1
        transl = np.concatenate([pred_camera[:, 1:3], camDepth], axis=1)
        pred_joints_cam = pred_joints + transl.reshape(pred_joints.shape[0], 1, 3)

        pred_keypoints_2d = pred_joints_cam[:, :, :2] / pred_joints_cam[:, :, [2]] * focal_length / 256.0
        return pred_keypoints_2d

    def flip_aug(self, x, s_id, e_id, left_indices, right_indices):
        x_left = x[s_id:e_id, left_indices, :].copy()
        x[s_id:e_id, left_indices, :] = x[s_id:e_id, right_indices, :].copy()
        x[s_id:e_id, right_indices, :] = x_left
        # print(x[s_id, left_indices, 0], x[s_id, right_indices, 0])
        return x

    def wrong_flip_augment(self, rand_xyz_29, rand_xyz_29_tmp, pred_uv_29, pred_score):

        if random.random() < 0.001:
            s_idx, e_idx = random.randint(-self.seq_len // 5, self.seq_len - 1), random.randint(1, self.seq_len + self.seq_len // 5)
            while s_idx > e_idx:
                s_idx, e_idx = random.randint(-self.seq_len // 5, self.seq_len - 1), random.randint(1, self.seq_len + self.seq_len // 5)

            s_idx = max(s_idx, 0)
            e_idx = min(e_idx, self.seq_len)
            upperbody_pair = [[13, 16, 18, 20, 22, 25], [14, 17, 19, 21, 23, 26]]
            lowerbody_pair = [[1, 4, 7, 10, 27], [2, 5, 8, 11, 28]]

            p = random.random()
            left_idx, right_idx = [], []
            if p < 0.25:
                left_idx, right_idx = upperbody_pair[0], upperbody_pair[1]
            elif p < 0.75:
                per_joint_p = np.random.rand(6)
                choosed_idx = [i for i in range(6) if per_joint_p[i] > 0.5]
                left_idx = [upperbody_pair[0][idx] for idx in choosed_idx]
                right_idx = [upperbody_pair[1][idx] for idx in choosed_idx]

            p = random.random()
            left_idx2, right_idx2 = [], []
            if p < 0.25:
                left_idx2, right_idx2 = lowerbody_pair[0], lowerbody_pair[1]
            elif p < 0.75:
                per_joint_p = np.random.rand(5)
                choosed_idx = [i for i in range(5) if per_joint_p[i] > 0.5]
                left_idx2 = [lowerbody_pair[0][idx] for idx in choosed_idx]
                right_idx2 = [lowerbody_pair[1][idx] for idx in choosed_idx]

            left_idx = left_idx + left_idx2
            right_idx = right_idx + right_idx2
            if len(left_idx) > 0:
                rand_xyz_29 = self.flip_aug(rand_xyz_29, s_idx, e_idx, left_idx, right_idx)
                rand_xyz_29_tmp = self.flip_aug(rand_xyz_29_tmp, s_idx, e_idx, left_idx, right_idx)
                pred_uv_29 = self.flip_aug(pred_uv_29, s_idx, e_idx, left_idx, right_idx)
                pred_score = self.flip_aug(pred_score, s_idx, e_idx, left_idx, right_idx)

        return rand_xyz_29, rand_xyz_29_tmp, pred_uv_29, pred_score


def rectify_pose(pose):
    """
    Rectify "upside down" people in global coord

    Args:
        pose (72,): Pose.
    Returns:
        Rotated pose.
    """
    pose = pose.copy()
    R_mod = cv2.Rodrigues(np.array([np.pi * 0.5, 0, 0]))[0]
    R_root = cv2.Rodrigues(pose[:3])[0]
    # new_root = R_root.dot(R_mod)
    new_root = R_mod.dot(R_root)
    pose[:3] = cv2.Rodrigues(new_root)[0].reshape(3)
    return pose
