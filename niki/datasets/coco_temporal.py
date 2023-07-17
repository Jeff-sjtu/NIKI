import os.path as osp

import cv2
import joblib
import numpy as np
import torch
import torch.utils.data as data
from pytorch3d.transforms.rotation_conversions import matrix_to_axis_angle

from niki.models.layers.smpl.SMPL import SMPL_layer
from niki.utils.bbox import _box_to_center_scale, _center_scale_to_box
from niki.utils.data_utils import split_into_chunks
# from niki.models.regress_phi_models import *
# from niki.utils.post_process import *
from niki.utils.pose_utils import normalize_uv_temporal, reconstruction_error
from niki.utils.transforms import get_affine_transform, im_to_torch

s_coco_2_smpl_jt = [
    -1, -1, -1,
    -1, 13, 14,
    -1, 15, 16,
    -1, -1, -1,
    -1, -1, -1,
    -1,
    5, 6,
    7, 8,
    9, 10,
    -1, -1
]


def xyxy_to_center_scale_batch(bbox_xyxy):
    assert len(bbox_xyxy.shape) == 2
    new_bbox = bbox_xyxy.copy()
    new_bbox[:, 0] = (bbox_xyxy[:, 0] + bbox_xyxy[:, 2]) * 0.5
    new_bbox[:, 1] = (bbox_xyxy[:, 1] + bbox_xyxy[:, 3]) * 0.5

    new_bbox[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
    new_bbox[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]
    return new_bbox


class coco_temporal(data.Dataset):
    EVAL_JOINTS = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]

    block_list = ['s_09_act_05_subact_02_ca', 's_09_act_10_subact_02_ca', 's_09_act_13_subact_01_ca']

    action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases',
                   'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']

    parents = torch.tensor([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
                            16, 17, 18, 19, 20, 21, 15, 22, 23, 10, 11], dtype=torch.long)

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

    def __init__(self, gt_path, dataset_name, train=True, only_read=True, seq_len=16):
        self.root_idx_17 = 0
        self.root_idx_smpl = 0

        self.occlusion = False
        self.dataset_name = dataset_name
        self.train = train
        self.db_gt = self.load_db(gt_path, only_read)
        self.db_pred = self.db_gt

        self.seq_len = seq_len
        if self.train:
            overlap = (self.seq_len - 1) / float(self.seq_len)
        else:
            overlap = 0

        self.stride = int(self.seq_len * (1 - overlap) + 0.5)

        db_len = len(self.db_gt['xyz_17'])
        self.db_gt['vid_name'] = np.array(['' for _ in range(db_len)])
        # self.db_gt['frame_id'] = np.array([i for i in range(db_len)])

        self.vid_indices = split_into_chunks(self.db_gt, self.seq_len, self.stride, filtered=False)
        wrong_indices = []

        self.wrong_start_indices = [item[0] for item in wrong_indices]
        h36m_jregressor = np.load('./model_files/J_regressor_h36m.npy')
        self.smpl_layer = SMPL_layer(
            './model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
            h36m_jregressor=h36m_jregressor,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, idx):
        return self._get_item_xyz(idx)

    def img_transform(self, src, bbox):
        xmin, ymin, xmax, ymax = bbox
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio, scale_mult=self._scale_mult)
        scale = scale * 1.0

        input_size = self._input_size
        inp_h, inp_w = input_size
        trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        bbox = _center_scale_to_box(center, scale)

        img = im_to_torch(img)
        # mean
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)

        # std
        img[0].div_(0.225)
        img[1].div_(0.224)
        img[2].div_(0.229)

        img_center = np.array([float(src.shape[1]) * 0.5, float(src.shape[0]) * 0.5])

        return img, bbox, img_center

    def _get_item_xyz(self, idx):
        start_index, end_index = self.vid_indices[idx]

        images = 0

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

        smpl_weight = self.get_sequence(start_index, end_index, self.db_pred['smpl_weight']).reshape(self.seq_len, 1)
        pred_betas = self.get_sequence(start_index, end_index, self.db_pred['pred_betas']).reshape(self.seq_len, -1)[:, :10]
        gt_betas = self.get_sequence(start_index, end_index, self.db_gt['shape']).reshape(self.seq_len, 10)

        # pred_betas = gt_betas
        beta_weight = np.ones_like(gt_betas)
        beta_weight[:, :] = smpl_weight
        flag = smpl_weight.reshape(-1) > 0
        pred_betas[flag, :] = gt_betas[flag, :]

        pose_matrix = self.get_sequence(start_index, end_index, self.db_gt['pose']).reshape(self.seq_len, 24, 3, 3)
        for i in range(self.seq_len):
            if smpl_weight[i, 0] < 0.5:
                pose_matrix[i] = np.eye(3).reshape(1, 3, 3)

        pose = matrix_to_axis_angle(torch.from_numpy(pose_matrix).reshape(-1, 3, 3))
        pose = pose.reshape(self.seq_len, 72)
        theta_weight = np.ones((self.seq_len, 72))
        theta_weight = theta_weight * smpl_weight
        xyz_17_weight = xyz_17_weight * smpl_weight.reshape(self.seq_len, 1, 1)
        xyz_29_weight = xyz_29_weight * smpl_weight.reshape(self.seq_len, 1, 1)

        pred_phi = self.get_sequence(start_index, end_index, self.db_pred['pred_phi'])
        # phi = np.zeros((self.seq_len, 23, 2))
        # phi_weight = np.zeros((self.seq_len, 23, 2))
        rotmat_swing, rotmat_twist, angle_twist = self.smpl_layer.twist_swing_decompose_rot(
            torch.from_numpy(pose_matrix).float(),
            torch.from_numpy(gt_betas).float())

        phis = torch.cat((
            torch.cos(angle_twist),
            torch.sin(angle_twist)
        ), dim=2)
        phi_weight = (angle_twist > -10) * 1.0
        phi_weight = torch.cat((
            phi_weight, phi_weight
        ), dim=2) * torch.from_numpy(smpl_weight).unsqueeze(-1)

        kp_3d_29[xyz_29_weight < 0.5] = kp_3d_29_pred[xyz_29_weight < 0.5]

        is_continuous = (start_index not in self.wrong_start_indices) * 1.0
        valid_smpl = torch.ones(self.seq_len, 1).float()
        phi_mask = (angle_twist.numpy().reshape(self.seq_len, 23) > -10) * 1.0
        phi_mask = torch.from_numpy((phi_mask.sum(axis=1) > 22) * 1.0).float()
        valid_smpl = valid_smpl * phi_mask.reshape(self.seq_len, 1) * torch.from_numpy(smpl_weight)

        target = {
            'pred_betas': torch.from_numpy(pred_betas).float(),
            'gt_betas': torch.from_numpy(gt_betas).float(),
            'betas_weight': torch.from_numpy(beta_weight).float(),
            'pred_phi': torch.from_numpy(pred_phi).float(),
            'gt_phi': phis.float(),
            'phi_weight': phi_weight.float(),
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
            # 'gt_theta': torch.from_numpy(pose).float(),
            'gt_theta': pose.float(),
            'theta_weight': torch.from_numpy(theta_weight).float(),
            'is_occlusion': float(self.occlusion),
            'is_3dhp': float(0.0),
            'images': images,
            'valid_smpl': valid_smpl.float()
        }

        bbox_xyxy = self.get_sequence(start_index, end_index, self.db_pred['bbox']).reshape(self.seq_len, 4)
        scale = 1.25

        bbox = xyxy_to_center_scale_batch(bbox_xyxy)
        bbox[:, 2:] = bbox[:, 2:] * scale

        uv17 = self.get_sequence(start_index, end_index, self.db_pred['joints_3d']).reshape(self.seq_len, 17, 3, 2)
        uv29 = np.zeros((self.seq_len, 29, 2))
        uv29_weight = np.zeros((self.seq_len, 29, 2))
        for i in range(24):
            id1 = i
            id2 = s_coco_2_smpl_jt[i]
            if id2 >= 0:
                uv29[:, id1, :2] = uv17[:, id2, :2, 0].copy()
                uv29_weight[:, id1, :2] = uv17[:, id2, :2, 1].copy()

        uv_29_normalized = normalize_uv_temporal(uv29, bbox, scale=1.0)

        scale_trans = np.zeros((self.seq_len, 4))
        scale_trans[:, 0] = 0.8
        target['bbox'] = torch.from_numpy(bbox).float()
        target['gt_uv_29'] = torch.from_numpy(uv_29_normalized).float()
        target['gt_scale_trans'] = torch.from_numpy(scale_trans).float()
        target['uv_29_weight'] = torch.from_numpy(uv29_weight).float()
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

        img_width = self.get_sequence(start_index, end_index, self.db_pred['width']).reshape(self.seq_len, 1)  # img_ann['width'], img_ann['height']
        img_height = self.get_sequence(start_index, end_index, self.db_pred['height']).reshape(self.seq_len, 1)  # img_ann['width'], img_ann['height']
        img_sizes = np.concatenate([img_width, img_height], axis=1)
        img_center = img_sizes * 0.5
        img_center_bbox_coord = (img_center - (bbox[:, :2] - bbox[:, 2:] * 0.5)) / bbox[:, 2:]  # 0-1
        img_center_bbox_coord = (img_center_bbox_coord - 0.5) * 256.0
        target['img_center'] = torch.from_numpy(img_center_bbox_coord).float()
        # target['img_sizes'] = torch.from_numpy(img_sizes).float()

        acc_flag = np.zeros((self.seq_len, 29))
        target['is_accurate'] = torch.from_numpy(acc_flag).unsqueeze(-1).float()

        img_path = self.get_sequence(start_index, end_index, self.db_pred['img_path'], to_np=False)
        target['img_path'] = [str(item) for item in img_path]

        img_feat = self.get_sequence(start_index, end_index, self.db_pred['features']).reshape(self.seq_len, -1)
        if img_feat.shape[-1] == 512:
            ones_pad = np.ones((self.seq_len, 512))
            img_feat = np.concatenate([img_feat, ones_pad], axis=-1)

        target['img_feat'] = torch.from_numpy(img_feat).float()

        masked_indices = np.ones((self.seq_len, 29))
        for i in range(self.seq_len):
            rand_indices = np.random.choice(np.arange(29), replace=False, size=3)
            masked_indices[i, rand_indices] = 0

        target['masked_indices'] = torch.from_numpy(masked_indices).float()

        # for k, v in target.items():
        #     print(k, type(v))

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

    def batch_calc_bone_length_29(self, xyz):
        # xyz: batch x 29 x 3
        bone_pair_diff = [xyz[:, item1, :] - xyz[:, item2, :] for item1, item2 in self.skeleton_29jts]  # list of (batch, 3)
        bone_length = [torch.sqrt((item**2).sum(dim=-1)) for item in bone_pair_diff]  # list of (batch, )

        return torch.stack(bone_length, dim=1)

    def calc_bone_length_29(self, xyz):
        # xyz: batch x 29 x 3
        bone_pair_diff = [xyz[item1, :] - xyz[item2, :] for item1, item2 in self.skeleton_29jts]  # list of (3)
        bone_length = [np.sqrt((item**2).sum(axis=-1)) for item in bone_pair_diff]

        return np.array(bone_length)