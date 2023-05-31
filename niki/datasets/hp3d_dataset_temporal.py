import os.path as osp
import random

import cv2
import joblib
import numpy as np
import torch
import torch.utils.data as data

from niki.utils.data_utils import split_into_chunks, split_into_chunks_filtered
from niki.utils.kp_utils import (convert_kps, get_perm_idxs,
                                 translate_mpii3d_imgname,
                                 translate_mpii3d_imgname2)
from niki.utils.occlusion_utils import translate_img_path
# from niki.models.layers.smpl.SMPL import SMPL_layer
from niki.utils.pose_utils import (calc_cam_scale_trans_const_scale,
                                   calc_cam_scale_trans_refined1,
                                   normalize_uv_temporal, reconstruction_error)
# from niki.utils.post_process import *
from niki.utils.transforms import (get_affine_transform, get_single_image_crop,
                                   im_to_torch)

# from niki.beta_decompose.draw_joints import draw_joints4body_parts_batch


class hp3d_dataset_temporal(data.Dataset):
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

    all_joint_names = [
        'spine3', 'spine4', 'spine2', 'spine', 'pelvis',  # 4
        'neck', 'head', 'head_top', 'left_clavicle', 'left_shoulder', 'left_elbow',  # 10
        'left_wrist', 'left_hand', 'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist',  # 16
        'right_hand', 'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe',  # 22
        'right_hip', 'right_knee', 'right_ankle', 'right_foot', 'right_toe']

    s_spin_2_smpl_jt = [
        39, 28, 27,  # pelvis, _, _
        -1, 29, 26,  # _, left_knee, right_knee
        -1, 30, 25,  # _, left_ankle, right_ankle
        -1, -1, -1,  # _, _, _
        37, -1, -1,  # neck, _, _
        -1,  # _
        34, 33,  # left shoulder, right_shoulder
        35, 32,  # left elbow, right elbow
        36, 31,  # left wrist, right wrist
        -1, -1,
        38,  # head top
        -1, -1,  # _, _
        -1, -1  # left foot, right foot
    ]

    s_3dhp_2_smpl_jt = [
        4, 18, 23,  # pelvis , lhip, rhip
        -1, 19, 24,  # _, left_knee("lknee"), right_knee
        -1, 20, 25,  # _, left_ankle, right_ankle
        -1, -1, -1,  # _, _, _
        5, -1, -1,  # neck("neck"), _, _
        -1,  # _
        9, 14,  # left shoulder, right_shoulder
        10, 15,  # left elbow, right elbow
        11, 16,  # left wrist, right wrist
        -1, -1,
        7,  # head top
        -1, -1,  # _, _
        # 21, 26 # left foot, right foot
        -1, -1
    ]

    s_3dhp_2_h36m_jt = [
        4,
        18, 19, 20,
        23, 24, 25,
        -1, 5,
        -1, 6,
        9, 10, 11,
        14, 15, 16
    ]

    def __init__(self, gt_path, pred_path, dataset_name, train=False, usage='phi', only_read=True,
                 use_amass=False, seq_len=16, use_pretrained_feat=True, use_joint_part_seg=False, load_img=False):
        self.root_idx_17 = 0
        self.root_idx_smpl = 0

        self.occlusion = ('amb' in gt_path)
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
        # filtered = (self.dataset_name == '3dpw')
        # self.vid_indices = split_into_chunks(self.db_gt, self.seq_len, self.stride, filtered)
        if True:
            self.vid_indices, wrong_indices = split_into_chunks_filtered(self.db_gt, self.seq_len, self.stride)
        else:
            self.vid_indices = split_into_chunks(self.db_gt, self.seq_len, self.stride, filtered=False)
            wrong_indices = []

        self.wrong_start_indices = [item[0] for item in wrong_indices]
        print(self.wrong_start_indices)

        self.usage = usage
        assert self.db_gt['joints3D'].shape[1] == 49

        self.return_jts2d = True
        self.use_amass = use_amass

        self.use_pretrained_feat = use_pretrained_feat
        self.input_size = 256
        self.use_joint_part_seg = use_joint_part_seg

        self.load_img = load_img

    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, idx):
        return self._get_item_xyz(idx)

    def _get_item_phi(self, idx):
        return None

    def _get_item_xyz(self, idx):
        start_index, end_index = self.vid_indices[idx]

        if self.load_img:
            img_paths = self.get_sequence(start_index, end_index, self.db_gt['img_paths'])
            center_scale = self.get_sequence(start_index, end_index, self.db_gt['amb_center_scale'])
            occ_xyxy = self.get_sequence(start_index, end_index, self.db_gt['amb_synth_size'])

            images = []
            for t in range(len(img_paths)):
                path = img_paths[t]
                src = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                center = center_scale[t, :2]
                scale = center_scale[t, 2:]

                synth_xmin, synth_ymin, synth_xmax, synth_ymax = occ_xyxy[t, :]
                src[synth_ymin:synth_ymax, synth_xmin:synth_xmax, :] = np.random.rand(synth_ymax - synth_ymin, synth_xmax - synth_xmin, 3) * 255

                trans = get_affine_transform(center, scale, rot=0, output_size=[256, 256])
                img = cv2.warpAffine(src, trans, (int(256), int(256)), flags=cv2.INTER_LINEAR)

                img = im_to_torch(img)

                images.append(img)
            images = torch.stack(images, dim=0)
        else:
            images = None

        if self.train:
            kp_3d_49 = self.get_sequence(start_index, end_index, self.db_gt['joints3D'])
            kp_3d_14 = convert_kps(kp_3d_49.copy(), src='spin', dst='common')

            kp_3d_29 = np.zeros((self.seq_len, 29, 3))
            xyz_29_weight = np.zeros_like(kp_3d_29)

            kp29_indices = get_perm_idxs(src='smpl29', dst='common')  # get index in src format
            assert (len(kp29_indices) == 14)
            kp_3d_29[:, kp29_indices] = kp_3d_14.copy()
            kp_3d_29 = kp_3d_29 - kp_3d_29[:, [1, 2], :].mean(axis=1, keepdims=True)
            xyz_29_weight[:, kp29_indices] = 1

            kp_3d_29_pred = self.get_sequence(start_index, end_index, self.db_pred['pred_xyz_29']) * 2.2
            kp_3d_24_struct_pred = self.get_sequence(start_index, end_index, self.db_pred['pred_xyz_24_struct']) * 2.2
            pred_score = self.get_sequence(start_index, end_index, self.db_pred['pred_scores']) * 0.1
            pred_score = pred_score.reshape(self.seq_len, 29, 1)
            pred_sigma = self.get_sequence(start_index, end_index, self.db_pred['pred_sigma'])
            pred_sigma = pred_sigma.reshape(self.seq_len, 29, 1)

            kp_3d_29_pred = kp_3d_29_pred - kp_3d_29_pred[:, [1, 2], :].mean(axis=1, keepdims=True)
            kp_3d_24_struct_pred = kp_3d_24_struct_pred - kp_3d_24_struct_pred[:, [1, 2], :].mean(axis=1, keepdims=True)

            kp_3d_17 = np.zeros((self.seq_len, 17, 3))
            xyz_17_weight = np.zeros_like(kp_3d_17)

            kp17_indices = get_perm_idxs(src='h36m', dst='common')
            kp_3d_17[:, kp17_indices] = kp_3d_14.copy()
            xyz_17_weight[:, kp17_indices] = 1
            kp_3d_17 = kp_3d_17 - kp_3d_17[:, [1, 4], :].mean(axis=1, keepdims=True)

            pred_phi = self.get_sequence(start_index, end_index, self.db_pred['pred_phi'])
            phi = np.zeros((self.seq_len, 23, 2))
            phi[:, :, 0] = 1
            phi_weight = np.zeros_like(phi)

            pred_betas = self.get_sequence(start_index, end_index, self.db_pred['pred_betas']).reshape(self.seq_len, 10)
            gt_betas = np.zeros((self.seq_len, 10))
            beta_weight = np.zeros_like(gt_betas)

            add_beta_noise = True
            if add_beta_noise:
                noise_scale = 0
                noise = np.random.randn(self.seq_len, 10) * noise_scale
                pred_betas = gt_betas + noise

            # pred_betas = gt_betas
            is_continuous = (start_index not in self.wrong_start_indices) * 1.0

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
                # 'theta': torch.zeros(self.seq_len, 72).float(),
                'gt_theta': torch.zeros(self.seq_len, 72).float(),
                'theta_weight': torch.zeros(self.seq_len, 72).float(),
                'is_occlusion': float(self.occlusion),
                'is_3dhp': float(1.0),
                'valid_smpl': torch.zeros(self.seq_len, 1).float(),
                'images': 0
            }

            if self.return_jts2d:
                if self.occlusion:
                    bbox = self.get_sequence(start_index, end_index, self.db_pred['amb_center_scale'])
                else:
                    bbox = self.get_sequence(start_index, end_index, self.db_pred['bbox']).reshape(self.seq_len, 4)
                    scale = 1.2

                    bbox[:, 2:] = bbox[:, 2:] * scale

                uv_29, uv_29_normalized, uv_29_weight, scale_trans = self.get_uv24_cam(
                    self.db_gt, start_index, end_index, bbox, kp_3d_29, xyz_29_weight
                )
                target['bbox'] = torch.from_numpy(bbox).float()
                target['gt_uv_29'] = torch.from_numpy(uv_29_normalized).float()
                target['gt_scale_trans'] = torch.from_numpy(scale_trans).float()
                target['uv_29_weight'] = torch.from_numpy(uv_29_weight).float()
                target['rand_xyz_29'] = torch.from_numpy(kp_3d_29_pred).float()

                # pred_cam = self.get_sequence(start_index, end_index, self.db_pred['pred_camera']).reshape(self.seq_len, -1)

                # pred_cam_root = self.get_sequence(start_index, end_index, self.db_pred['pred_cam_root']).reshape(self.seq_len, 3)
                # pred_cam = np.concatenate((
                #     1000.0 / (256 * pred_cam_root[:, [0]] + 1e-9),
                #     pred_cam_root[:, 1:]
                # ), axis=1)

                # pred_cam = self.get_sequence(start_index, end_index, self.db_pred['pred_camera']).reshape(self.seq_len, 3)
                pred_cam_scale = self.get_sequence(start_index, end_index, self.db_pred['pred_camera']).reshape(self.seq_len, 1)
                pred_cam = np.concatenate((
                    pred_cam_scale,
                    np.zeros_like(pred_cam_scale),
                    np.zeros_like(pred_cam_scale)
                ), axis=1)
                target['pred_cam'] = torch.from_numpy(pred_cam).float()
                target['pred_cam_scale'] = torch.from_numpy(pred_cam_scale).float()

                gt_valid_mask = ((target['gt_uv_29'] <= 0.5) & (target['gt_uv_29'] >= -0.5)) * 1.0
                gt_valid_mask = (gt_valid_mask.sum(axis=-1) > 1.5) * 1.0
                weight_mask = (target['uv_29_weight'].sum(axis=-1) > 1.5)
                gt_valid_mask[~weight_mask] = -1
                target['gt_valid_mask'] = gt_valid_mask

                pred_uv_29 = self.get_sequence(start_index, end_index, self.db_pred['pred_uvd']).reshape(self.seq_len, 29, 3)[:, :, :2]
                pred_uvd_29 = self.get_sequence(start_index, end_index, self.db_pred['pred_uvd']).reshape(self.seq_len, 29, 3)
                pred_uvd_29[:, :, 2] = pred_uvd_29[:, :, 2] * 2.2
                target['pred_uv_29'] = torch.from_numpy(pred_uv_29).float()
                target['pred_uvd_29'] = torch.from_numpy(pred_uvd_29).float()

                # pred_uvd_29 = self.get_sequence(start_index, end_index, self.db_pred['pred_uvd']).reshape(self.seq_len, 29, 3)
                # target['pred_uvd_29'] = torch.from_numpy(pred_uvd_29).float()

                img_sizes = np.array([2048, 2048])
                # assert (bbox[:, :2] <= img_sizes).all()
                img_center = img_sizes * 0.5
                img_center_bbox_coord = (img_center - (bbox[:, :2] - bbox[:, 2:] * 0.5)) / bbox[:, 2:]  # 0-1
                img_center_bbox_coord = (img_center_bbox_coord - 0.5) * 256.0
                target['img_center'] = torch.from_numpy(img_center_bbox_coord).float()
                target['is_accurate'] = torch.ones(self.seq_len, 29, 1).float() * (-1)

                img_paths = []
                for idx in range(start_index, end_index + 1):
                    raw_img_path = self.db_pred['img_name'][idx].split('/')[2:]
                    if raw_img_path[0][0] == 'S':  # train
                        img_path = translate_mpii3d_imgname([self.db_pred['img_name'][idx]])[0]
                    else:
                        img_path = translate_mpii3d_imgname2([self.db_pred['img_name'][idx]])[0]
                    img_paths.append(img_path)

                target['img_path'] = img_paths

                if self.use_pretrained_feat:
                    img_feat = self.get_sequence(start_index, end_index, self.db_pred['features']).reshape(self.seq_len, -1)
                    # ones_pad = np.ones((self.seq_len, 512))
                    # img_feat = np.concatenate([img_feat, ones_pad], axis=-1)
                    if img_feat.shape[-1] == 512:
                        ones_pad = np.ones((self.seq_len, 512))
                        img_feat = np.concatenate([img_feat, ones_pad], axis=-1)

                    target['img_feat'] = torch.from_numpy(img_feat).float()

                    masked_indices = np.ones((self.seq_len, 29))
                    # np.random.choice(np.arange(29),replace=False,size=0.03)
                    target['masked_indices'] = torch.from_numpy(masked_indices).float()

                    if 'features_flip' in self.db_pred:
                        img_feat_flip = self.get_sequence(start_index, end_index, self.db_pred['features_flip']).reshape(self.seq_len, 512)
                        img_feat_flip = np.concatenate([img_feat_flip, ones_pad], axis=-1)
                        target['img_feat_flip'] = torch.from_numpy(img_feat_flip).float()
                else:

                    video = torch.cat(
                        [get_single_image_crop(image, bbox_tmp, w=self.input_size, h=self.input_size).unsqueeze(0) for image, bbox_tmp in zip(img_paths, bbox)], dim=0
                    )
                    # print(len(img_paths), len(bbox), zip(img_paths, bbox))
                    # print(video.shape, '3dhp')
                    target['video'] = video.float()

            if 'features_flip' in self.db_pred:
                img_feat_flip = self.get_sequence(start_index, end_index, self.db_pred['features_flip']).reshape(self.seq_len, 512)
                ones_pad = np.ones((self.seq_len, 512))
                img_feat_flip = np.concatenate([img_feat_flip, ones_pad], axis=-1)
                target['img_feat_flip'] = torch.from_numpy(img_feat_flip).float()

            if 'features_flip' in self.db_pred and random.random() < 0.2:
                target = self.flip_target(start_index, end_index, target)
        else:
            kp_3d_14 = convert_kps(self.get_sequence(start_index, end_index, self.db_gt['joints3D']), src='spin', dst='common')

            kp_3d_29 = np.zeros((self.seq_len, 29, 3))
            xyz_29_weight = np.zeros_like(kp_3d_29)
            kp29_indices = get_perm_idxs(src='smpl29', dst='common')  # get index in src format
            kp_3d_29[:, kp29_indices] = kp_3d_14.copy()
            xyz_29_weight[:, kp29_indices] = 1

            kp_3d_29_pred = self.get_sequence(start_index, end_index, self.db_pred['pred_xyz_29']) * 2.2
            kp_3d_24_struct_pred = self.get_sequence(start_index, end_index, self.db_pred['pred_xyz_24_struct']).reshape(self.seq_len, 24, 3) * 2.2
            kp_3d_17_pred = self.get_sequence(start_index, end_index, self.db_pred['pred_xyz_17']) * 2.2
            pred_score = self.get_sequence(start_index, end_index, self.db_pred['pred_scores']) * 0.1
            pred_sigma = self.get_sequence(start_index, end_index, self.db_pred['pred_sigma'])

            kp_3d_29 = kp_3d_29 - kp_3d_29[:, [1, 2], :].mean(axis=1, keepdims=True)
            kp_3d_29_pred = kp_3d_29_pred - kp_3d_29_pred[:, [1, 2], :].mean(axis=1, keepdims=True)
            kp_3d_24_struct_pred = kp_3d_24_struct_pred - kp_3d_24_struct_pred[:, [1, 2], :].mean(axis=1, keepdims=True)

            kp17_indices = get_perm_idxs(src='h36m', dst='common')  # get index in src format
            kp_3d_17 = np.zeros((self.seq_len, 17, 3))
            kp_3d_17[:, kp17_indices] = kp_3d_14.copy()

            pred_phi = self.get_sequence(start_index, end_index, self.db_pred['pred_phi'])
            phi = np.zeros((self.seq_len, 23, 2))
            phi[:, :, 0] = 1
            phi_weight = np.zeros_like(phi)
            beta = np.zeros((self.seq_len, 10))
            beta_weight = np.zeros_like(beta)
            # pred_theta = self.get_sequence(start_index, end_index, self.db_pred['thetas'])
            pred_betas = self.get_sequence(start_index, end_index, self.db_pred['pred_betas'])

            img_name = self.get_sequence(start_index, end_index, self.db_gt['img_name'])
            img_name = [str(item) for item in img_name]

            # pred_camera = self.get_sequence(start_index, end_index, self.db_gt['pred_camera'])

            # pred_cam_root = self.get_sequence(start_index, end_index, self.db_pred['pred_cam_root']).reshape(self.seq_len, 3)
            # pred_cam = np.concatenate((
            #     1000.0 / (256 * pred_cam_root[:, [0]] + 1e-9),
            #     pred_cam_root[:, 1:]
            # ), axis=1)

            pred_cam_scale = self.get_sequence(start_index, end_index, self.db_pred['pred_camera']).reshape(self.seq_len, 1)
            pred_cam = np.concatenate((
                pred_cam_scale,
                np.zeros_like(pred_cam_scale),
                np.zeros_like(pred_cam_scale)
            ), axis=1)

            is_continuous = (start_index not in self.wrong_start_indices) * 1.0

            target = {
                'img_name': img_name,
                'gt_xyz_29': torch.from_numpy(kp_3d_29).float(),
                'pred_xyz_29': torch.from_numpy(kp_3d_29_pred).float(),
                'pred_xyz_24_struct': torch.from_numpy(kp_3d_24_struct_pred).float(),
                'pred_xyz_17': torch.from_numpy(kp_3d_17_pred).float(),
                'pred_score': torch.from_numpy(pred_score).float(),
                'pred_sigma': torch.from_numpy(pred_sigma).float(),
                'xyz_29_weight': torch.from_numpy(xyz_29_weight).float(),
                'gt_xyz_17': torch.from_numpy(kp_3d_17).float(),
                'pred_phi': torch.from_numpy(pred_phi).float(),
                'gt_phi': torch.from_numpy(phi).float(),
                'pred_cam': torch.from_numpy(pred_cam).float(),
                'pred_cam_scale': torch.from_numpy(pred_cam_scale).float(),
                'pred_betas': torch.from_numpy(pred_betas).float(),
                'gt_theta': torch.zeros(self.seq_len, 72).float(),
                'theta_weight': torch.zeros(self.seq_len, 72).float(),
                'is_continuous': float(is_continuous),
                'images': images
            }

            if self.return_jts2d:
                # pred_cam = self.get_sequence(start_index, end_index, self.db_pred['pred_camera']).reshape(self.seq_len, -1)

                # pred_cam_root = self.get_sequence(start_index, end_index, self.db_pred['pred_cam_root']).reshape(self.seq_len, 3)
                # pred_cam = np.concatenate((
                #     1000.0 / (256 * pred_cam_root[:, [0]] + 1e-9),
                #     pred_cam_root[:, 1:]
                # ), axis=1)

                pred_cam_scale = self.get_sequence(start_index, end_index, self.db_pred['pred_camera']).reshape(self.seq_len, 1)
                pred_cam = np.concatenate((
                    pred_cam_scale,
                    np.zeros_like(pred_cam_scale),
                    np.zeros_like(pred_cam_scale)
                ), axis=1)
                target['pred_cam'] = torch.from_numpy(pred_cam).float()
                target['pred_cam_scale'] = torch.from_numpy(pred_cam_scale).float()

                pred_uv_29 = self.get_sequence(start_index, end_index, self.db_pred['pred_uvd']).reshape(self.seq_len, 29, 3)[:, :, :2]
                pred_uvd_29 = self.get_sequence(start_index, end_index, self.db_pred['pred_uvd']).reshape(self.seq_len, 29, 3)
                pred_uvd_29[:, :, 2] = pred_uvd_29[:, :, 2] * 2.2
                target['pred_uv_29'] = torch.from_numpy(pred_uv_29).float()
                target['pred_uvd_29'] = torch.from_numpy(pred_uv_29).float()

                if self.occlusion:
                    bbox = self.get_sequence(start_index, end_index, self.db_pred['amb_center_scale'])
                    syn_size = self.get_sequence(start_index, end_index, self.db_pred['amb_synth_size'])
                    target['amb_synth_size'] = torch.from_numpy(syn_size).type(torch.int32)
                    img_paths = self.get_sequence(start_index, end_index, self.db_pred['img_paths'], to_np=False)
                    target['img_path'] = list(img_paths)
                else:
                    bbox = self.get_sequence(start_index, end_index, self.db_pred['bbox']).reshape(self.seq_len, 4)
                    scale = 1.2

                    bbox[:, 2:] = bbox[:, 2:] * scale

                    img_names = self.get_sequence(start_index, end_index, self.db_pred['img_name'], to_np=False)
                    img_paths = [translate_img_path(img_name, self.dataset_name) for img_name in img_names]
                    target['img_path'] = img_paths

                target['bbox'] = torch.from_numpy(bbox).float()

                target['gt_uv_29'] = torch.zeros(self.seq_len, 29, 2)
                target['uv_29_weight'] = torch.zeros(self.seq_len, 29, 2)

                img_sizes = np.array([2048, 2048])
                # assert (bbox[:, :2] <= img_sizes).all()
                img_center = img_sizes * 0.5
                img_center_bbox_coord = (img_center - (bbox[:, :2] - bbox[:, 2:] * 0.5)) / bbox[:, 2:]  # 0-1
                img_center_bbox_coord = (img_center_bbox_coord - 0.5) * 256.0
                target['img_center'] = torch.from_numpy(img_center_bbox_coord).float()
                target['img_sizes'] = torch.from_numpy(img_sizes).reshape(1, 2).expand(self.seq_len, -1).float()

                if self.use_pretrained_feat:
                    img_feat = self.get_sequence(start_index, end_index, self.db_pred['features']).reshape(self.seq_len, -1)
                    if img_feat.shape[-1] == 512:
                        ones_pad = np.ones((self.seq_len, 512))
                        img_feat = np.concatenate([img_feat, ones_pad], axis=-1)

                    target['img_feat'] = torch.from_numpy(img_feat).float()
                    if 'features_flip' in self.db_pred:
                        img_feat_flip = self.get_sequence(start_index, end_index, self.db_pred['features_flip']).reshape(self.seq_len, 512)
                        img_feat_flip = np.concatenate([img_feat_flip, ones_pad], axis=-1)
                        target['img_feat_flip'] = torch.from_numpy(img_feat_flip).float()
                else:
                    img_paths = []
                    for idx in range(start_index, end_index + 1):
                        raw_img_path = self.db_pred['img_name'][idx].split('/')[2:]
                        if raw_img_path[0][0] == 'S':  # train
                            img_path = translate_mpii3d_imgname([self.db_pred['img_name'][idx]])[0]
                        else:
                            img_path = translate_mpii3d_imgname2([self.db_pred['img_name'][idx]])[0]
                        img_paths.append(img_path)

                    video = torch.cat(
                        [get_single_image_crop(image, bbox_tmp, w=self.input_size, h=self.input_size).unsqueeze(0) for image, bbox_tmp in zip(img_paths, bbox)], dim=0
                    )
                    # print(video.shape, '3dhp')
                    target['video'] = video.float()

        # if self.use_joint_part_seg:
        #     target['joint_part_seg'] = torch.from_numpy(draw_joints4body_parts_batch(64, pred_uv_29)).float()

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
            # v = data[start_index:start_index + 1].repeat(self.seq_len, axis=0)

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

    def evaluate_xyz_29(self, preds, gts):
        assert len(gts) == len(preds)
        sample_num = len(gts)

        kp29_indices = get_perm_idxs(src='smpl29', dst='common')

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

            gt_3d_kpt = gt_3d_kpt * 1000.0
            pred_3d_kpt = pred_3d_kpt * 1000.0

            # root joint alignment
            pred_3d_kpt = pred_3d_kpt[kp29_indices, :]
            gt_3d_kpt = gt_3d_kpt[kp29_indices, :]

            # print(pred_3d_kpt, gt_3d_kpt)

            pred_pelvis = (pred_3d_kpt[[2], :] + pred_3d_kpt[[3], :]) / 2.0
            target_pelvis = (gt_3d_kpt[[2], :] + gt_3d_kpt[[3], :]) / 2.0

            pred_3d_kpt -= pred_pelvis
            gt_3d_kpt -= target_pelvis

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

        eval_summary = f'PRED XYZ_17_raw tot: {tot_err:2f}, tot_pa: {tot_err_align:2f}, x: {tot_err_x:2f}, y: {tot_err_y:.2f}, z: {tot_err_z:2f}'

        print(eval_summary)

        return tot_err_align, tot_err

    def evaluate_xyz_17(self, preds, gts):
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

            gt_3d_kpt = gt_3d_kpt * 1000.0
            pred_3d_kpt = pred_3d_kpt * 1000.0

            # root joint alignment
            pred_3d_kpt = np.take(pred_3d_kpt, self.EVAL_JOINTS, axis=0)
            gt_3d_kpt = np.take(gt_3d_kpt, self.EVAL_JOINTS, axis=0)

            pred_pelvis = (pred_3d_kpt[[2], :] + pred_3d_kpt[[3], :]) / 2.0
            target_pelvis = (gt_3d_kpt[[2], :] + gt_3d_kpt[[3], :]) / 2.0

            pred_3d_kpt -= pred_pelvis
            gt_3d_kpt -= target_pelvis

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

        eval_summary = f'PRED XYZ_17 tot: {tot_err:2f}, tot_pa: {tot_err_align:2f}, x: {tot_err_x:2f}, y: {tot_err_y:.2f}, z: {tot_err_z:2f}'

        print(eval_summary)

        return tot_err_align, tot_err

    def get_uv24_cam(self, db, start_index, end_index, bbox, kp_3d_29, xyz_29_weight):
        uv_28 = self.get_sequence(start_index, end_index, db['kpts_2d']).reshape(self.seq_len, 28, 3)
        uv_29 = np.zeros((self.seq_len, 29, 2))
        uv_29_weight = np.zeros((self.seq_len, 29, 2))
        uvd_29_weight = np.zeros((self.seq_len, 29, 3))
        for i, item in enumerate(self.s_3dhp_2_smpl_jt):
            if item >= 0:
                uv_29[:, i] = uv_28[:, item, :2]
                uv_29_weight[:, i] = uv_28[:, item, [2]]
                uvd_29_weight[:, i] = uv_28[:, item, [2]]

        scale_trans = np.zeros((self.seq_len, 4))
        uv_29_normalized = normalize_uv_temporal(uv_29, bbox, 1.0)
        # uv_24_normalized = normalize_uv_temporal(uv_29[:, :24], bbox, 1.0)

        # xyz_24 = kp_3d_29[:, :24]
        # xyz_24_weight = xyz_29_weight[:, :24] * uvd_29_weight[:, :24]

        xyz_29 = kp_3d_29
        xyz_29_weight = xyz_29_weight * uvd_29_weight

        img_sizes = self.get_sequence(start_index, end_index, db['img_size']).reshape(self.seq_len, 2)  # img_ann['width'], img_ann['height']
        # img_sizes = np.zeros((self.seq_len, 2))
        # img_sizes[:] = 2048.0
        img_center = img_sizes * 0.5
        img_center_bbox_coord = (img_center - (bbox[:, :2] - bbox[:, 2:] * 0.5)) / bbox[:, 2:]  # 0-1
        img_center_bbox_coord = (img_center_bbox_coord - 0.5) * 256.0

        # uv_24_normalized = np.zeros_like(uv_24)
        for idx in range(self.seq_len):
            # crop image and transform 2d keypoints
            # cam_scale_trans, cam_valid, diff = calc_cam_scale_trans(
            #                             xyz_24[idx].reshape(-1, 3).copy(),
            #                             uv_24_normalized[idx].reshape(-1, 2).copy(),
            #                             xyz_24_weight[idx],
            #                             img_center=img_center_bbox_coord[idx])

            cam_scale_trans2, cam_valid2, diff2 = calc_cam_scale_trans_refined1(
                xyz_29[idx].reshape(-1, 3).copy(),
                uv_29_normalized[idx].reshape(-1, 2).copy(),
                xyz_29_weight[idx],
                img_center_bbox_coord[idx])

            used_scale = cam_scale_trans2[0]
            trans, _, diff3 = calc_cam_scale_trans_const_scale(
                xyz_29[idx].reshape(-1, 3).copy(),
                uv_29_normalized[idx].reshape(-1, 2).copy(),
                xyz_29_weight[idx],
                used_scale)
            # if diff > 0:
            #     print(cam_scale_trans, cam_valid, diff)
            #     print([used_scale, trans[0], trans[1]], cam_valid2, diff2, diff3, 'new')
            cam_scale_trans2[1:] = trans

            scale_trans[idx, :3] = cam_scale_trans2
            scale_trans[idx, 3] = cam_valid2

        return uv_29, uv_29_normalized, uv_29_weight, scale_trans

    def flip_target(self, start_index, end_index, target):
        raise NotImplementedError
        img_feat_flip = target['img_feat_flip'].clone()
        target['img_feat_flip'] = target['img_feat'].clone()
        target['img_feat'] = img_feat_flip

        # 'phi', 'gt_xyz_29', 'pred_xyz_29', 'pred_score', 'gt_xyz_17'
        # 'gt_uv_29', 'gt_scale_trans', 'rand_xyz_29', 'pred_uv_29', 'pred_cam', 'img_center'
        target['pred_phi'] = self.flip_phi23(target['pred_phi'].clone())
        target['gt_phi'] = self.flip_phi23(target['gt_phi'].clone())
        target['gt_xyz_29'], target['pred_xyz_29'], target['pred_score'], target['gt_uv_29'], target['rand_xyz_29'], target['pred_uv_29'] = \
            map(self.flip_kpt29, [target['gt_xyz_29'].clone(), target['pred_xyz_29'].clone(), target['pred_score'].clone(), target['gt_uv_29'].clone(), target['rand_xyz_29'].clone(), target['pred_uv_29'].clone()])

        target['pred_xyz_24_struct'] = self.flip_kpt24(target['pred_xyz_24_struct'].clone())
        target['pred_sigma'] = self.flip_kpt29(target['pred_sigma'].clone())

        target['gt_xyz_17'] = self.flip_kpt17(target['gt_xyz_17'])
        target['gt_scale_trans'], target['pred_cam'], target['img_center'] = \
            self.flip_params(target['gt_scale_trans'].clone(), target['pred_cam'].clone(), target['img_center'].clone())

        return target

    def flip_kpt29(self, pred_xyz):
        assert pred_xyz.dim() == 3
        if pred_xyz.shape[2] != 1:
            pred_xyz[:, :, 0] = - pred_xyz[:, :, 0]

        for pair in self.joint_pairs_29:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            pred_xyz[:, idx, :] = pred_xyz[:, inv_idx, :]

        return pred_xyz

    def flip_kpt24(self, pred_xyz):
        assert pred_xyz.dim() == 3
        if pred_xyz.shape[2] != 1:
            pred_xyz[:, :, 0] = - pred_xyz[:, :, 0]

        for pair in self.joint_pairs_24:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            pred_xyz[:, idx, :] = pred_xyz[:, inv_idx, :]

        return pred_xyz

    def flip_kpt17(self, pred_xyz):
        assert pred_xyz.dim() == 3
        if pred_xyz.shape[2] != 1:
            pred_xyz[:, :, 0] = - pred_xyz[:, :, 0]

        for pair in self.joint_pairs_17:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            pred_xyz[:, idx, :] = pred_xyz[:, inv_idx, :]

        return pred_xyz

    def flip_phi23(self, pred_phi):
        assert pred_phi.dim() == 3
        pred_phi[:, :, 1] = -1 * pred_phi[:, :, 1]

        for pair in self.joint_pairs_24:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0 - 1, dim1 - 1)).long()
            inv_idx = torch.Tensor((dim1 - 1, dim0 - 1)).long()
            pred_phi[:, idx] = pred_phi[:, inv_idx]

        return pred_phi

    def flip_params(self, gt_scale_trans, pred_cam, img_center):
        gt_scale_trans[:, 1] = -1 * gt_scale_trans[:, 1]
        pred_cam[:, 1] = -1 * pred_cam[:, 1]
        img_center[:, 0] = img_center[:, 0] * (-1)

        return gt_scale_trans, pred_cam, img_center
