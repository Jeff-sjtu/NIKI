import math
import random

import cv2
import numpy as np
import torch

from ..bbox import _box_to_center_scale, _center_scale_to_box
from ..transforms import (addDPG, affine_transform, flip_joints_3d, flip_thetas, flip_xyz_joints_3d,
                          get_affine_transform, im_to_torch, batch_rodrigues_numpy, flip_twist,
                          rotmat_to_quat_numpy, rotate_xyz_jts, rot_aa, flip_cam_xyz_joints_3d)

smpl_parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
                16, 17, 18, 19, 20, 21]


skeleton_29 = [ 
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), # 5
    (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), # 11
    (9, 13), (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), # 17
    (17, 19), (18, 20), (19, 21), (20, 22), (21, 23), (15, 24), # 23
    (22, 25), (23, 26), (10, 27), (11, 28) # 27
]



class SimpleTransform3DSMPLCam(object):
    """Generation of cropped input person, pose coords, smpl parameters.

    Parameters
    ----------
    img: torch.Tensor
        A tensor with shape: `(3, h, w)`.
    label: dict
        A dictionary with 4 keys:
            `bbox`: [xmin, ymin, xmax, ymax]
            `joints_3d`: numpy.ndarray with shape: (n_joints, 2),
                    including position and visible flag
            `width`: image width
            `height`: image height
    dataset:
        The dataset to be transformed, must include `joint_pairs` property for flipping.
    scale_factor: int
        Scale augmentation.
    input_size: tuple
        Input image size, as (height, width).
    output_size: tuple
        Heatmap size, as (height, width).
    rot: int
        Ratation augmentation.
    train: bool
        True for training trasformation.
    """

    def __init__(self, dataset, scale_factor, color_factor, occlusion, add_dpg,
                 input_size, output_size, depth_dim, bbox_3d_shape,
                 rot, sigma, train, loss_type='MSELoss', scale_mult=1.25, focal_length=1000, two_d=False,
                 root_idx=0, get_paf=False, is_vibe=False):
        if two_d:
            self._joint_pairs = dataset.joint_pairs
        else:
            self._joint_pairs_17 = dataset.joint_pairs_17
            self._joint_pairs_24 = dataset.joint_pairs_24
            self._joint_pairs_29 = dataset.joint_pairs_29

        self._scale_factor = scale_factor
        self._color_factor = color_factor
        self._occlusion = occlusion
        self._rot = rot
        self._add_dpg = add_dpg

        self._input_size = input_size
        self._heatmap_size = output_size

        self._sigma = sigma
        self._train = train
        self._loss_type = loss_type
        self._aspect_ratio = float(input_size[1]) / input_size[0]  # w / h
        self._feat_stride = np.array(input_size) / np.array(output_size)

        self.pixel_std = 1

        self.bbox_3d_shape = dataset.bbox_3d_shape
        self._scale_mult = scale_mult
        # self.kinematic = dataset.kinematic
        self.two_d = two_d

        # convert to unit: meter
        self.depth_factor2meter = self.bbox_3d_shape[2] if self.bbox_3d_shape[2] < 500 else self.bbox_3d_shape[2]*1e-3

        self.focal_length = focal_length
        self.root_idx = root_idx

        self.get_paf = get_paf
        self.is_vibe = is_vibe

        if train:
            self.num_joints_half_body = dataset.num_joints_half_body
            self.prob_half_body = dataset.prob_half_body

            self.upper_body_ids = dataset.upper_body_ids
            self.lower_body_ids = dataset.lower_body_ids

    def test_transform(self, src, bbox, occlusion=None, input_size=None, direct_return_img=False):
        xmin, ymin, xmax, ymax = bbox
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio, scale_mult=self._scale_mult)
        scale = scale * 1.0

        if input_size is None:
            input_size = self._input_size
        else:
            input_size = (input_size, input_size)
        
        inp_h, inp_w = input_size
        trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)

        if occlusion is not None:
            synth_xmin, synth_ymin, synth_xmax, synth_ymax = occlusion
            synth_w = synth_xmax - synth_xmin
            synth_h = synth_ymax - synth_ymin

            synth_xmin, synth_ymin = max(0, int(synth_xmin)), max(0, int(synth_ymin))
            synth_ymax = min(int(synth_ymin + synth_h), int(inp_h))
            synth_xmax = min(int(synth_xmin + synth_w), int(inp_w))
            img[synth_ymin:synth_ymax, synth_xmin:synth_xmax, :] = np.random.rand(synth_ymax-synth_ymin, synth_xmax-synth_xmin, 3) * 255

            
        bbox = _center_scale_to_box(center, scale)

        if direct_return_img:
            return img, bbox, 0

        img = im_to_torch(img)

        if self.is_vibe:
            img[0].add_(-0.485)
            img[1].add_(-0.456)
            img[2].add_(-0.406)

            # std
            img[0].div_(0.229)
            img[1].div_(0.224)
            img[2].div_(0.225)
        else:
            img[0].add_(-0.406)
            img[1].add_(-0.457)
            img[2].add_(-0.480)

            # std
            img[0].div_(0.225)
            img[1].div_(0.224)
            img[2].div_(0.229)

        img_center = np.array([float(src.shape[1]) * 0.5, float(src.shape[0]) * 0.5])

        return img, bbox, img_center

    def __call__(self, src, label):
        raise NotImplementedError
        return


def _box_to_center_scale_nosquare(x, y, w, h, aspect_ratio=1.0, scale_mult=1.5):
    """Convert box coordinates to center and scale.
    adapted from https://github.com/Microsoft/human-pose-estimation.pytorch
    """
    pixel_std = 1
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * scale_mult
    return center, scale
