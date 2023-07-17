import cv2
import torch
import numpy as np


def project_2d(pred_joints, pred_camera, focal_length=1000.0):
    # pred_joints: [B, 29, 3]
    pred_joints = pred_joints - pred_joints[:, [1, 2], :].mean(dim=1, keepdim=True)
    camDepth = focal_length / (256.0 * pred_camera[:, [0]] + 1e-9)  # batch x 1
    transl = torch.cat([pred_camera[:, 1:], camDepth], dim=1)
    pred_joints_cam = pred_joints + transl.unsqueeze(1)

    pred_keypoints_2d = pred_joints_cam[:, :, :2] / pred_joints_cam[:, :, [2]] * focal_length / 256.0
    return pred_keypoints_2d


def find_circumscribed_bbox(old_bbox):
    # xywh: [B, T, 4]
    bs, seq_len = old_bbox.shape[:2]
    # find circumscribed bbox
    cx, cy = old_bbox[:, :, [0]], old_bbox[:, :, [1]]
    w, h = old_bbox[:, :, [2]], old_bbox[:, :, [3]]

    assert torch.sum(torch.abs(w - h)) < 1e-5

    left = cx - w * 0.5
    up = cy - h * 0.5
    right = cx + w * 0.5
    down = cy + h * 0.5

    left, _ = torch.min(left, dim=1, keepdim=True)
    up, _ = torch.min(up, dim=1, keepdim=True)
    right, _ = torch.max(right, dim=1, keepdim=True)
    down, _ = torch.max(down, dim=1, keepdim=True)

    cx = (left + right) * 0.5
    cy = (up + down) * 0.5
    w = right - left
    h = down - up

    w = torch.maximum(w, h)

    bbox = torch.cat((cx, cy, w, w), dim=2).expand(bs, seq_len, 4)
    return bbox


def update_cam(uv, uv_weight, xyz):
    # uv: [B, T, K, 2]
    # xyz: [B, T, K, 3]
    assert uv.dim() == 4
    assert uv_weight.dim() == 4, uv_weight.shape
    bs, seq_len = uv.shape[:2]
    uv = uv[:, :, :24, :].reshape(bs * seq_len, 24, 2)
    if uv_weight.shape[-1] == 1:
        uv_weight = uv_weight[:, :, :24, :].reshape(bs * seq_len, 24, 1)
    else:
        uv_weight = uv_weight[:, :, :24, :].reshape(bs * seq_len, 24, 2)

    xyz = xyz[:, :, :24, :].reshape(bs * seq_len, 24, 3)

    Ax = torch.zeros((bs * seq_len, 24, 3), device=uv.device, dtype=uv.dtype)
    Ay = torch.zeros((bs * seq_len, 24, 3), device=uv.device, dtype=uv.dtype)

    Ax[:, :, 0] = uv[:, :, 0]
    Ax[:, :, 1] = -1
    Ay[:, :, 0] = uv[:, :, 1]
    Ay[:, :, 2] = -1

    Ax = Ax * uv_weight[:, :, [0]]
    Ay = Ay * uv_weight[:, :, [0]]

    # [B * T, 2K, 3]
    A = torch.cat((Ax, Ay), dim=1)

    bx = (xyz[:, :, 0] - 256 * uv[:, :, 0] / 1000 * xyz[:, :, 2]) * uv_weight[:, :, 0]
    by = (xyz[:, :, 1] - 256 * uv[:, :, 1] / 1000 * xyz[:, :, 2]) * uv_weight[:, :, 0]

    # [B * T, 2K, 1]
    b = torch.cat((bx, by), dim=1)[:, :, None]
    # [B * T, 3, 3]
    ATA = A.transpose(1, 2).bmm(A)
    mask_zero = ATA.sum(dim=(1, 2))

    ATA_non_zero = ATA[mask_zero != 0].reshape(-1, 3, 3)
    A_non_zero = A[mask_zero != 0].reshape(-1, 48, 3)
    b_non_zero = b[mask_zero != 0].reshape(-1, 48, 1)

    new_cam_non_zero = torch.inverse(ATA_non_zero).bmm(A_non_zero.transpose(1, 2)).bmm(b_non_zero)

    new_cam = torch.zeros(bs * seq_len, 3, 1, device=uv.device)
    new_cam[mask_zero != 0] = new_cam_non_zero

    new_cam_weight = (mask_zero != 0).float()
    new_cam_weight = new_cam_weight.reshape(bs, seq_len, 1)

    new_cam = new_cam.reshape(bs, seq_len, 3)
    norm_z = new_cam[:, :, [0]]
    norm_z[new_cam_weight > 0] = 1 / norm_z[new_cam_weight > 0]
    norm_z[new_cam_weight == 0] = 0.9
    new_cam[:, :, [0]] = norm_z

    return new_cam, new_cam_weight


def update_cam2(old_bbox, new_bbox, old_cam):
    # xyz: [B, T, K, 3]
    # bbox: [B, T, 4]
    c_old = old_bbox[:, :, :2]
    c_new = new_bbox[:, :, :2]
    w_old = old_bbox[:, :, [-1]]
    w_new = new_bbox[:, :, [-1]]

    s_old = old_cam[:, :, [0]]
    t_old = old_cam[:, :, 1:]

    s_new = s_old * w_old / w_new
    t_new = t_old + (c_old - c_new) / (s_old * w_old)

    new_cam = torch.cat((s_new, t_new), dim=2)
    # print(t_new[0], t_old[0], new_cam[0], '===')
    return new_cam


def update_uv(old_uv, old_bbox, new_bbox):
    # uvd: [B, T, K, 3]
    bs, seq_len, K = old_uv.shape[:3]

    old_wh = old_bbox[:, :, 2:].unsqueeze(2).expand(bs, seq_len, K, 2)
    old_cxcy = old_bbox[:, :, :2].unsqueeze(2).expand(bs, seq_len, K, 2)

    new_wh = new_bbox[:, :, 2:].unsqueeze(2).expand(bs, seq_len, K, 2)
    new_cxcy = new_bbox[:, :, :2].unsqueeze(2).expand(bs, seq_len, K, 2)

    global_uv = old_uv * old_wh + old_cxcy
    new_uv = (global_uv - new_cxcy) / new_wh

    return new_uv


def reproject_uv(targets):

    change_gt = ('gt_xyz_29' in targets)
    # uv, cam, bbox
    old_bbox = targets['bbox'].clone()
    # img_center = targets['img_center'].clone()

    old_pred_uv = targets['pred_uv_29'].clone()

    pred_sigma = targets['pred_sigma']
    pred_uv_weight = (1 - pred_sigma * 5).clamp_min(0)

    old_pred_xyz = targets['pred_xyz_29']

    if change_gt:
        old_gt_uv = targets['gt_uv_29'].clone()

        gt_uv_weight = targets['uv_29_weight'].clone()
        xyz29_weight = targets['xyz_29_weight'].clone()
        xyz29_weight = xyz29_weight[:, :, :, [0]] * xyz29_weight[:, :, :, [1]] * xyz29_weight[:, :, :, [2]]

        old_gt_xyz = targets['gt_xyz_29']

        gt_cam_mask = targets['gt_scale_trans'][:, :, [3]].clone()

    new_bbox = find_circumscribed_bbox(old_bbox)
    new_pred_uv = update_uv(old_pred_uv, old_bbox, new_bbox)
    new_img_center = (targets['img_sizes'] * 0.5 - new_bbox[:, :, :2])  / new_bbox[:, :, 2:] * 256.0
    new_pred_cam, _ = update_cam(new_pred_uv, torch.ones_like(pred_uv_weight), old_pred_xyz)

    targets['pred_cam'] = new_pred_cam
    targets['pred_cam_0center'] = new_pred_cam

    targets['pred_uv_29'] = new_pred_uv

    targets['bbox'] = new_bbox
    targets['old_bbox'] = old_bbox
    targets['img_center'] = new_img_center

    if change_gt:
        new_gt_uv = update_uv(old_gt_uv, old_bbox, new_bbox)

        gt_uv_weight[:, :, :, 0] = gt_uv_weight[:, :, :, 0] * xyz29_weight[:, :, :, 0]
        gt_uv_weight[:, :, :, 1] = gt_uv_weight[:, :, :, 1] * xyz29_weight[:, :, :, 0]

        new_gt_cam, gt_cam_weight = update_cam(new_gt_uv, gt_uv_weight, old_gt_xyz)

        gt_cam_weight = gt_cam_weight * gt_cam_mask
        new_gt_cam = torch.cat((new_gt_cam, gt_cam_weight), dim=2)

        targets['gt_scale_trans'] = new_gt_cam
        targets['gt_uv_29'] = new_gt_uv

    return targets



def center_scale_to_box(center, scale):
    pixel_std = 1.0
    w = scale[0] * pixel_std
    h = scale[1] * pixel_std
    xmin = center[0] - w * 0.5
    ymin = center[1] - h * 0.5
    xmax = xmin + w
    ymax = ymin + h
    bbox = [xmin, ymin, xmax, ymax]
    return bbox


def xyxy_to_center_scale_batch(bbox_xyxy):
    assert len(bbox_xyxy.shape) == 2
    new_bbox = bbox_xyxy.copy()
    new_bbox[:, 0] = (bbox_xyxy[:, 0] + bbox_xyxy[:, 2]) * 0.5
    new_bbox[:, 1] = (bbox_xyxy[:, 1] + bbox_xyxy[:, 3]) * 0.5

    new_bbox[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
    new_bbox[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]
    return new_bbox



def get_one_box(det_output, thrd=0.9):
    max_area = 0
    max_bbox = None

    if det_output['boxes'].shape[0] == 0 or thrd < 1e-5:
        return None

    for i in range(det_output['boxes'].shape[0]):
        bbox = det_output['boxes'][i]
        score = det_output['scores'][i]
        if float(score) < thrd:
            continue
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if float(area) > max_area:
            max_bbox = [float(x) for x in bbox]
            max_area = area

    if max_bbox is None:
        return get_one_box(det_output, thrd=thrd - 0.1)

    return max_bbox


def get_max_iou_box(det_output, prev_bbox, thrd=0.9):
    max_score = 0
    max_bbox = None
    for i in range(det_output['boxes'].shape[0]):
        bbox = det_output['boxes'][i]
        score = det_output['scores'][i]
        # if float(score) < thrd:
        #     continue
        # area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        iou = calc_iou(prev_bbox, bbox)
        iou_score = float(score) * iou
        if float(iou_score) > max_score:
            max_bbox = [float(x) for x in bbox]
            max_score = iou_score
    if max_bbox is None:
        max_bbox = prev_bbox

    return max_bbox


def calc_iou(bbox1, bbox2):
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    box1Area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    box2Area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    
    iou = interArea / float(box1Area + box2Area - interArea)

    return iou


def split_indices(total_len, seq_len, redo_last=True):
    indices = []
    for s in range(0, total_len - seq_len + 1, seq_len):
        indices.append([s+i for i in range(seq_len)])
    
    if (total_len - seq_len) % seq_len != 0:
        left_num = (total_len - seq_len) % seq_len
        left_start = total_len // seq_len
        last_index = [left_start*seq_len + i for i in range(left_num)]

        if redo_last:
            last_index += [total_len - 1] * (seq_len - left_num)

        indices.append(last_index)
    
    return indices