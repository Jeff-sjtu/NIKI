import argparse
import logging
import os
import random

import numpy as np
import torch
from niki.datasets.naive_dataset_temporal import (
    mix_temporal_dataset_full_wamass, mix_temporal_dataset_full_woamass,
    mix_temporal_dataset_full_wocc, mix_temporal_dataset_wamass,
    naive_dataset_temporal,
    mix_temporal_dataset_full_wcoco)

from niki.models.criterionTS import LossTS

# from torch.nn.utils import clip_grad
from niki.models.NIKITS import NIKITS

from niki.models.layers.smpl.SMPL import SMPL_layer
from niki.utils.config import update_config
from niki.utils.eval_utils import calc_accel_error
from niki.utils.metrics import DataLogger
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description='PyTorch Pose Estimation Validate')
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    required=False,
                    default='',
                    type=str)
parser.add_argument('--model_name',
                    help='experiment configure file name',
                    required=False,
                    default='bert_ST_small',
                    type=str)
parser.add_argument('--exp-id',
                    help='experiment configure file name',
                    required=False,
                    default='0',
                    type=str)
parser.add_argument('--dataset',
                    help='experiment configure file name',
                    required=False,
                    default='w_hp3d_wo_amass',
                    type=str)
parser.add_argument('--pretrained',
                    help='experiment configure file name',
                    required=False,
                    default='',
                    type=str)
parser.add_argument('--start_epoch',
                    help='experiment configure file name',
                    required=False,
                    default=0,
                    type=int)
parser.add_argument('--seed', default=123123, type=int,
                    help='random seed')

opt = parser.parse_args()
exp_id = opt.exp_id
seed = opt.seed
pretrained = opt.pretrained
if len(opt.cfg) > 0:
    cfg_name = opt.cfg
    opt = update_config(opt.cfg)
    if exp_id != '0':
        opt.exp_id = exp_id
    if pretrained != '':
        opt.pretrained = pretrained
    opt.seed = seed
    opt.cfg = cfg_name

opt.exp_id = opt.cfg.split('/')[-1] + '-' + opt.exp_id
opt.refined_pose_shape = True
print(opt)

if not os.path.exists('./exp/kpt_smpl_unocc/{}/{}'.format(opt.model_name, opt.exp_id)):
    os.makedirs('./exp/kpt_smpl_unocc/{}/{}'.format(opt.model_name, opt.exp_id))

filehandler = logging.FileHandler(
    './exp/kpt_smpl_unocc/{}/{}/training.log'.format(opt.model_name, opt.exp_id))
streamhandler = logging.StreamHandler()

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)


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


h36m_jregressor = np.load('./model_files/J_regressor_h36m.npy')
smpl_layer = SMPL_layer(
    './model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
    h36m_jregressor=h36m_jregressor,
    dtype=torch.float32
).cuda()


def _init_fn(worker_id):
    np.random.seed(opt.seed)
    random.seed(opt.seed)


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    # torch.backends.cudnn.benchmark = True
    setup_seed(opt.seed)
    torch.multiprocessing.set_sharing_strategy('file_system')

    none_paths = ['', '', '']

    data_occ = opt.DATASET.occlusion if 'occlusion' in opt.DATASET else False
    print('occlusion', data_occ)

    use_imgfeat = opt.IMG_FEAT if 'IMG_FEAT' in opt else True
    print('use_imgfeat', use_imgfeat)

    use_flip = ('feature_flipped' in opt.DATASET.train_paths[0])
    print('use_flip', use_flip)

    gendered_smpl = ('GENDERED' in opt.DATASET)
    print('gendered_smpl', gendered_smpl)
    assert not gendered_smpl, 'Not Implemented'

    img_feat_size = 2048 if 'hrnet' in opt.DATASET.train_paths[0] else 1024
    print(img_feat_size)

    if gendered_smpl:
        pass
        # train_dataset = mix_temporal_dataset_gendered_wocc(opt.DATASET.train_paths, none_paths, train=True, usage='xyz', occlusion=data_occ)
    elif opt.DATASET.name == 'wo_hp3d_w_amass':
        train_dataset = mix_temporal_dataset_wamass(
            opt.DATASET.train_paths, none_paths, opt=opt, train=True, usage='xyz', occlusion=data_occ, img_feat_size=img_feat_size)
    elif opt.DATASET.name == 'w_hp3d':
        train_dataset = mix_temporal_dataset_full_woamass(
            opt.DATASET.train_paths, none_paths, opt=opt, train=True, usage='xyz', occlusion=data_occ, img_feat_size=img_feat_size)
    elif opt.DATASET.name == 'w_coco':
        train_dataset = mix_temporal_dataset_full_wcoco(
            opt.DATASET.train_paths, none_paths, opt=opt, train=True, usage='xyz', occlusion=data_occ, img_feat_size=img_feat_size)
    elif opt.DATASET.name == 'wocc':
        train_dataset = mix_temporal_dataset_full_wocc(
            opt.DATASET.train_paths, none_paths, opt=opt, train=True, usage='xyz', occlusion=data_occ, wrong_flip_aug=True, img_feat_size=img_feat_size)
    else:
        print(opt.dataset)
        raise NotImplementedError

    valid_dataset_pred_h36m = naive_dataset_temporal(
        opt.DATASET.valid_paths['h36m_xocc'], '', dataset_name='h36m', train=False, usage='xyz', occlusion=data_occ, seq_len=opt.seq_len)
    if gendered_smpl:
        pass
    elif opt.DATASET.name == 'w_3doh':
        valid_dataset_pred_pw3d = naive_dataset_temporal(
            opt.DATASET.valid_paths['3dpw_xocc'], '', dataset_name='pw3d', train=False, usage='xyz', occlusion=False, seq_len=opt.seq_len)
        valid_dataset_pred_3doh = naive_dataset_temporal(
            opt.DATASET.valid_paths['3doh'], '', dataset_name='3doh', train=False, usage='xyz', occlusion=False, seq_len=opt.seq_len)

        valid_loader_pred_3doh = torch.utils.data.DataLoader(
            valid_dataset_pred_3doh, batch_size=32, shuffle=False,
            num_workers=4, drop_last=False, persistent_workers=True)
    else:
        valid_dataset_pred_pw3d = naive_dataset_temporal(
            opt.DATASET.valid_paths['3dpw_xocc'], '', dataset_name='pw3d', train=False, usage='xyz', occlusion=data_occ, seq_len=opt.seq_len)

    valid_dataset_pred_h36m_noocc = naive_dataset_temporal(
        opt.DATASET.valid_paths['h36m'], '', dataset_name='h36m', train=False, usage='xyz', occlusion=False, seq_len=opt.seq_len)
    valid_dataset_pred_pw3d_noocc = naive_dataset_temporal(
        opt.DATASET.valid_paths['3dpw'], '', dataset_name='pw3d', train=False, usage='xyz', occlusion=False, seq_len=opt.seq_len)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.train_batch, shuffle=True,
        num_workers=opt.num_workers, drop_last=False, persistent_workers=True,
        worker_init_fn=_init_fn)

    valid_loader_pred_h36m = torch.utils.data.DataLoader(
        valid_dataset_pred_h36m, batch_size=32, shuffle=False,
        num_workers=4, drop_last=False, persistent_workers=True)

    valid_loader_pred_pw3d = torch.utils.data.DataLoader(
        valid_dataset_pred_pw3d, batch_size=32, shuffle=False,
        num_workers=4, drop_last=False, persistent_workers=True)

    valid_loader_pred_h36m_noocc = torch.utils.data.DataLoader(
        valid_dataset_pred_h36m_noocc, batch_size=32, shuffle=False,
        num_workers=4, drop_last=False, persistent_workers=True)

    valid_loader_pred_pw3d_noocc = torch.utils.data.DataLoader(
        valid_dataset_pred_pw3d_noocc, batch_size=32, shuffle=False,
        num_workers=4, drop_last=False, persistent_workers=True)

    lr = 1e-4

    if opt.model_name == 'NIKI':
        pass
        # model = FlowIK(opt).cuda()
    elif opt.model_name == 'NIKITS':
        model = NIKITS(opt).cuda()

    if len(opt.pretrained) > 0:
        print('Use pretrained model', opt.pretrained)

        tmp_dict = torch.load(opt.pretrained)
        new_tmp_dict = {}
        for k, v in tmp_dict.items():
            if k in model.state_dict() and model.state_dict()[k].shape == v.shape:
                new_tmp_dict[k] = v
            else:
                print(k)

        model.load_state_dict(
            new_tmp_dict, strict=False
        )

    if opt.criterion_name == 'LossTS':
        criterion = LossTS(smpl_layer).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    logger.info(
        f'\n\nModel name: {opt.model_name}, exp_name: {opt.exp_id}, learning rate: {lr}')
    logger.info(opt)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[40 - opt.start_epoch, 50 - opt.start_epoch], gamma=0.1)

    print("start training...")
    for i in range(opt.start_epoch, 60):
        print('epoch', i)
        avg_loss, summary_str = train(
            train_loader, optimizer, criterion, model=model, epoch=i)
        logger.info(f'epoch {i}, loss: {avg_loss}, {summary_str}')
        lr_scheduler.step()

        # if (i+1) % 5 == 0:
        if i % 2 == 0:
            with torch.no_grad():
                if opt.DATASET.name == 'w_3doh':
                    print('====== Evaluate 3DOH ======')
                    valid(valid_dataset_pred_3doh, valid_loader_pred_3doh,
                          logger, model=model, optimizer=optimizer)
                    # valid_gt(valid_dataset_pred_pw3d, valid_loader_pred_pw3d, logger, model=model)
                    print('====== Evaluate 3DPW ======')
                    valid(valid_dataset_pred_pw3d, valid_loader_pred_pw3d,
                          logger, model=model, optimizer=optimizer)
                    # valid_gt(valid_dataset_pred_pw3d, valid_loader_pred_pw3d, logger, model=model)
                else:
                    print('====== Evaluate Human3.6M ======')
                    valid(valid_dataset_pred_h36m, valid_loader_pred_h36m,
                          logger, model=model, optimizer=optimizer)

                    print('====== Evaluate 3DPW ======')
                    valid(valid_dataset_pred_pw3d, valid_loader_pred_pw3d,
                          logger, model=model, optimizer=optimizer)
                    # valid_gt(valid_dataset_pred_pw3d, valid_loader_pred_pw3d, logger, model=model)

                    print('====== Evaluate Human3.6M No OCC ======')
                    valid(valid_dataset_pred_h36m_noocc, valid_loader_pred_h36m_noocc,
                          logger, model=model, optimizer=optimizer)

                    print('====== Evaluate 3DPW No OCC ======')
                    valid(valid_dataset_pred_pw3d_noocc, valid_loader_pred_pw3d_noocc,
                          logger, model=model, optimizer=optimizer)

            saved_path2 = f'./exp/kpt_smpl_unocc/{opt.model_name}/{opt.exp_id}/model_{i}.pth'
            torch.save(model.state_dict(), saved_path2)


def train(train_loader, optimizer, criterion, model, epoch=0):
    torch.set_grad_enabled(True)
    tot_loss = 0
    loss_loggers = {}
    iter_num = len(train_loader)

    model.train()

    smpl_layer.train()
    train_loader = tqdm(train_loader, dynamic_ncols=True)

    for i, item in enumerate(train_loader):
        for k, _ in item.items():
            if isinstance(item[k], torch.Tensor):
                item[k] = item[k].cuda()
        # item = reproject_uv(item)

        # get input
        inp = {
            'pred_xyz_29': item['pred_xyz_29'],  # batch x time_seq x 29 x 3
            'pred_uv': item['pred_uv_29'],
            'pred_sigma': item['pred_sigma'],
            'pred_xyz_24_struct': item['pred_xyz_24_struct'],
            'pred_beta': item['pred_betas'],
            'pred_phi': item['pred_phi'],
            'pred_cam': item['pred_cam'],
            'img_feat': item['img_feat']
        }

        batch_size = item['pred_xyz_29'].shape[0]

        optimizer.zero_grad()
        out = model(inp=inp, epoch=epoch, target=item)

        loss, loss_dict = criterion(out, item, epoch=epoch)

        optimizer.zero_grad()
        loss.backward()
        # for group in optimizer.param_groups:
        #     for param in group["params"]:
        #         clip_grad.clip_grad_norm_(param, 10)
        optimizer.step()

        tot_loss += loss.detach().cpu().numpy()

        summary_str = 'loss: {loss:.2f}'.format(loss=tot_loss / (i + 1))
        for k, v in loss_dict.items():
            if k not in loss_loggers.keys():
                loss_loggers[k] = DataLogger()

            loss_loggers[k].update(v, batch_size)
            summary_str += f' | {k}: {loss_loggers[k].avg:.3f}'

        train_loader.set_description(summary_str)

    return tot_loss / iter_num, summary_str


def valid(valid_dataset, valid_loader, logger, model, optimizer=None):

    smpl_layer.eval()
    model.eval()

    # smpl_layer.train()

    pred_xyz_17s = []
    pred_xyz_29s = []
    pred_uv_24s = []
    pred_uv_24s_struct = []
    pred_xyz_29s_struct = []

    target_xyz_17s = []
    target_xyz_29s = []
    target_uv_24s = []
    target_uv_24s_weights = []

    for item in tqdm(valid_loader, dynamic_ncols=True):
        # item = reproject_uv(item)

        # get input
        inp = {
            'pred_xyz_29': item['pred_xyz_29'],  # batch x time_seq x 29 x 3
            'pred_uv': item['pred_uv_29'],
            'pred_sigma': item['pred_sigma'],
            'pred_xyz_24_struct': item['pred_xyz_24_struct'],
            'pred_beta': item['pred_betas'],
            'pred_phi': item['pred_phi'],
            'pred_cam': item['pred_cam'],
            'img_feat': item['img_feat']
        }
        for k, _ in inp.items():
            if isinstance(inp[k], torch.Tensor):
                inp[k] = inp[k].cuda()

        batch_size, time_seq = item['pred_xyz_29'].shape[:2]
        # optimizer.zero_grad()
        with torch.no_grad():
            output = model(inp=inp)

        torch.set_grad_enabled(False)

        pred_xyz_29 = output['pred_xyz_29']
        pred_beta = output['pred_beta'].detach()
        # pred_beta = item['betas'].cuda()
        # pred_phi = item['phi'].cuda()
        pred_cam = output['pred_cam'].detach()

        # gt_cam_scale = item['gt_cam_scale'].reshape(-1, 4).cuda()
        # cam_err = torch.abs(gt_cam_scale[:, 0] - para_out['pred_cam'][:, 0]).mean()
        # gt_beta = item['gt_betas'].reshape(-1, 10).cuda()
        # beta_err = torch.abs(gt_beta - pred_beta).mean(dim=0)

        if 'pred_xyz_17' in output.keys():
            pred_xyz_17 = output.pred_xyz_17
            pred_xyz29_struct = output.pred_xyz_29_struct.float().cpu().numpy()
        else:
            pred_phi = output['pred_phi'].detach().cuda()
            output = smpl_layer.hybrik(
                pose_skeleton=pred_xyz_29.reshape(-1, 29, 3),
                betas=pred_beta.reshape(-1, 10),
                phis=pred_phi.reshape(-1, 23, 2),
                global_orient=None,
                return_verts=False,
                return_29_jts=True
            )

            pred_xyz_17 = output.joints_from_verts
            pred_xyz29_struct = output.joints.float().cpu().numpy()

        if 'gt_uv_29' in item.keys():
            pred_uv = project_2d(pred_xyz_29.reshape(
                batch_size * time_seq, 29, 3), pred_cam.reshape(batch_size * time_seq, 3))
            pred_uv_struct = project_2d(output.pred_xyz_29_struct.float().reshape(
                batch_size * time_seq, 29, 3), pred_cam.reshape(batch_size * time_seq, 3))
            target_uv = item['gt_uv_29'].reshape(
                batch_size * time_seq, 29, 2)[:, :24]

            pred_uv_24s.append(pred_uv[:, :24].float().cpu().numpy())
            pred_uv_24s_struct.append(
                pred_uv_struct[:, :24].float().cpu().numpy())

            target_uv_24s.append(target_uv.numpy())
            target_uv_24s_weights.append(
                item['uv_29_weight'].reshape(
                    batch_size * time_seq, 29, 2).numpy()[:, :24]
            )

        pred_xyz_17s.append(pred_xyz_17.reshape(
            batch_size * time_seq, 17, 3).float().cpu().numpy())
        pred_xyz_29s.append(pred_xyz_29.reshape(
            batch_size * time_seq, 29, 3).float().cpu().numpy())

        pred_xyz_29s_struct.append(
            pred_xyz29_struct.reshape(batch_size * time_seq, 29, 3))

        target_xyz_17s.append(item['gt_xyz_17'].reshape(
            batch_size * time_seq, 17, 3).numpy())
        target_xyz_29s.append(item['gt_xyz_29'].reshape(
            batch_size * time_seq, 29, 3).numpy())

        # print(cam_err, beta_err)

    pred_xyz_17s = np.concatenate(pred_xyz_17s, axis=0)
    pred_xyz_29s = np.concatenate(pred_xyz_29s, axis=0)
    target_xyz_17s = np.concatenate(target_xyz_17s, axis=0)
    target_xyz_29s = np.concatenate(target_xyz_29s, axis=0)
    pred_xyz_29s_struct = np.concatenate(pred_xyz_29s_struct, axis=0)

    if 'gt_uv_29' in item.keys():
        pred_uv_24s = np.concatenate(pred_uv_24s, axis=0)
        pred_uv_24s_struct = np.concatenate(pred_uv_24s_struct, axis=0)
        target_uv_24s = np.concatenate(target_uv_24s, axis=0)
        target_uv_24s_weights = np.concatenate(target_uv_24s_weights, axis=0)
        uv_diff = np.absolute(pred_uv_24s - target_uv_24s) * \
            target_uv_24s_weights
        uv_diff = uv_diff.sum() / (target_uv_24s_weights.sum() + 1e-5)
        uv_diff_struct = np.absolute(
            pred_uv_24s_struct - target_uv_24s) * target_uv_24s_weights
        uv_diff_struct = uv_diff_struct.sum() / (target_uv_24s_weights.sum() + 1e-5)
    else:
        uv_diff, uv_diff_struct = -1, -1
    accel, accel_err = calc_accel_error(
        pred_xyz_17s.copy(), target_xyz_17s.copy())
    print('accel:', accel, '| accel_error:', accel_err,
          '| uv_diff:', uv_diff, '| uv_diff_struct:', uv_diff_struct)

    pa_17, mp_17 = valid_dataset.evaluate_xyz_17(pred_xyz_17s, target_xyz_17s)
    pa_29, mp_29 = valid_dataset.evaluate_xyz_29(
        pred_xyz_29s.copy(), target_xyz_29s)
    pa_29_2, mp_29_2 = valid_dataset.evaluate_xyz_29(
        pred_xyz_29s, pred_xyz_29s_struct)
    logger.info(
        f'Pred xyz14, mpjpe={mp_17}, pa={pa_17}, accel_error={accel_err}, uv_diff={uv_diff}; \nPred xyz29 mpjpe={mp_29}, pa={pa_29}\n')


def project_2d(pred_joints, pred_camera, focal_length=1000.0):
    # pred_joints: [B, 29, 3]
    pred_joints = pred_joints - \
        pred_joints[:, [1, 2], :].mean(dim=1, keepdim=True)
    camDepth = focal_length / (256.0 * pred_camera[:, [0]] + 1e-9)  # batch x 1
    transl = torch.cat([pred_camera[:, 1:], camDepth], dim=1)
    pred_joints_cam = pred_joints + transl.unsqueeze(1)

    pred_keypoints_2d = pred_joints_cam[:, :, :2] / \
        pred_joints_cam[:, :, [2]] * focal_length / 256.0
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
    assert uv_weight.dim() == 4
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

    bx = (xyz[:, :, 0] - 256 * uv[:, :, 0] /
          1000 * xyz[:, :, 2]) * uv_weight[:, :, 0]
    by = (xyz[:, :, 1] - 256 * uv[:, :, 1] /
          1000 * xyz[:, :, 2]) * uv_weight[:, :, 0]

    # [B * T, 2K, 1]
    b = torch.cat((bx, by), dim=1)[:, :, None]
    # [B * T, 3, 3]
    ATA = A.transpose(1, 2).bmm(A)
    mask_zero = ATA.sum(dim=(1, 2))

    ATA_non_zero = ATA[mask_zero != 0].reshape(-1, 3, 3)
    A_non_zero = A[mask_zero != 0].reshape(-1, 48, 3)
    b_non_zero = b[mask_zero != 0].reshape(-1, 48, 1)

    new_cam_non_zero = torch.inverse(ATA_non_zero).bmm(
        A_non_zero.transpose(1, 2)).bmm(b_non_zero)

    new_cam = torch.zeros(bs * seq_len, 3, 1, device=uv.device)
    new_cam[mask_zero != 0] = new_cam_non_zero

    new_cam_weight = (mask_zero != 0).float()
    new_cam_weight = new_cam_weight.reshape(bs, seq_len, 1)

    new_cam = new_cam.reshape(bs, seq_len, 3)
    norm_z = new_cam[:, :, [0]]
    norm_z[new_cam_weight > 0] = 1 / norm_z[new_cam_weight > 0]
    norm_z[new_cam_weight == 0] = 0
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


if __name__ == "__main__":
    main()
