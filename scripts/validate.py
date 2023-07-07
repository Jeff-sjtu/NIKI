import argparse
import logging

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix
from tqdm import tqdm

from niki.datasets.naive_dataset_temporal import naive_dataset_temporal
from niki.models.layers.smpl.SMPL import SMPL_layer
# from torch.nn.utils import clip_grad
# from niki.models.flowIK import FlowIK
from niki.models.NIKITS import NIKITS
from niki.utils.config import update_config
from niki.utils.eval_utils import calc_accel_error

parser = argparse.ArgumentParser(description='PyTorch Pose Estimation Validate')
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
parser.add_argument('--ckpt',
                    help='checkpoint file name',
                    required=True,
                    type=str)

opt = parser.parse_args()
ckpt = opt.ckpt
cfg_name = opt.cfg
opt = update_config(opt.cfg)
opt.ckpt = ckpt
opt.cfg = cfg_name

streamhandler = logging.StreamHandler()

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(streamhandler)

h36m_jregressor = np.load('./model_files/J_regressor_h36m.npy')
smpl_layer = SMPL_layer(
    './model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
    h36m_jregressor=h36m_jregressor,
    dtype=torch.float32
).cuda()


def main():
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')

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

    valid_dataset_pred_pw3d = naive_dataset_temporal(opt.DATASET.valid_paths['3dpw_xocc'], '', dataset_name='pw3d', train=False, usage='xyz', occlusion=data_occ, seq_len=opt.seq_len)
    valid_dataset_pred_h36m_noocc = naive_dataset_temporal(opt.DATASET.valid_paths['h36m'], '', dataset_name='h36m', train=False, usage='xyz', occlusion=False, seq_len=opt.seq_len)
    valid_dataset_pred_pw3d_noocc = naive_dataset_temporal(opt.DATASET.valid_paths['3dpw'], '', dataset_name='pw3d', train=False, usage='xyz', occlusion=False, seq_len=opt.seq_len)

    valid_loader_pred_pw3d = torch.utils.data.DataLoader(
        valid_dataset_pred_pw3d, batch_size=32, shuffle=False,
        num_workers=4, drop_last=False, persistent_workers=True)

    valid_loader_pred_h36m_noocc = torch.utils.data.DataLoader(
        valid_dataset_pred_h36m_noocc, batch_size=32, shuffle=False,
        num_workers=4, drop_last=False, persistent_workers=True)

    valid_loader_pred_pw3d_noocc = torch.utils.data.DataLoader(
        valid_dataset_pred_pw3d_noocc, batch_size=32, shuffle=False,
        num_workers=4, drop_last=False, persistent_workers=True)

    if opt.model_name == 'NIKI':
        pass
        # model = FlowIK(opt).cuda()
    elif opt.model_name == 'NIKITS':
        model = NIKITS(opt).cuda()

    print('Load CKPT', opt.ckpt)

    tmp_dict = torch.load(opt.ckpt)
    new_tmp_dict = {}
    for k, v in tmp_dict.items():
        if k in model.state_dict() and model.state_dict()[k].shape == v.shape:
            new_tmp_dict[k] = v
        else:
            print(k)

    model.load_state_dict(
        new_tmp_dict, strict=False
    )

    with torch.no_grad():

        print('====== Evaluate 3DPW XOCC ======')
        valid(valid_dataset_pred_pw3d, valid_loader_pred_pw3d, logger, model=model, name='3dpw-xocc')

        print('====== Evaluate Human3.6M ======')
        valid(valid_dataset_pred_h36m_noocc, valid_loader_pred_h36m_noocc, logger, model=model, name='h36m')

        print('====== Evaluate 3DPW ======')
        valid(valid_dataset_pred_pw3d_noocc, valid_loader_pred_pw3d_noocc, logger, model=model, name='3dpw')


def valid(valid_dataset, valid_loader, logger, model, name=''):

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

    diff_verts = []

    for item in tqdm(valid_loader, dynamic_ncols=True):
        # get input
        inp = {
            'pred_xyz_29': item['pred_xyz_29'],  # batch x time_seq x 29 x 3
            'pred_uv': item['pred_uv_29'],
            'pred_sigma': item['pred_sigma'],
            'pred_xyz_24_struct': item['pred_xyz_24_struct'],
            'pred_beta': item['pred_betas'],
            'pred_phi': item['pred_phi'],
            'pred_cam': item['pred_cam']
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
        pred_phi = output['pred_phi'].detach().cuda()
        pred_cam = output['pred_cam'].detach()

        if 'pred_xyz_17' in output.keys():
            pred_xyz_17 = output.pred_xyz_17
            pred_xyz29_struct = output.pred_xyz_29_struct.float().cpu().numpy()
        else:
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
            pred_uv = project_2d(pred_xyz_29.reshape(batch_size * time_seq, 29, 3), pred_cam.reshape(batch_size * time_seq, 3))
            # pred_uv = project_2d(output.joints.float().reshape(batch_size * time_seq, 29, 3), pred_cam.reshape(batch_size * time_seq, 3))
            pred_uv_struct = project_2d(output.pred_xyz_29_struct.float().reshape(batch_size * time_seq, 29, 3), pred_cam.reshape(batch_size * time_seq, 3))

            target_uv = item['gt_uv_29'].reshape(batch_size * time_seq, 29, 2)[:, :24]

            pred_uv_24s.append(pred_uv[:, :24].float().cpu().numpy())
            pred_uv_24s_struct.append(pred_uv_struct[:, :24].float().cpu().numpy())

            target_uv_24s.append(target_uv.numpy())
            target_uv_24s_weights.append(
                item['uv_29_weight'].reshape(batch_size * time_seq, 29, 2).numpy()[:, :24]
            )

        pred_vert = output.verts.float().reshape(batch_size * time_seq, -1, 3).cpu().numpy()

        gt_rot_aa = item['gt_theta'].reshape(batch_size * time_seq, 24, 3).cuda()
        gt_rot_mat = axis_angle_to_matrix(gt_rot_aa).reshape(batch_size * time_seq, 24, 9)
        gt_output = smpl_layer(
            pose_axis_angle=gt_rot_mat,
            betas=item['gt_betas'].cuda().reshape(batch_size * time_seq, 10),
            global_orient=None,
            pose2rot=False,
        )
        gt_vert = gt_output['vertices'].cpu().numpy()

        diff_vert = np.sqrt(np.sum((pred_vert - gt_vert) ** 2, axis=2)).mean(axis=1)
        pred_xyz_17s.append(pred_xyz_17.reshape(batch_size * time_seq, 17, 3).float().cpu().numpy())
        pred_xyz_29s.append(pred_xyz_29.reshape(batch_size * time_seq, 29, 3).float().cpu().numpy())
        pred_xyz_29s_struct.append(pred_xyz29_struct.reshape(batch_size * time_seq, 29, 3))

        target_xyz_17s.append(item['gt_xyz_17'].reshape(batch_size * time_seq, 17, 3).numpy())
        target_xyz_29s.append(item['gt_xyz_29'].reshape(batch_size * time_seq, 29, 3).numpy())

        diff_verts.append(diff_vert)

        # print(cam_err, beta_err)

    diff_verts = np.concatenate(diff_verts, axis=0)

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
        uv_diff = np.absolute(pred_uv_24s - target_uv_24s) * target_uv_24s_weights
        uv_diff = uv_diff.sum() / (target_uv_24s_weights.sum() + 1e-5)
        uv_diff_struct = np.absolute(
            pred_uv_24s_struct - target_uv_24s) * target_uv_24s_weights
        uv_diff_struct = uv_diff_struct.sum() / (target_uv_24s_weights.sum() + 1e-5)
    else:
        uv_diff, uv_diff_struct = -1, -1

    accel, accel_err = calc_accel_error(pred_xyz_17s.copy(), target_xyz_17s.copy())
    print('accel:', accel, '| accel_error:', accel_err, '| uv_diff:', uv_diff, '| uv_diff_struct:', uv_diff_struct)
    pve = np.mean(diff_verts) * 1000
    print('pve:', pve)

    pa_17, mp_17 = valid_dataset.evaluate_xyz_17(pred_xyz_17s, target_xyz_17s)
    pa_29, mp_29 = valid_dataset.evaluate_xyz_29(pred_xyz_29s.copy(), target_xyz_29s)
    pa_29_2, mp_29_2 = valid_dataset.evaluate_xyz_29(pred_xyz_29s, pred_xyz_29s_struct)
    logger.info(f'Pred xyz14, mpjpe={mp_17}, pa={pa_17}, accel_error={accel_err}, uv_diff={uv_diff}; \nPred xyz29 mpjpe={mp_29}, pa={pa_29}\nPVE={pve}\n')


def project_2d(pred_joints, pred_camera, focal_length=1000.0):
    assert pred_joints.dim() == 3
    assert pred_camera.dim() == 2

    # pred_joints: [B, 29, 3]
    pred_joints = pred_joints - pred_joints[:, [1, 2], :].mean(dim=1, keepdim=True)
    camDepth = focal_length / (256.0 * pred_camera[:, [0]] + 1e-9)  # batch x 1
    transl = torch.cat([pred_camera[:, 1:], camDepth], dim=1)
    pred_joints_cam = pred_joints + transl.unsqueeze(1)

    pred_keypoints_2d = pred_joints_cam[:, :, :2] * focal_length / (256.0 * pred_joints_cam[:, :, [2]])
    return pred_keypoints_2d


if __name__ == "__main__":
    main()
