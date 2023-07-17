"""Video demo script."""
import argparse
import os

import cv2
import joblib
import numpy as np
import torch
from easydict import EasyDict as edict
from niki.utils.hybrik_utils import builder
from niki.utils.config import update_config
from niki.utils.hybrik_utils.simple_transform_3d_smpl_cam import SimpleTransform3DSMPLCam
from niki.utils.render_pytorch3d import render_mesh
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm
from niki.utils.demo_utils import *
from niki.models.NIKI_1stage import FlowIK_camnet
from niki.utils.demo_utils import *

det_transform = T.Compose([T.ToTensor()])


def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


def get_video_info(in_file):
    stream = cv2.VideoCapture(in_file)
    assert stream.isOpened(), 'Cannot capture source'
    # self.path = input_source
    datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))
    fps = stream.get(cv2.CAP_PROP_FPS)
    frameSize = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # bitrate = int(stream.get(cv2.CAP_PROP_BITRATE))
    videoinfo = {'fourcc': fourcc, 'fps': fps, 'frameSize': frameSize}
    stream.release()

    return stream, videoinfo, datalen


def recognize_video_ext(ext=''):
    if ext == 'mp4':
        return cv2.VideoWriter_fourcc(*'mp4v'), '.' + ext
    elif ext == 'avi':
        return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
    elif ext == 'mov':
        return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
    else:
        print("Unknow video format {}, will use .mp4 instead of it".format(ext))
        return cv2.VideoWriter_fourcc(*'mp4v'), '.mp4'


parser = argparse.ArgumentParser(description='HybrIK Demo')

parser.add_argument('--video-name',
                    help='video name',
                    default='',
                    type=str,
                    required=True)
parser.add_argument('--out-dir',
                    help='output folder',
                    default='',
                    type=str,
                    required=True)
parser.add_argument('--gpu',
                    help='gpu',
                    default=0,
                    type=int)
parser.add_argument('--flip-test',
                    default=False,
                    dest='flip_test',
                    help='flip test',
                    action='store_true')
parser.add_argument('--save-pt', default=False, dest='save_pt',
                    help='save prediction', action='store_true')
parser.add_argument('--not-vis', default=False, dest='not_vis',
                    help='do not visualize', action='store_true')
parser.add_argument('--hybrik_cam', default=True,
                    help='use camera parameter predict by HybrIK, often gets better results under non-occlusion videos if set to be True', 
                    type=bool)
opt = parser.parse_args()


cfg_file = 'configs/hybrik_config.yaml'
CKPT = 'exp/checkpoint_49_cocoeft.pth'
cfg = update_config(cfg_file)

v_cfg_file = 'configs/NIKI-1stage.yaml'
V_CKPT = 'exp/niki_model_28.pth'
v_cfg = update_config(v_cfg_file)


bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2200, 2200, 2200))
bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
dummpy_set = edict({
    'joint_pairs_17': None,
    'joint_pairs_24': None,
    'joint_pairs_29': None,
    'bbox_3d_shape': bbox_3d_shape
})

res_keys = [
    'pred_uvd',
    'pred_xyz_29',
    'pred_scores',
    'pred_sigma',
    'f',
    'pred_betas',
    'pred_phi',
    'scale_mult',
    'pred_cam_root',
    'bbox',
    'height',
    'width',
    'img_path',
    'img_sizes'
]
res_db = {k: [] for k in res_keys}

transformation = SimpleTransform3DSMPLCam(
    dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
    color_factor=cfg.DATASET.COLOR_FACTOR,
    occlusion=cfg.DATASET.OCCLUSION,
    input_size=cfg.MODEL.IMAGE_SIZE,
    output_size=cfg.MODEL.HEATMAP_SIZE,
    depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
    bbox_3d_shape=bbox_3d_shape,
    rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
    train=False, add_dpg=False,
    loss_type=cfg.LOSS['TYPE'])


det_model = fasterrcnn_resnet50_fpn(pretrained=True)
hybrik_model = builder.build_sppe(cfg.MODEL)
flow_model = FlowIK_camnet(v_cfg)

print(f'Loading model from {CKPT}...')
save_dict = torch.load(CKPT, map_location='cpu')
if type(save_dict) == dict:
    model_dict = save_dict['model']
    hybrik_model.load_state_dict(model_dict)
else:
    hybrik_model.load_state_dict(save_dict)

print(f'Loding LGD model from {V_CKPT}')
save_dict = torch.load(V_CKPT, map_location='cpu')
flow_model.load_state_dict(save_dict, strict=False)

camnet_dict = 'exp/niki_model_28.pth'
tmp_dict = torch.load(camnet_dict)
new_tmp_dict = {}
for k, v in tmp_dict.items():
    if 'regressor.camnet' in k:
        new_k = k[len('regressor.camnet.'):]
        new_tmp_dict[new_k] = v

flow_model.regressor.camnet.load_state_dict(new_tmp_dict)

det_model.cuda(opt.gpu)
hybrik_model.cuda(opt.gpu)
flow_model.cuda(opt.gpu)
det_model.eval()
hybrik_model.eval()
flow_model.eval()

if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir)
if not os.path.exists(os.path.join(opt.out_dir, 'raw_images')):
    os.makedirs(os.path.join(opt.out_dir, 'raw_images'))
if not os.path.exists(os.path.join(opt.out_dir, 'res_images')):
    os.makedirs(os.path.join(opt.out_dir, 'res_images'))
if not os.path.exists(os.path.join(opt.out_dir, 'res_2d_images')):
    os.makedirs(os.path.join(opt.out_dir, 'res_2d_images'))

_, info, _ = get_video_info(opt.video_name)
video_basename = os.path.basename(opt.video_name).split('.')[0]

savepath = f'./{opt.out_dir}/res_{video_basename}.mp4'
savepath2d = f'./{opt.out_dir}/res_2d_{video_basename}.mp4'
info['savepath'] = savepath
info['savepath2d'] = savepath2d

write_stream = cv2.VideoWriter(
    *[info[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
# write2d_stream = cv2.VideoWriter(
#     *[info[k] for k in ['savepath2d', 'fourcc', 'fps', 'frameSize']])
if not write_stream.isOpened():
    print("Try to use other video encoders...")
    ext = info['savepath'].split('.')[-1]
    fourcc, _ext = recognize_video_ext(ext)
    info['fourcc'] = fourcc
    info['savepath'] = info['savepath'][:-4] + _ext
    info['savepath2d'] = info['savepath2d'][:-4] + _ext
    write_stream = cv2.VideoWriter(
        *[info[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
    # write2d_stream = cv2.VideoWriter(
    #     *[info[k] for k in ['savepath2d', 'fourcc', 'fps', 'frameSize']])

assert write_stream.isOpened(), 'Cannot open video for writing'
# assert write2d_stream.isOpened(), 'Cannot open video for writing'

os.system(f'ffmpeg -i {opt.video_name} {opt.out_dir}/raw_images/{video_basename}-%06d.png')

files = os.listdir(f'{opt.out_dir}/raw_images')
files.sort()

img_path_list = []

for file in tqdm(files):
    if not os.path.isdir(file) and file[-4:] in ['.jpg', '.png']:

        img_path = os.path.join(opt.out_dir, 'raw_images', file)
        img_path_list.append(img_path)

prev_box = None
renderer = None
smpl_faces = torch.from_numpy(hybrik_model.smpl.faces.astype(np.int32))

idx = 0
print('### Run Model...')
for img_path in tqdm(img_path_list, dynamic_ncols=True):
    dirname = os.path.dirname(img_path)
    basename = os.path.basename(img_path)

    with torch.no_grad():
        # Run Detection
        input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        det_input = det_transform(input_image).to(opt.gpu)
        det_output = det_model([det_input])[0]

        if prev_box is None:
            tight_bbox = get_one_box(det_output)  # xyxy
            if tight_bbox is None:
                continue
        else:
            tight_bbox = get_max_iou_box(det_output, prev_box)  # xyxy

            area = (tight_bbox[2] - tight_bbox[0]) * (tight_bbox[3] - tight_bbox[1])

            max_bbox = get_one_box(det_output)  # xyxy
            if max_bbox is not None:
                max_area = (max_bbox[2] - max_bbox[0]) * (max_bbox[3] - max_bbox[1])
                if area < max_area * 0.1:
                    tight_bbox = max_bbox

        prev_box = tight_bbox

        # Run HybrIK
        # bbox: [x1, y1, x2, y2]
        pose_input, bbox, img_center = transformation.test_transform(
            input_image, tight_bbox)
        pose_input = pose_input.to(opt.gpu)[None, :, :, :]
        pose_output = hybrik_model(
            pose_input, flip_test=opt.flip_test,
            bboxes=torch.from_numpy(np.array(bbox)).to(pose_input.device).unsqueeze(0).float(),
            img_center=torch.from_numpy(img_center).to(pose_input.device).unsqueeze(0).float(),
            do_hybrik=False
        )

        # === Save PT ===
        assert pose_input.shape[0] == 1, 'Only support single batch inference for now'

        pred_uvd_jts = pose_output.pred_uvd_jts.reshape(
            -1, 3).cpu().data.numpy()
        pred_xyz_jts_29 = pose_output.pred_xyz_jts_29.reshape(
            -1, 3).cpu().data.numpy()
        pred_scores = pose_output.maxvals.cpu(
            ).data[:, :29].reshape(29).numpy()
        pred_betas = pose_output.pred_shape.squeeze(
            dim=0).cpu().data.numpy()
        pred_phi = pose_output.pred_phi.squeeze(dim=0).cpu().data.numpy()
        pred_cam_root = pose_output.cam_root.squeeze(dim=0).cpu().numpy()
        pred_sigma = pose_output.sigma.cpu().data.numpy()

        img_size = np.array((input_image.shape[1], input_image.shape[0]))

        res_db['pred_uvd'].append(pred_uvd_jts)
        res_db['pred_xyz_29'].append(pred_xyz_jts_29)
        res_db['pred_scores'].append(pred_scores)
        res_db['pred_sigma'].append(pred_sigma)
        res_db['f'].append(1000.0)
        res_db['pred_betas'].append(pred_betas)
        res_db['pred_phi'].append(pred_phi)
        res_db['pred_cam_root'].append(pred_cam_root)
        res_db['bbox'].append(np.array(bbox))
        res_db['height'].append(img_size[1])
        res_db['width'].append(img_size[0])
        res_db['img_path'].append(img_path)
        res_db['img_sizes'].append(img_size)


total_img = len(res_db['img_path'])

for k in res_db.keys():
    try:
        v = np.stack(res_db[k], axis=0)
    except Exception:
        v = res_db[k]
        print(k, ' failed')

    res_db[k] = v

joblib.dump(res_db, 'vis_tmp.pt')

res_db = joblib.load('vis_tmp.pt')
total_img = len(res_db['img_path'])
print('total_img', total_img)
seq_len = 16

video_res_db = {}
total_img = (total_img // seq_len) * seq_len
video_res_db['transl'] = torch.zeros((total_img, 3))
video_res_db['vertices'] = torch.zeros((total_img, 6890, 3))
video_res_db['img_path'] = res_db['img_path']
video_res_db['bbox'] = torch.zeros((total_img, 4))
video_res_db['pred_uv'] = torch.zeros((total_img, 29, 2))
res_count = torch.zeros((total_img, 1))

mean_beta = res_db['pred_betas'].mean(axis=0)
res_db['pred_betas'][:] = mean_beta

update_bbox = v_cfg.get('update_bbox', False)
USE_HYBRIK_CAM = opt.hybrik_cam

idx = 0
for i in tqdm(range(0, total_img - seq_len + 1, seq_len), dynamic_ncols=True):
    pred_xyz_29 = res_db['pred_xyz_29'][i:i + seq_len, :, :] * 2.2
    pred_uv = res_db['pred_uvd'][i:i + seq_len, :, :2]
    pred_sigma = res_db['pred_sigma'][i:i + seq_len, :, :].squeeze(1)
    pred_beta = res_db['pred_betas'][i:i + seq_len, :]
    pred_phi = res_db['pred_phi'][i:i + seq_len, :]
    pred_cam_root = res_db['pred_cam_root'][i:i + seq_len, :]
    pred_cam = np.concatenate((
        1000.0 / (256 * pred_cam_root[:, [2]] + 1e-9),
        pred_cam_root[:, :2]
    ), axis=1)
    bbox = res_db['bbox'][i:i + seq_len, :] # xyxy

    pred_xyz_29 = pred_xyz_29 - pred_xyz_29[:, [1, 2], :].mean(axis=1, keepdims=True)

    bbox_cs = xyxy_to_center_scale_batch(bbox)
    inp = {
        'pred_xyz_29': pred_xyz_29,
        'pred_uv': pred_uv,
        'pred_sigma': pred_sigma,
        'pred_beta': pred_beta,
        'pred_phi': pred_phi,
        'pred_cam': pred_cam,
        'bbox': bbox_cs,
        'img_sizes': res_db['img_sizes'][i:i + seq_len, :]
    }

    for k in inp.keys():
        inp[k] = torch.from_numpy(inp[k]).float().cuda().unsqueeze(0)

    if update_bbox:
        inp = reproject_uv(inp)
    else:
        img_center = (inp['img_sizes'] * 0.5 - inp['bbox'][:, :, :2])  / inp['bbox'][:, :, 2:] * 256.0
        inp['img_center'] = img_center

    with torch.no_grad():
        output = flow_model.forward_getcam(inp=inp)

        video_res_db['vertices'][i:i + seq_len] = output.verts.cpu()[0]
        video_res_db['bbox'][i:i + seq_len] = inp['bbox'][0].cpu()
        video_res_db['transl'][i:i + seq_len] = output.transl.cpu()[0]
        video_res_db['pred_uv'][i:i + seq_len] = output.inv_pred2uv.cpu()[0]
        if USE_HYBRIK_CAM:
            video_res_db['transl'][i:i + seq_len] = torch.from_numpy(res_db['pred_cam_root'][i:i + seq_len])


# rendering
for i in tqdm(range(total_img), dynamic_ncols=True):
    img_path = video_res_db['img_path'][i]
    input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    bbox = video_res_db['bbox'][i]
    vertices = video_res_db['vertices'][[i]]
    transl = video_res_db['transl'][[i]]

    # Visualization
    image = input_image.copy()
    bbox_xywh = bbox
    focal = 1000.0
    focal = focal / 256 * bbox_xywh[2]

    verts_batch = vertices.to(opt.gpu)
    transl_batch = transl.to(opt.gpu)

    color_batch = render_mesh(
        vertices=verts_batch, faces=smpl_faces,
        translation=transl_batch,
        focal_length=focal, height=image.shape[0], width=image.shape[1])

    valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
    image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
    image_vis_batch = (image_vis_batch * 255).cpu().numpy()

    color = image_vis_batch[0]
    valid_mask = valid_mask_batch[0].cpu().numpy()
    input_img = image
    alpha = 0.9
    image_vis = alpha * color[:, :, :3] * valid_mask + (
        1 - alpha) * input_img * valid_mask + (1 - valid_mask) * input_img

    image_vis = image_vis.astype(np.uint8)
    image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)

    x1, y1, x2, y2 = center_scale_to_box(bbox[:2], bbox[2:])
    image_vis = cv2.rectangle(image_vis, (int(x1), int(y1)), (int(x2), int(y2)), (154, 201, 219), 5)

    idx += 1
    res_path = os.path.join(
        opt.out_dir, 'res_images', f'image-{idx:06d}.jpg')
    cv2.imwrite(res_path, image_vis)
    write_stream.write(image_vis)


res_db_path = os.path.join(f'./{opt.out_dir}/', f'{video_basename}.pt')
joblib.dump(res_db, res_db_path)
print('Prediction is saved in:', res_db_path)


write_stream.release()
# write2d_stream.release()