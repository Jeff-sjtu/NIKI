import numpy as np
from sklearn.mixture import GaussianMixture


def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert (S2.shape[1] == S1.shape[1]), (S1.shape, S2.shape)

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Error:
    S1_hat = scale * R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    if S1.ndim == 2:
        S1_hat = compute_similarity_transform(S1.copy(), S2.copy())
    else:
        S1_hat = np.zeros_like(S1)
        for i in range(S1.shape[0]):
            S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat


def reconstruction_error(S1, S2):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    return S1_hat


def world2cam(world_coord, R, T):
    cam_coord = np.dot(R, world_coord - T)
    return cam_coord


def weak_cam2pixel(cam_coord, root_z, f, c):
    x = cam_coord[:, 0] / (root_z + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (root_z + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)

    avg_f = (f[0] + f[1]) / 2
    cam_param = np.array([avg_f / (root_z + 1e-8), c[0], c[1]])
    return img_coord, cam_param


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return img_coord


def cam2pixel_temporal(cam_coords, fs, cs):
    x = cam_coords[:, :, 0] / \
        (cam_coords[:, :, 2] + 1e-8) * fs[:, [0]] + cs[:, [0]]
    y = cam_coords[:, :, 1] / \
        (cam_coords[:, :, 2] + 1e-8) * fs[:, [1]] + cs[:, [1]]
    z = cam_coords[:, :, 2]
    img_coord = np.concatenate(
        (x[:, :, None], y[:, :, None], z[:, :, None]), 2)
    return img_coord


def cam2pixel_matrix(cam_coord, intrinsic_param):
    cam_coord = cam_coord.transpose(1, 0)
    cam_homogeneous_coord = np.concatenate(
        (cam_coord, np.ones((1, cam_coord.shape[1]), dtype=np.float32)), axis=0)
    img_coord = np.dot(intrinsic_param, cam_homogeneous_coord) / \
        (cam_coord[2, :] + 1e-8)
    img_coord = np.concatenate((img_coord[:2, :], cam_coord[2:3, :]), axis=0)
    return img_coord.transpose(1, 0)


def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return cam_coord


def pixel2cam_matrix(pixel_coord, intrinsic_param):

    x = (pixel_coord[:, 0] - intrinsic_param[0][2]) / \
        intrinsic_param[0][0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - intrinsic_param[1][2]) / \
        intrinsic_param[1][1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return cam_coord


def get_intrinsic_metrix(f, c, inv=False):
    intrinsic_metrix = np.zeros((3, 3)).astype(np.float32)

    if inv:
        intrinsic_metrix[0, 0] = 1.0 / f[0]
        intrinsic_metrix[0, 2] = -c[0] / f[0]
        intrinsic_metrix[1, 1] = 1.0 / f[1]
        intrinsic_metrix[1, 2] = -c[1] / f[1]
        intrinsic_metrix[2, 2] = 1
    else:
        intrinsic_metrix[0, 0] = f[0]
        intrinsic_metrix[0, 2] = c[0]
        intrinsic_metrix[1, 1] = f[1]
        intrinsic_metrix[1, 2] = c[1]
        intrinsic_metrix[2, 2] = 1

    return intrinsic_metrix


def normalize_uv_temporal(uv, bbox, scale=1):
    c_x = bbox[:, [0]]
    c_y = bbox[:, [1]]
    w = bbox[:, [2]]
    h = bbox[:, [3]]

    new_uv = uv.copy()

    new_uv[:, :, 0] = (new_uv[:, :, 0] - c_x) / (w * scale)
    new_uv[:, :, 1] = (new_uv[:, :, 1] - c_y) / (h * scale)

    return new_uv


def calc_cam_scale_trans_refined1(xyz_29, uv_29, uvd_weight, img_center):

    # the equation to be solved:
    # u_256 / f * (1-cx/u) * (z + tz) = x + tx
    #   -> (u - cx) * (z * 1/f + tz/f) = x + tx
    #
    # v_256 / f * (1-cy/v) * (z + tz) = y + ty

    # return: tz/f, tx, ty

    weight = (uvd_weight.sum(axis=-1, keepdims=True) >= 3.0) * 1.0  # 24 x 1
    # assert weight.sum() >= 2, 'too few valid keypoints to calculate cam para'

    uv_29_fullsize = uv_29 * 256.0
    uv_c_diff = uv_29_fullsize - img_center

    if weight.sum() <= 2:
        # print('bad data')
        return np.zeros(3), 0.0, -1

    num_joints = len(uv_29)

    Ax = np.zeros((num_joints, 3))
    Ax[:, 0] = uv_c_diff[:, 0]
    Ax[:, 1] = -1

    Ay = np.zeros((num_joints, 3))
    Ay[:, 0] = uv_c_diff[:, 1]
    Ay[:, 2] = -1

    Ax = Ax * weight
    Ay = Ay * weight

    A = np.concatenate([Ax, Ay], axis=0)

    bx = (xyz_29[:, 0] - uv_c_diff[:, 0] *
          xyz_29[:, 2] / 1000.0) * weight[:, 0]
    by = (xyz_29[:, 1] - uv_c_diff[:, 1] *
          xyz_29[:, 2] / 1000.0) * weight[:, 0]
    b = np.concatenate([bx, by], axis=0)

    A_s = np.dot(A.T, A)
    b_s = np.dot(A.T, b)

    cam_para = np.linalg.solve(A_s, b_s)

    # f_estimated = 1.0 / cam_para[0]
    f_estimated = 1000.0
    tz = cam_para[0] * f_estimated
    tx, ty = cam_para[1:]

    target_camera = np.zeros(4)
    target_camera[0] = f_estimated
    target_camera[1:] = np.array([tx, ty, tz])

    backed_projected_xyz = back_projection_matrix(
        uv_29_fullsize, xyz_29, target_camera, img_center)
    diff = np.sum((backed_projected_xyz - xyz_29)**2, axis=-1) * weight[:, 0]
    diff = np.sqrt(diff).sum() / (weight.sum() + 1e-6) * \
        1000  # roughly mpjpe > 70

    out = np.zeros(3)
    out[1:] = cam_para[1:]
    out[0] = 1000.0 / 256.0 / tz

    return out, 1.0, diff


def calc_cam_scale_trans(xyz_29, uv_29, uvd_weight, f=1000.0, img_center=None):

    # the equation to be solved: 
    # u * 256 / f * (z + f/256 * 1/scale) = x + tx
    # v * 256 / f * (z + f/256 * 1/scale) = y + ty

    weight = (uvd_weight.sum(axis=-1, keepdims=True) >= 3.0) * 1.0 # 24 x 1
    # assert weight.sum() >= 2, 'too few valid keypoints to calculate cam para'

    if weight.sum() < 2:
        # print('bad data')
        return np.zeros(3), 0.0, -1

    num_joints = len(uv_29)

    Ax = np.zeros((num_joints, 3))
    Ax[:, 1] = -1
    Ax[:, 0] = uv_29[:, 0]

    Ay = np.zeros((num_joints, 3))
    Ay[:, 2] = -1
    Ay[:, 0] = uv_29[:, 1]

    Ax = Ax * weight
    Ay = Ay * weight

    A = np.concatenate([Ax, Ay], axis=0)

    bx = (xyz_29[:, 0] - 256 * uv_29[:, 0] / f * xyz_29[:, 2]) * weight[:, 0]
    by = (xyz_29[:, 1] - 256 * uv_29[:, 1] / f * xyz_29[:, 2]) * weight[:, 0]
    b = np.concatenate([bx, by], axis=0)

    A_s = np.dot(A.T, A)
    b_s = np.dot(A.T, b)

    cam_para = np.linalg.solve(A_s, b_s)

    trans = cam_para[1:]
    scale = 1.0 / cam_para[0]

    scale_trans = np.array([scale, trans[0], trans[1]])

    return scale_trans, 1.0, 0.0


def back_projection(uvd, xyz, pred_camera, focal_length=5000.):
    camScale = pred_camera[:1].reshape(1, -1)
    camTrans = pred_camera[1:].reshape(1, -1)

    camDepth = focal_length / (256 * camScale)

    pred_xyz = np.zeros_like(xyz)
    pred_xyz[:, 2] = xyz[:, 2].copy()
    pred_xyz[:, :2] = (uvd[:, :2] * 256 / focal_length) * \
        (pred_xyz[:, 2:] + camDepth) - camTrans

    return pred_xyz


def back_projection_matrix(uv, xyz, pred_camera, img_center):
    # pred_camera: f, tx, ty, tz
    f, tx, ty, tz = pred_camera
    cx, cy = img_center
    intrinsic_inv = np.array(
        [[1 / f, 0, -cx / f],
         [0, 1 / f, -cy / f],
         [0, 0, 1]]
    )

    uv_homo = np.ones((len(uv), 3))
    uv_homo[:, :2] = uv

    xyz_cam = np.matmul(uv_homo, intrinsic_inv.T)  # 29 x 3
    abs_z = xyz[:, [2]] + tz  # 29 x 1
    xyz_cam = xyz_cam * abs_z

    pred_xyz = xyz_cam - pred_camera[1:]

    return pred_xyz


def back_projection_batch(uvd, xyz, pred_camera, focal_length=1000.):
    batch_size = xyz.shape[0]
    camScale = pred_camera[:, :1].reshape(batch_size, 1, 1)
    camTrans = pred_camera[:, 1:3].reshape(batch_size, 1, 2)

    camDepth = focal_length / (256 * camScale)

    pred_xyz = np.zeros_like(xyz)
    pred_xyz[:, :, 2] = xyz[:, :, 2].copy()
    pred_xyz[:, :, :2] = (uvd[:, :, :2] * 256 / focal_length) * \
        (pred_xyz[:, :, 2:] + camDepth) - camTrans

    return pred_xyz


class Error_Score_Evaluator:
    def __init__(self):
        self.kpt_errs = []
        self.kpt_scores = []

        self.gm_mean, self.gm_cov = np.zeros(2), np.eye(2)

    def update(self, kpt_diff, kpt_score, valid_mask=None):
        kpt_err = np.sqrt((kpt_diff ** 2).sum(axis=-1))
        if valid_mask is None:
            valid_mask = np.ones_like(kpt_err)
        kpt_err, kpt_score, valid_mask = kpt_err.reshape(-1), kpt_score.reshape(-1), (valid_mask.reshape(-1) > 0.5)
        kpt_err_valid = kpt_err[valid_mask].tolist()
        kpt_score_valid = kpt_score[valid_mask].tolist()

        self.kpt_errs += kpt_err_valid
        self.kpt_scores += kpt_score_valid

    def calc_correction(self):
        kpt_errs = np.array(self.kpt_errs)
        kpt_scores = np.array(self.kpt_scores)
        # print()
        if len(kpt_errs) == 0:
            return
        mean_err = kpt_errs.mean()
        mean_score = kpt_scores.mean()

        X = np.stack([kpt_errs, kpt_scores], axis=1)

        gm = GaussianMixture(n_components=1, random_state=0).fit(X)

        gm_mean, gm_cov = gm.means_[0], gm.covariances_[0]
        self.gm_mean, self.gm_cov = gm_mean, gm_cov

        print(f'error mean: {mean_err}, score_mean: {mean_score}, {gm_mean, gm_cov}')
        '''
        PW3D: 
        array([0.107235  , 0.47090596]), 
        array([[ 0.02559005, -0.01198124],
            [-0.01198124,  0.02639371]])
        array([0.36691626, 0.3448269 ]), 
        array([[ 0.07646557, -0.04886626],
            [-0.04886626,  0.09448104]]))
        '''

    def sample(self, kpt_diff, valid_mask):
        batch_size, seq_len = kpt_diff.shape[0], kpt_diff.shape[1]
        kpt_err = np.sqrt((kpt_diff ** 2).sum(axis=-1)).reshape(-1)
        valid_mask = valid_mask.reshape(-1)
        # print(kpt_err[valid_mask].mean(), self.gm_mean, self.gm_cov)

        marginal_mean = self.gm_mean[1] + self.gm_cov[1, 0] / \
            self.gm_cov[0, 0] * (kpt_err - self.gm_mean[0])
        marginal_var = self.gm_cov[1, 1] - self.gm_cov[1,
                                                       0] / self.gm_cov[0, 0] * self.gm_cov[0, 1]

        score = np.random.rand(len(kpt_err))
        score *= np.sqrt(marginal_var)
        score += marginal_mean

        return score.reshape(batch_size, seq_len)

    def update_params(self, mean, cov):
        self.gm_mean = mean
        self.gm_cov = cov
