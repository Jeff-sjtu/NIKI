import torch
import torch.nn as nn
# import torch.nn.functional as F


class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask):
        super(RealNVP, self).__init__()

        self.register_buffer('mask', mask)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(mask))])

    def _init(self):
        for m in self.t:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)
        for m in self.s:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)

    def inverse_p(self, z, cond=None):
        x = z
        log_det_J = z.new_zeros(z.shape[0])
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
            log_det_J += s.sum(dim=1)
        return x, log_det_J

    def forward_p(self, x, cond=None):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        # z = z / self.area
        # log_det_j -= torch.log(self.area)
        return z, log_det_J

    def log_prob(self, x):
        DEVICE = x.device
        if self.prior.loc.device != DEVICE:
            self.prior.loc = self.prior.loc.to(DEVICE)
            self.prior.scale_tril = self.prior.scale_tril.to(DEVICE)
            self.prior._unbroadcasted_scale_tril = self.prior._unbroadcasted_scale_tril.to(
                DEVICE)
            self.prior.covariance_matrix = self.prior.covariance_matrix.to(
                DEVICE)
            self.prior.precision_matrix = self.prior.precision_matrix.to(
                DEVICE)

        z, logp = self.backward_p(x)
        return self.prior.log_prob(z) + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        # logp = self.prior.log_prob(z)
        x = self.forward_p(z)
        return x

    def forward(self, x):
        # DEVICE = x.device
        # px = torch.arange(-5, 5, 0.5, device=DEVICE)
        # py = torch.arange(-5, 5, 0.5, device=DEVICE)
        # xx, yy = torch.meshgrid(px, py)
        # samples = torch.stack((xx, yy), dim=2).reshape(-1, 2) + x[:1, :]
        # prop_q = (0.25 * torch.exp(-torch.abs(samples) / 2))
        # prop_g = torch.exp(self.log_prob(samples))
        # prop_q = prop_q[:, 0] * prop_q[:, 1]
        # prop = (prop_g * prop_q).sum(0).reshape(1, 1, 1) * 0.01 * 0.01

        # return self.log_prob(x) + prop
        return self.log_prob(x)


class Invertible1x1Conv(nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """

    def __init__(self, C):
        super(Invertible1x1Conv, self).__init__()

        # Sample a random orthonormal matrix to initialize weights
        W_inverse = torch.eye(2)

        # Ensure determinant is 1.0 not -1.0
        self.W_inverse = nn.Parameter(W_inverse)

    def forward_p(self, z):
        BATCH = z.shape[0]
        W = self.W_inverse.float().inverse()

        # Forward computation
        log_det_W = torch.logdet(W).expand(BATCH)
        z = torch.matmul(z, W)
        return z, log_det_W

    def backward_p(self, x):
        BATCH = x.shape[0]

        x = torch.matmul(x, self.W_inverse)
        log_det_inv_W = torch.logdet(self.W_inverse).expand(BATCH)
        return x, log_det_inv_W


class WaveGlow(nn.Module):
    def __init__(self, nets, nett, mask, prior):
        super(WaveGlow, self).__init__()

        self.prior = prior
        self.register_buffer('mask', mask)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(mask))])
        self.w = torch.nn.ModuleList(
            [Invertible1x1Conv(2) for _ in range(len(mask))])

    def _init(self):
        for m in self.t:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)
        for m in self.s:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)

    def forward_p(self, z):
        x = z
        for i in range(len(self.t)):
            x, _ = self.w[i].forward_p(x)
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def backward_p(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
            z, log_det_w = self.w[i].backward_p(z)
            log_det_J -= log_det_w
        # z = z / self.area
        # log_det_j -= torch.log(self.area)
        return z, log_det_J

    def log_prob(self, x):
        DEVICE = x.device
        if self.prior.loc.device != DEVICE:
            self.prior.loc = self.prior.loc.to(DEVICE)
            self.prior.scale_tril = self.prior.scale_tril.to(DEVICE)
            self.prior._unbroadcasted_scale_tril = self.prior._unbroadcasted_scale_tril.to(
                DEVICE)
            self.prior.covariance_matrix = self.prior.covariance_matrix.to(
                DEVICE)
            self.prior.precision_matrix = self.prior.precision_matrix.to(
                DEVICE)

        z, logp = self.backward_p(x)
        return self.prior.log_prob(z) + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        # logp = self.prior.log_prob(z)
        x = self.forward_p(z)
        return x
