import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict

base_config = {
    "architectures": [
        "BertForMaskedLM"
    ],
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 64,
    "initializer_range": 0.02,
    "intermediate_size": 256,
    "max_position_embeddings": 128,
    "num_attention_heads": 8,
    "num_hidden_layers": 2,
    "type_vocab_size": 2,
    "layer_norm_eps": 1e-5,
    "output_attentions": False,
    "output_hidden_states": False,
    "num_joints": 29
}


def nets(inp_dim):
    return nn.Sequential(nn.Linear(inp_dim, 512), nn.LeakyReLU(), nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, inp_dim), nn.Tanh())


def nett(inp_dim):
    return nn.Sequential(nn.Linear(inp_dim, 512), nn.LeakyReLU(), nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, inp_dim))


class CondNet(nn.Module):
    def __init__(self, jts_dim, shape_dim, hidden_dim=512, use_act=False, use_shape=False):
        super(CondNet, self).__init__()

        self.inp_dim = jts_dim + shape_dim
        self.out_dim = jts_dim

        self.jts_dim = jts_dim
        self.shape_dim = shape_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.inp_dim, hidden_dim),
            # nn.Dropout(),
            nn.Identity(),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.Dropout(),
            nn.Identity(),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, self.out_dim))

        self.use_act = use_act
        if use_act:
            self.act = nn.Tanh()

    def forward(self, jts, shape, *args):

        inp_feat = torch.cat((jts, shape), dim=1)

        out_feat = self.encoder(inp_feat)
        jts_feat = out_feat[:, :self.jts_dim]

        if self.use_act:
            jts_feat = self.act(jts_feat)

        return jts_feat


class ShapeCondRealNVP(nn.Module):
    def __init__(self, opt, jts_tot_dim, shape_tot_dim, num_stack, use_shape=True):
        super(ShapeCondRealNVP, self).__init__()

        config = edict(base_config)

        config.seq_len = 2
        config.temporal_len = 2

        config.divided_attention = False
        self.config = config

        self.jts_tot_dim = jts_tot_dim

        # self.shape_proj = nn.Linear(shape_tot_dim, 64)
        shape_tot_dim2 = shape_tot_dim
        self.shape_proj = nn.Identity()
        # shape_tot_dim2 = 32
        # self.shape_proj = nn.Sequential(
        #     nn.Linear(shape_tot_dim, shape_tot_dim2),
        #     nn.Dropout())
        self.shape_tot_dim = shape_tot_dim2

        self.num_stack = num_stack

        # === jts net ===
        # N x jts_tot_dim
        jts_masks = torch.from_numpy(np.array([
            [0, 1] * (jts_tot_dim // 2),
            [1, 0] * (jts_tot_dim // 2)
        ] * (num_stack // 2)).astype(np.float32))
        # jts_masks = torch.from_numpy(np.array([
        #     [0, 1] * (jts_tot_dim // 2),
        #     [1, 0] * (jts_tot_dim // 2),
        #     [0] * (jts_tot_dim // 2) + [1] * (jts_tot_dim // 2),
        #     [1] * (jts_tot_dim // 2) + [0] * (jts_tot_dim // 2),
        # ] * (num_stack // 2)).astype(np.float32))

        shape_mask = torch.from_numpy(np.array([
            0, 0, 0, 0
        ] * (num_stack // 4)).astype(np.float32))

        # jts_masks = (torch.rand_like(jts_masks) > 0.5).float()
        # shape_mask = torch.ones_like(shape_mask)

        self.register_buffer('jts_masks', jts_masks)
        self.register_buffer('shape_masks', shape_mask)

        self.jts_nett = torch.nn.ModuleList([
            CondNet(
                jts_dim=jts_tot_dim,
                shape_dim=self.shape_tot_dim,
                use_shape=use_shape,
                use_act=False)
            for _ in range(self.num_stack)])
        self.jts_nets = torch.nn.ModuleList([
            CondNet(
                jts_dim=jts_tot_dim,
                shape_dim=self.shape_tot_dim,
                use_shape=use_shape,
                use_act=True)
            for _ in range(self.num_stack)])
        # self.jts_nets = torch.nn.ModuleList([CondNet(jts_tot_dim, self.shape_tot_dim, use_act=True) for _ in range(self.num_stack)])

        # self.jts_nett = torch.nn.ModuleList([nett(jts_tot_dim) for _ in range(self.num_stack)])
        # self.jts_nets = torch.nn.ModuleList([nets(jts_tot_dim) for _ in range(self.num_stack)])

    def _init(self):
        for m in self.t:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)
        for m in self.s:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)

    def inverse_p(self, z, shape, cond=None):
        assert z.dim() == 2 and shape.dim() == 2
        # batch_size = z.shape[0]

        x = z
        shape_cond = self.shape_proj(shape)

        log_det_J = z.new_zeros(z.shape[0])
        for i in range(len(self.jts_nett)):
            x_ = x * self.jts_masks[i]
            shape_cond_ = self.shape_masks[i] * shape_cond
            s = self.jts_nets[i](x_, shape_cond_) * (1 - self.jts_masks[i])
            t = self.jts_nett[i](x_, shape_cond_) * (1 - self.jts_masks[i])
            x = x_ + (1 - self.jts_masks[i]) * (x * torch.exp(s) + t)
            log_det_J += s.sum(dim=1)

        return x, None, log_det_J

    def forward_p(self, x, shape, cond=None):
        assert x.dim() == 2 and shape.dim() == 2
        # batch_size = x.shape[0]

        log_det_J, z = x.new_zeros(x.shape[0]), x
        shape_cond = self.shape_proj(shape)

        for i in reversed(range(len(self.jts_nett))):
            # update jts only
            z_ = self.jts_masks[i] * z
            shape_cond_ = self.shape_masks[i] * shape_cond
            s = self.jts_nets[i](z_, shape_cond_) * (1 - self.jts_masks[i])
            t = self.jts_nett[i](z_, shape_cond_) * (1 - self.jts_masks[i])
            z = (1 - self.jts_masks[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)

        # z = z / self.area
        # log_det_j -= torch.log(self.area)
        return z, None, log_det_J

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
