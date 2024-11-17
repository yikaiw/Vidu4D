# Modified by Gengshan Yang (2023)
# Borrowed from https://raw.githubusercontent.com/JiahuiLei/CaDeX/master/core/net_bank/nvp_v2/models/nvp_v2_5.py

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from lab4d.utils.quat_transform import quaternion_to_matrix


class CouplingLayer(nn.Module):
    def __init__(self, map_s, map_t, projection, mask):
        super().__init__()
        self.map_s = map_s
        self.map_t = map_t
        self.projection = projection
        self.register_buffer("mask", mask)  # 1,1,1,3

    def forward(self, F, y):
        y1 = y * self.mask

        F_y1 = torch.cat([F, self.projection(y1)], dim=-1)
        s = self.map_s(F_y1)
        t = self.map_t(F_y1)

        x = y1 + (1 - self.mask) * ((y - t) * torch.exp(-s))
        ldj = (-s).sum(-1)

        return x, ldj

    def inverse(self, F, x):
        x1 = x * self.mask

        F_x1 = torch.cat([F, self.projection(x1)], dim=-1)
        s = self.map_s(F_x1)
        t = self.map_t(F_x1)

        y = x1 + (1 - self.mask) * (x * torch.exp(s) + t)
        ldj = s.sum(-1)

        return y, ldj


class MLP(nn.Module):
    def __init__(self, c_in, c_out, c_hiddens, act=nn.LeakyReLU, bn=nn.BatchNorm1d):
        super().__init__()
        layers = []
        d_in = c_in
        for d_out in c_hiddens:
            layers.append(nn.Linear(d_in, d_out))
            if bn is not None:
                layers.append(bn(d_out))
            layers.append(act())
            d_in = d_out
        layers.append(nn.Linear(d_in, c_out))
        self.mlp = nn.Sequential(*layers)
        self.c_out = c_out

    def forward(self, x):
        y = self.mlp(x)
        return y


def quaternions_to_rotation_matrices(quaternions):
    """
    Arguments:
    ---------
        quaternions: Tensor with size ...x4, where ... denotes any shape of
                     quaternions to be translated to rotation matrices
    Returns:
    -------
        rotation_matrices: Tensor with size ...x3x3, that contains the computed
                           rotation matrices
    """
    # Allocate memory for a Tensor of size ...x3x3 that will hold the rotation
    # matrix along the x-axis
    shape = quaternions.shape[:-1] + (3, 3)
    R = quaternions.new_zeros(shape)

    # A unit quaternion is q = w + xi + yj + zk
    xx = quaternions[..., 1] ** 2
    yy = quaternions[..., 2] ** 2
    zz = quaternions[..., 3] ** 2
    ww = quaternions[..., 0] ** 2
    n = (ww + xx + yy + zz).unsqueeze(-1)
    s = torch.zeros_like(n)
    s[n != 0] = 2 / n[n != 0]

    xy = s[..., 0] * quaternions[..., 1] * quaternions[..., 2]
    xz = s[..., 0] * quaternions[..., 1] * quaternions[..., 3]
    yz = s[..., 0] * quaternions[..., 2] * quaternions[..., 3]
    xw = s[..., 0] * quaternions[..., 1] * quaternions[..., 0]
    yw = s[..., 0] * quaternions[..., 2] * quaternions[..., 0]
    zw = s[..., 0] * quaternions[..., 3] * quaternions[..., 0]

    xx = s[..., 0] * xx
    yy = s[..., 0] * yy
    zz = s[..., 0] * zz

    R[..., 0, 0] = 1 - yy - zz
    R[..., 0, 1] = xy - zw
    R[..., 0, 2] = xz + yw

    R[..., 1, 0] = xy + zw
    R[..., 1, 1] = 1 - xx - zz
    R[..., 1, 2] = yz - xw

    R[..., 2, 0] = xz - yw
    R[..., 2, 1] = yz + xw
    R[..., 2, 2] = 1 - xx - yy

    return R


class Shift(nn.Module):
    def __init__(self, shift) -> None:
        super().__init__()
        self.shift = shift

    def forward(self, x):
        return x + self.shift


class NVP(nn.Module):
    # * from v2.5
    # * control the tanh range
    def __init__(
        self,
        n_layers,
        feature_dims,
        hidden_size,
        proj_dims,
        code_proj_hidden_size=[],
        proj_type="simple",
        block_normalize=True,
        normalization=True,
        explicit_affine=False,
        activation=nn.LeakyReLU,
        hardtanh_range=(-10.0, 10.0),
    ):
        super().__init__()
        self._checkpoint = False
        self._normalize = block_normalize
        self._explicit_affine = explicit_affine

        # make layers
        input_dims = 3
        normalization = nn.InstanceNorm1d if normalization else None

        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.code_projectors = nn.ModuleList()
        self.layer_idx = [i for i in range(n_layers)]

        i = 0
        mask_selection = []
        while i < n_layers:
            mask_selection.append(torch.randperm(input_dims))
            i += input_dims
        mask_selection = torch.cat(mask_selection)

        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]

        hardtanh_r = hardtanh_range[1] + hardtanh_range[0]
        hardtanh_shift = hardtanh_r / 2.0

        for i in self.layer_idx:
            # get mask
            mask2 = torch.zeros(input_dims)
            mask2[mask_selection[i]] = 1
            mask1 = 1 - mask2

            # get z transform
            map_s = nn.Sequential(
                MLP(
                    proj_dims + feature_dims,
                    input_dims,
                    hidden_size,
                    bn=normalization,
                    act=activation,
                ),
                # nn.Softplus(beta=100, threshold=20),
                # Shift(0),
                nn.Hardtanh(
                    min_val=hardtanh_range[0] - hardtanh_shift,
                    max_val=hardtanh_range[1] - hardtanh_shift,
                ),
                Shift(hardtanh_shift),
            )
            map_t = MLP(
                proj_dims + feature_dims,
                input_dims,
                hidden_size,
                bn=normalization,
                act=activation,
            )
            proj = get_projection_layer(proj_dims=proj_dims, type=proj_type)
            self.layers1.append(
                CouplingLayer(map_s, map_t, proj, mask1[None, None, None])
            )

            # get xy transform (tiny)
            map_s = nn.Sequential(
                MLP(
                    proj_dims + feature_dims,
                    input_dims,
                    hidden_size[:1],
                    bn=normalization,
                    act=activation,
                ),
                # nn.Softplus(beta=100, threshold=20),
                # Shift(0),
                nn.Hardtanh(
                    min_val=hardtanh_range[0] - hardtanh_shift,
                    max_val=hardtanh_range[1] - hardtanh_shift,
                ),
                Shift(hardtanh_shift),
            )
            map_t = MLP(
                proj_dims + feature_dims,
                input_dims,
                hidden_size[:1],
                bn=normalization,
                act=activation,
            )
            proj = get_projection_layer(proj_dims=proj_dims, type=proj_type)
            self.layers2.append(
                CouplingLayer(map_s, map_t, proj, mask2[None, None, None])
            )

            # get code projector
            if len(code_proj_hidden_size) == 0:
                code_proj_hidden_size = [feature_dims]
            self.code_projectors.append(
                MLP(
                    feature_dims,
                    feature_dims,
                    code_proj_hidden_size,
                    bn=normalization,
                    act=activation,
                )
            )

        if self._normalize:
            self.scales = nn.Sequential(
                nn.Linear(feature_dims, hidden_size[0]),
                nn.ReLU(),
                nn.Linear(hidden_size[0], 3),
            )

        if self._explicit_affine:
            self.rotations = nn.Sequential(
                nn.Linear(feature_dims, hidden_size[0]),
                nn.ReLU(),
                nn.Linear(hidden_size[0], 4),
            )
            self.translations = nn.Sequential(
                nn.Linear(feature_dims, hidden_size[0]),
                nn.ReLU(),
                nn.Linear(hidden_size[0], 3),
            )

        self.reinit()

    def reinit(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # if hasattr(m.weight,'data'):
                #    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5*gain))
                if hasattr(m.bias, "data"):
                    m.bias.data.zero_()

    def _check_shapes(self, F, x):
        B1, M1, _ = F.shape  # batch, templates, C
        B2, _, M2, D = x.shape  # batch, Npts, templates, 3
        assert B1 == B2 and M1 == M2 and D == 3

    def _expand_features(self, F, x):
        _, N, _, _ = x.shape
        return F[:, None].expand(-1, N, -1, -1)

    def _call(self, func, *args, **kwargs):
        if self._checkpoint:
            return checkpoint(func, *args, **kwargs)
        else:
            return func(*args, **kwargs)

    def _normalize_input(self, F, y):
        if not self._normalize:
            return 0, 1

        sigma = torch.nn.functional.elu(self.scales(F)) + 1
        sigma = sigma[:, None]

        return 0, sigma

    def _affine_input(self, F, y):
        if not self._explicit_affine:
            return torch.eye(3)[None, None, None].to(F.device), 0

        q = self.rotations(F)
        q = q / torch.sqrt((q**2).sum(-1, keepdim=True))
        # R = quaternions_to_rotation_matrices(q[:, None])
        R = quaternion_to_matrix(q[:, None])
        t = self.translations(F)[:, None]

        return R, t

    def forward(self, F, x):
        self._check_shapes(F, x)
        mu, sigma = self._normalize_input(F, x)
        R, t = self._affine_input(F, x)
        # F: B,N,T,C x: B,N,T,3
        y = x
        y = torch.matmul(y.unsqueeze(-2), R).squeeze(-2) + t
        for i in self.layer_idx:
            # get block condition code
            Fi = self.code_projectors[i](F)
            Fi = self._expand_features(Fi, y)
            # first transformation
            l1 = self.layers1[i]
            y, _ = self._call(l1, Fi, y)
            # second transformation
            l2 = self.layers2[i]
            y, _ = self._call(l2, Fi, y)
        y = y / sigma + mu
        return y

    def inverse(self, F, y):
        self._check_shapes(F, y)
        mu, sigma = self._normalize_input(F, y)
        R, t = self._affine_input(F, y)

        x = y
        x = (x - mu) * sigma
        ldj = 0
        for i in reversed(self.layer_idx):
            # get block condition code
            Fi = self.code_projectors[i](F)
            Fi = self._expand_features(Fi, x)
            # reverse second transformation
            l2 = self.layers2[i]
            x, _ = self._call(l2.inverse, Fi, x)
            # ldj = ldj + ldji
            # reverse first transformation
            l1 = self.layers1[i]
            x, _ = self._call(l1.inverse, Fi, x)
            # ldj = ldj + ldji
        x = torch.matmul((x - t).unsqueeze(-2), R.transpose(-2, -1)).squeeze(-2)
        return x


import torch
from torch import nn


class BaseProjectionLayer(nn.Module):
    @property
    def proj_dims(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()


class IdentityProjection(BaseProjectionLayer):
    def __init__(self, input_dims):
        super().__init__()
        self._input_dims = input_dims

    @property
    def proj_dims(self):
        return self._input_dims

    def forward(self, x):
        return x


class ProjectionLayer(BaseProjectionLayer):
    def __init__(self, input_dims, proj_dims):
        super().__init__()
        self._proj_dims = proj_dims

        self.proj = nn.Sequential(
            nn.Linear(input_dims, 2 * proj_dims),
            nn.ReLU(),
            nn.Linear(2 * proj_dims, proj_dims),
        )

    @property
    def proj_dims(self):
        return self._proj_dims

    def forward(self, x):
        return self.proj(x)


class FixedPositionalEncoding(ProjectionLayer):
    def __init__(self, input_dims, proj_dims):
        super().__init__(input_dims, proj_dims)
        pi = 3.141592653589793
        ll = proj_dims // 2
        self.sigma = pi * torch.pow(2, torch.linspace(0, ll - 1, ll)).view(1, -1)

    @property
    def proj_dims(self):
        return self._proj_dims * 3

    def forward(self, x):
        device = x.device
        return torch.cat(
            [
                torch.sin(
                    x[:, :, :, :, None] * self.sigma[None, None, None].to(device)
                ),
                torch.cos(
                    x[:, :, :, :, None] * self.sigma[None, None, None].to(device)
                ),
            ],
            dim=-1,
        ).view(x.shape[0], x.shape[1], x.shape[2], -1)


class GaussianRandomFourierFeatures(ProjectionLayer):
    def __init__(self, input_dims, proj_dims, gamma=1.0):
        super().__init__(input_dims, proj_dims)
        self._two_pi = 6.283185307179586
        self._gamma = gamma
        ll = proj_dims // 2
        self.register_buffer("B", torch.randn(3, ll))

    def forward(self, x):
        xB = x.matmul(self.B * self._two_pi * self._gamma)
        return torch.cat([torch.cos(xB), torch.sin(xB)], dim=-1)


def get_projection_layer(**kwargs):
    type = kwargs["type"]

    if type == "identity":
        return IdentityProjection(3)
    elif type == "simple":
        return ProjectionLayer(3, kwargs.get("proj_dims", 128))
    elif type == "fixed_positional_encoding":
        return FixedPositionalEncoding(3, kwargs.get("proj_dims", 10))
    elif type == "gaussianrff":
        return GaussianRandomFourierFeatures(
            3, kwargs.get("proj_dims", 10), kwargs.get("gamma", 1.0)
        )
