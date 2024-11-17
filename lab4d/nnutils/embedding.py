# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lab4d.utils.torch_utils import frameid_to_vid


def get_fourier_embed_dim(in_channels, N_freqs):
    """Compute number of channels in frequency-encoded output

    Args:
        in_channels (int): Number of input channels (3 for both xyz / direction)
        N_freqs (int): Number of frequency bands
    Returns:
        out_channels (int): Number of output channels
    """
    if N_freqs == -1:
        return 0

    out_channels = in_channels * (2 * N_freqs + 1)
    return out_channels


class PosEmbedding(nn.Module):
    """A Fourier embedding that maps x to (x, sin(2^k x), cos(2^k x), ...)
    Adapted from https://github.com/kwea123/nerf_pl/blob/master/models/nerf.py

    Args:
        in_channels (int): Number of input channels (3 for both xyz, direction)
        N_freqs (int): Number of frequency bands
        logscale (bool): If True, construct frequency bands in log-space
    """

    def __init__(self, in_channels, N_freqs, logscale=True):
        super().__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels

        # no embedding
        if N_freqs == -1:
            self.out_channels = 0
            return

        self.funcs = [torch.sin, torch.cos]
        self.nfuncs = len(self.funcs)
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)
        self.register_buffer("freq_bands", freq_bands, persistent=False)

        self.set_alpha(None)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def set_alpha(self, alpha):
        """Set the alpha parameter for the annealing window

        Args:
            alpha (float or None): 0 to 1
        """
        self.alpha = alpha

    def forward(self, x):
        """Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Args:
            x: (B, self.in_channels)
        Returns:
            out: (B, self.out_channels)
        """
        if self.N_freqs == -1:
            return torch.zeros_like(x[..., :0])

        # cosine features
        if self.N_freqs > 0:
            shape = x.shape
            device = x.device
            input_dim = shape[-1]
            output_dim = input_dim * (1 + self.N_freqs * self.nfuncs)
            out_shape = shape[:-1] + ((output_dim),)

            # assign input coordinates to the first few output channels
            x = x.reshape(-1, input_dim)
            out = torch.empty(x.shape[0], output_dim, dtype=x.dtype, device=device)
            out[:, :input_dim] = x

            # assign fourier features to the remaining channels
            out_bands = out[:, input_dim:].view(
                -1, self.N_freqs, self.nfuncs, input_dim
            )
            for i, func in enumerate(self.funcs):
                # (B, nfreqs, input_dim) = (1, nfreqs, 1) * (B, 1, input_dim)
                out_bands[:, :, i] = func(
                    self.freq_bands[None, :, None] * x[:, None, :]
                )

            self.apply_annealing(out_bands)

            out = out.view(out_shape)
        else:
            out = x
        return out

    def apply_annealing(self, out_bands):
        """Apply the annealing window w = 0.5*( 1+cos(pi + pi clip(alpha-j)) )

        Args:
            out_bands: (..., N_freqs, nfuncs, in_channels) Frequency bands
        """
        device = out_bands.device
        if self.alpha is not None:
            alpha_freq = self.alpha * self.N_freqs
            window = alpha_freq - torch.arange(self.N_freqs).to(device)
            window = torch.clamp(window, 0.0, 1.0)
            window = 0.5 * (1 + torch.cos(np.pi * window + np.pi))
            window = window.view(1, -1, 1, 1)
            out_bands[:] = window * out_bands

    def get_mean_embedding(self, device):
        """Compute the mean Fourier embedding

        Args:
            device (torch.device): Output device
        """
        mean_embedding = torch.zeros(self.out_channels, device=device)
        return mean_embedding


class TimeEmbedding(nn.Module):
    """A learnable feature embedding per frame

    Args:
        num_freq_t (int): Number of frequencies in time embedding
        frame_info (Dict): Metadata about the frames in a dataset
        out_channels (int): Number of output channels
    """

    def __init__(self, num_freq_t, frame_info, out_channels=128, time_scale=1.0):
        super().__init__()
        self.fourier_embedding = PosEmbedding(1, num_freq_t)
        t_channels = self.fourier_embedding.out_channels
        self.out_channels = out_channels

        self.frame_offset = frame_info["frame_offset"]
        self.num_frames = self.frame_offset[-1]
        self.num_vids = len(self.frame_offset) - 1

        frame_mapping = frame_info["frame_mapping"]  # list of list
        frame_mapping = torch.tensor(frame_mapping)  # (M,)
        frame_offset_raw = frame_info["frame_offset_raw"]

        max_ts = (frame_offset_raw[1:] - frame_offset_raw[:-1]).max()
        raw_fid = torch.arange(0, frame_offset_raw[-1])
        raw_fid_to_vid = frameid_to_vid(raw_fid, frame_offset_raw)
        raw_fid_to_vstart = torch.tensor(frame_offset_raw[raw_fid_to_vid])
        raw_fid_to_vidend = torch.tensor(frame_offset_raw[raw_fid_to_vid + 1])
        raw_fid_to_vidlen = raw_fid_to_vidend - raw_fid_to_vstart

        # M
        self.register_buffer(
            "frame_to_vid", raw_fid_to_vid[frame_mapping], persistent=False
        )
        # M, in range [0,N-1], M<N
        self.register_buffer("frame_mapping", frame_mapping, persistent=False)
        # N
        self.register_buffer("raw_fid_to_vid", raw_fid_to_vid, persistent=False)
        self.register_buffer("raw_fid_to_vidlen", raw_fid_to_vidlen, persistent=False)
        self.register_buffer("raw_fid_to_vstart", raw_fid_to_vstart, persistent=False)

        # a function, make it more/less senstiive to time
        def frame_to_tid_fn(frame_id):
            if not torch.is_tensor(frame_id):
                frame_id = torch.tensor(frame_id).to(self.frame_to_vid.device)
            vid_len = self.raw_fid_to_vidlen[frame_id.long()]
            tid_sub = frame_id - self.raw_fid_to_vstart[frame_id.long()]
            tid = (tid_sub - vid_len / 2) / max_ts * 2  # [-1, 1]
            tid = tid * time_scale
            return tid

        self.frame_to_tid = frame_to_tid_fn

        self.inst_embedding = InstEmbedding(self.num_vids, inst_channels=out_channels)
        self.mapping1 = nn.Linear(t_channels, out_channels)
        self.mapping2 = nn.Linear(2 * out_channels, out_channels)

    def forward(self, frame_id=None):
        """
        Args:
            frame_id: (...,) Frame id to evaluate at, or None to use all frames
        Returns:
            t_embed (..., self.W): Output time embeddings
        """
        if frame_id is None:
            inst_id, t_sample = self.frame_to_vid, self.frame_to_tid(self.frame_mapping)
        else:
            inst_id = self.raw_fid_to_vid[frame_id]
            t_sample = self.frame_to_tid(frame_id)

        if inst_id.ndim == 1:
            inst_id = inst_id[..., None]  # (N, 1)
            t_sample = t_sample[..., None]  # (N, 1)

        coeff = self.fourier_embedding(t_sample)

        inst_code = self.inst_embedding(inst_id[..., 0])
        coeff = self.mapping1(coeff)
        t_embed = torch.cat([coeff, inst_code], -1)
        t_embed = self.mapping2(t_embed)
        return t_embed

    def get_mean_embedding(self, device):
        """Compute the mean time embedding over all frames

        Args:
            device (torch.device): Output device
        """
        t_embed = self.forward(self.frame_mapping).mean(0, keepdim=True)
        # t_embed = self.basis.weight.mean(1)
        return t_embed


class InstEmbedding(nn.Module):
    """A learnable embedding per object instance

    Args:
        num_inst (int): Number of distinct object instances. If --nosingle_inst
            is passed, this is equal to the number of videos, as we assume each
            video captures a different instance. Otherwise, we assume all videos
            capture the same instance and set this to 1.
        inst_channels (int): Number of channels in the instance code
    """

    def __init__(self, num_inst, inst_channels):
        super().__init__()
        self.out_channels = inst_channels
        self.num_inst = num_inst
        self.set_beta_prob(0.0)  # probability of sampling a random instance
        if inst_channels > 0:
            self.mapping = nn.Embedding(num_inst, inst_channels)

    def forward(self, inst_id):
        """
        Args:
            inst_id: (M,) Instance id, or None to use the average instance
        Returns:
            out: (M, self.out_channels)
        """
        if self.out_channels == 0:
            return torch.zeros(inst_id.shape + (0,), device=inst_id.device)
        else:
            if self.num_inst == 1:
                return self.mapping(torch.zeros_like(inst_id))
            if self.training and self.beta_prob > 0:
                inst_id = self.randomize_instance(inst_id)
            inst_code = self.mapping(inst_id)
            return inst_code

    def randomize_instance(self, inst_id):
        """Randomize the instance code with probability beta_prob. Used for
        code swapping regularization

        Args:
            inst_id: (M, ...) Instance id
        Returns:
            inst_id: (M, ...) Randomized instance ids
        """
        minibatch_size = inst_id.shape[0]
        rand_id = torch.randint(self.num_inst, (minibatch_size,), device=inst_id.device)
        rand_id = rand_id.reshape((minibatch_size,) + (1,) * (len(inst_id.shape) - 1))
        rand_id = rand_id.expand_as(inst_id)
        rand_mask = torch.rand_like(rand_id.float()) < self.beta_prob
        inst_id = torch.where(rand_mask, rand_id, inst_id)
        return inst_id

    def get_mean_embedding(self):
        """Compute the mean instance id"""
        return self.mapping.weight.mean(0)

    def set_beta_prob(self, beta_prob):
        """Set the beta parameter for the instance code. This is the probability
        of sampling a random instance code

        Args:
            beta_prob (float): Instance code swapping probability, 0 to 1
        """
        self.beta_prob = beta_prob
