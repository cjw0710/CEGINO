from __future__ import annotations

from typing import List, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

from .channel_mlp import ChannelMLP
from .complex import ctanh, ComplexValued
from .normalization_layers import AdaIN, InstanceNorm
from .skip_connections import skip_connection
from ..utils import validate_scaling_factor


class SubModule(nn.Module):
    def __init__(self, main_module, indices):
        super().__init__()
        self.main_module = main_module
        self.indices = indices

    def forward(self, x):
        return self.main_module.forward(x, self.indices)


class ComplexGELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if x.is_complex():
            return F.gelu(x.real).type(x.dtype) + 1j * F.gelu(x.imag).type(x.dtype)
        return F.gelu(x)
from .harmonic_projector import HarmonicProjector
from .harmonic_spectral_convolution import HarmonicSpectralConv

Number = Union[int, float]


class HarmonicLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_modes, lmax: int = 2, radial_bins: int = 16):
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = (n_modes, n_modes, n_modes)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = tuple(int(m) for m in n_modes)
        self.projector = HarmonicProjector(self.n_modes, lmax=lmax, radial_bins=radial_bins)
        self.hconv = HarmonicSpectralConv(
            in_channels=in_channels,
            out_channels=out_channels,
            shell_count=self.projector.shell_count,
            lmax=lmax,
        )

    def forward(self, x: torch.Tensor, output_shape=None):
        x_fft = torch.fft.fftn(x, dim=(-3, -2, -1))
        x_fft_sliced = self.projector.slice_fft(x_fft)
        coeffs = self.projector.project(x_fft_sliced)
        coeffs = self.hconv(coeffs)
        x_fft_sliced = self.projector.inverse(coeffs)

        full_shape = tuple(x.shape[-3:]) if output_shape is None else tuple(output_shape)
        full_fft = self.projector.scatter_to_full_fft(x_fft_sliced, full_shape)
        out = torch.fft.ifftn(full_fft, dim=(-3, -2, -1))
        return out

    def transform(self, x: torch.Tensor, output_shape=None):
        if output_shape is None or tuple(output_shape) == tuple(x.shape[-3:]):
            return x

        mode = 'trilinear'
        if x.is_complex():
            real = F.interpolate(x.real, size=output_shape, mode=mode, align_corners=False)
            imag = F.interpolate(x.imag, size=output_shape, mode=mode, align_corners=False)
            return torch.complex(real, imag)
        return F.interpolate(x, size=output_shape, mode=mode, align_corners=False)


class HarmonicFNOBlocks(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        resolution_scaling_factor=None,
        n_layers=1,
        max_n_modes=None,
        fno_block_precision='full',
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        non_linearity=F.gelu,
        stabilizer=None,
        norm=None,
        ada_in_features=None,
        preactivation=False,
        fno_skip='linear',
        channel_mlp_skip='soft-gating',
        complex_data=True,
        separable=False,
        factorization=None,
        rank=1.0,
        fixed_rank_modes=False,
        implementation='reconstructed',
        decomposition_kwargs=dict(),
        lmax=2,
        radial_bins=16,
        **kwargs,
    ):
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self._n_modes = n_modes
        self.n_dim = len(n_modes)
        self.resolution_scaling_factor = validate_scaling_factor(resolution_scaling_factor, self.n_dim, n_layers)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.stabilizer = stabilizer
        self.complex_data = complex_data
        self.preactivation = preactivation
        self.ada_in_features = ada_in_features
        self.lmax = lmax
        self.radial_bins = radial_bins
        self.n_norms = 2

        self.non_linearity = ComplexGELU() if self.complex_data else non_linearity

        self.convs = nn.ModuleList([
            HarmonicLayer(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                n_modes=self.n_modes,
                lmax=self.lmax,
                radial_bins=self.radial_bins,
            )
            for _ in range(n_layers)
        ])

        self.fno_skips = nn.ModuleList([
            skip_connection(self.in_channels, self.out_channels, skip_type=fno_skip, n_dim=self.n_dim)
            for _ in range(n_layers)
        ])

        self.channel_mlp = nn.ModuleList([
            ChannelMLP(
                in_channels=self.out_channels,
                hidden_channels=round(self.out_channels * channel_mlp_expansion),
                dropout=channel_mlp_dropout,
                n_dim=self.n_dim,
            )
            for _ in range(n_layers)
        ])

        self.channel_mlp_skips = nn.ModuleList([
            skip_connection(self.in_channels, self.out_channels, skip_type=channel_mlp_skip, n_dim=self.n_dim)
            for _ in range(n_layers)
        ])

        if self.complex_data:
            self.fno_skips = nn.ModuleList([ComplexValued(x) for x in self.fno_skips])
            self.channel_mlp = nn.ModuleList([ComplexValued(x) for x in self.channel_mlp])
            self.channel_mlp_skips = nn.ModuleList([ComplexValued(x) for x in self.channel_mlp_skips])

        if norm is None:
            self.norm = None
        elif norm == 'instance_norm':
            self.norm = nn.ModuleList([InstanceNorm() for _ in range(n_layers * self.n_norms)])
        elif norm == 'group_norm':
            self.norm = nn.ModuleList([nn.GroupNorm(num_groups=1, num_channels=self.out_channels) for _ in range(n_layers * self.n_norms)])
        elif norm == 'ada_in':
            self.norm = nn.ModuleList([AdaIN(ada_in_features, out_channels) for _ in range(n_layers * self.n_norms)])
        else:
            raise ValueError(f'Got norm={norm} but expected None or one of [instance_norm, group_norm, ada_in]')

    def forward(self, x, index=0, output_shape=None):
        if self.preactivation:
            return self.forward_with_preactivation(x, index, output_shape)
        return self.forward_with_postactivation(x, index, output_shape)

    def forward_with_postactivation(self, x, index=0, output_shape=None):
        layer = self.convs[index]

        if self.complex_data and not x.is_complex():
            x_input = x.to(torch.cfloat)
        else:
            x_input = x

        x_skip_fno = self.fno_skips[index](x_input)
        x_skip_fno = layer.transform(x_skip_fno, output_shape=output_shape)

        x_skip_channel_mlp = self.channel_mlp_skips[index](x_input)
        x_skip_channel_mlp = layer.transform(x_skip_channel_mlp, output_shape=output_shape)

        if self.stabilizer == 'tanh':
            x_input = ctanh(x_input) if self.complex_data else torch.tanh(x_input)

        x_fno = layer(x_input, output_shape=output_shape)
        if self.norm is not None:
            x_fno = self.norm[self.n_norms * index](x_fno)

        x = x_fno + x_skip_fno
        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        x = self.channel_mlp[index](x) + x_skip_channel_mlp
        if self.norm is not None:
            x = self.norm[self.n_norms * index + 1](x)

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        return x.real if self.complex_data else x

    def forward_with_preactivation(self, x, index=0, output_shape=None):
        layer = self.convs[index]
        if self.complex_data and not x.is_complex():
            x = x.to(torch.cfloat)

        x = self.non_linearity(x)
        if self.norm is not None:
            x = self.norm[self.n_norms * index](x)

        x_skip_fno = self.fno_skips[index](x)
        x_skip_fno = layer.transform(x_skip_fno, output_shape=output_shape)

        x_skip_channel_mlp = self.channel_mlp_skips[index](x)
        x_skip_channel_mlp = layer.transform(x_skip_channel_mlp, output_shape=output_shape)

        if self.stabilizer == 'tanh':
            x = ctanh(x) if self.complex_data else torch.tanh(x)

        x_fno = layer(x, output_shape=output_shape)
        x = x_fno + x_skip_fno

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        if self.norm is not None:
            x = self.norm[self.n_norms * index + 1](x)

        x = self.channel_mlp[index](x) + x_skip_channel_mlp
        return x.real if self.complex_data else x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self._n_modes = n_modes

    def get_block(self, indices):
        if self.n_layers == 1:
            raise ValueError('A single layer is parametrized, directly use the main class.')
        return SubModule(self, indices)

    def __getitem__(self, indices):
        return self.get_block(indices)

    def set_ada_in_embeddings(self, *embeddings):
        if self.norm is None:
            return
        if len(embeddings) == 1:
            for norm in self.norm:
                norm.set_embedding(embeddings[0])
        else:
            for norm, embedding in zip(self.norm, embeddings):
                norm.set_embedding(embedding)
