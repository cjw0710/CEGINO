from __future__ import annotations

import math
from typing import List, Tuple

import torch
from torch import nn


class HarmonicProjector(nn.Module):
    """
    Shell-wise harmonic projector on a Cartesian FFT lattice.

    This is a practical first-stage implementation for the continuous-SO(3)
    branch: it keeps the existing full-FFT pipeline, but reparameterizes the
    selected spectral cube into shell-wise spherical harmonic coefficients.

    Current implementation supports lmax <= 2.
    """

    def __init__(self, n_modes: Tuple[int, int, int], lmax: int = 2, radial_bins: int = 16, eps: float = 1e-6):
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = (n_modes, n_modes, n_modes)
        if lmax > 2:
            raise ValueError('Current HarmonicProjector implementation supports lmax <= 2.')

        self.n_modes = tuple(int(m) for m in n_modes)
        self.lmax = int(lmax)
        self.radial_bins = int(radial_bins)
        self.eps = eps
        self.num_harmonics = (self.lmax + 1) ** 2
        self.l_slices = self._build_l_slices(self.lmax)

        idx_x, idx_y, idx_z, shell_indices, basis_list, pinv_list = self._build_buffers()
        self.register_buffer('idx_x', idx_x)
        self.register_buffer('idx_y', idx_y)
        self.register_buffer('idx_z', idx_z)
        self.shell_count = len(shell_indices)

        for i, (idx, basis, pinv) in enumerate(zip(shell_indices, basis_list, pinv_list)):
            self.register_buffer(f'shell_idx_{i}', idx)
            self.register_buffer(f'shell_basis_{i}', basis)
            self.register_buffer(f'shell_pinv_{i}', pinv)

    @staticmethod
    def _build_l_slices(lmax: int) -> List[slice]:
        slices = []
        start = 0
        for l in range(lmax + 1):
            width = 2 * l + 1
            slices.append(slice(start, start + width))
            start += width
        return slices

    @staticmethod
    def _symmetric_k_values(n_mode: int) -> torch.Tensor:
        k = n_mode // 2
        k_pos = k + (n_mode % 2)
        k_neg = k
        pos = torch.arange(k_pos, dtype=torch.float32)
        neg = torch.arange(-k_neg, 0, dtype=torch.float32)
        return torch.cat([pos, neg], dim=0)

    def _build_buffers(self):
        kx = self._symmetric_k_values(self.n_modes[0])
        ky = self._symmetric_k_values(self.n_modes[1])
        kz = self._symmetric_k_values(self.n_modes[2])
        idx_x = torch.arange(self.n_modes[0], dtype=torch.long)
        idx_y = torch.arange(self.n_modes[1], dtype=torch.long)
        idx_z = torch.arange(self.n_modes[2], dtype=torch.long)

        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
        coords = torch.stack([KX, KY, KZ], dim=-1).reshape(-1, 3)
        radii = torch.norm(coords, dim=-1)

        max_r = max(float(radii.max().item()), 1.0)
        bin_edges = torch.linspace(0.0, max_r + 1e-6, self.radial_bins + 1)
        shell_ids = torch.bucketize(radii, bin_edges[1:-1], right=False)

        shell_indices = []
        basis_list = []
        pinv_list = []

        for shell_id in range(self.radial_bins):
            idx = torch.nonzero(shell_ids == shell_id, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue

            shell_coords = coords[idx]
            shell_basis = self._real_spherical_harmonics(shell_coords, self.lmax)
            shell_pinv = torch.linalg.pinv(shell_basis)

            shell_indices.append(idx.to(torch.long))
            basis_list.append(shell_basis.to(torch.float32))
            pinv_list.append(shell_pinv.to(torch.float32))

        return idx_x, idx_y, idx_z, shell_indices, basis_list, pinv_list

    def _real_spherical_harmonics(self, coords: torch.Tensor, lmax: int) -> torch.Tensor:
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        r = torch.norm(coords, dim=-1).clamp_min(self.eps)
        x = x / r
        y = y / r
        z = z / r

        basis = []
        # l = 0
        basis.append(0.28209479177387814 * torch.ones_like(x))

        if lmax >= 1:
            basis.extend([
                0.4886025119029199 * y,
                0.4886025119029199 * z,
                0.4886025119029199 * x,
            ])

        if lmax >= 2:
            basis.extend([
                1.0925484305920792 * x * y,
                1.0925484305920792 * y * z,
                0.31539156525252005 * (3.0 * z * z - 1.0),
                1.0925484305920792 * x * z,
                0.5462742152960396 * (x * x - y * y),
            ])

        basis = torch.stack(basis, dim=-1)

        # Radius-zero shell should only keep l=0 active.
        zero_mask = (coords.abs().sum(dim=-1) < self.eps)
        if zero_mask.any() and basis.shape[1] > 1:
            basis = basis.clone()
            basis[zero_mask, 1:] = 0.0
        return basis

    def slice_fft(self, x_fft: torch.Tensor) -> torch.Tensor:
        return x_fft[:, :, self.idx_x[:, None, None], self.idx_y[None, :, None], self.idx_z[None, None, :]]

    def scatter_to_full_fft(self, x_fft_sliced: torch.Tensor, full_shape: Tuple[int, int, int]) -> torch.Tensor:
        batchsize, channels = x_fft_sliced.shape[:2]
        out = torch.zeros((batchsize, channels, *full_shape), dtype=x_fft_sliced.dtype, device=x_fft_sliced.device)
        out[:, :, self.idx_x[:, None, None], self.idx_y[None, :, None], self.idx_z[None, None, :]] = x_fft_sliced
        return out

    def project(self, x_fft_sliced: torch.Tensor):
        batchsize, channels = x_fft_sliced.shape[:2]
        flat = x_fft_sliced.reshape(batchsize, channels, -1)
        coeffs_per_l = [[] for _ in range(self.lmax + 1)]

        for shell_id in range(self.shell_count):
            idx = getattr(self, f'shell_idx_{shell_id}')
            pinv = getattr(self, f'shell_pinv_{shell_id}').to(device=flat.device)
            pinv = pinv.to(dtype=flat.dtype)
            x_shell = flat.index_select(-1, idx)
            coeff_flat = torch.einsum('hp,bcp->bch', pinv, x_shell)
            for l, sl in enumerate(self.l_slices):
                coeffs_per_l[l].append(coeff_flat[..., sl])

        coeffs_per_l = [torch.stack(chunks, dim=2) for chunks in coeffs_per_l]
        return coeffs_per_l

    def inverse(self, coeffs_per_l):
        batchsize, channels = coeffs_per_l[0].shape[:2]
        flat_out = torch.zeros(
            batchsize,
            channels,
            self.n_modes[0] * self.n_modes[1] * self.n_modes[2],
            dtype=coeffs_per_l[0].dtype,
            device=coeffs_per_l[0].device,
        )

        for shell_id in range(self.shell_count):
            idx = getattr(self, f'shell_idx_{shell_id}')
            basis = getattr(self, f'shell_basis_{shell_id}').to(device=coeffs_per_l[0].device)
            basis = basis.to(dtype=coeffs_per_l[0].dtype)
            coeff_flat = torch.cat([coeffs_per_l[l][:, :, shell_id, :] for l in range(self.lmax + 1)], dim=-1)
            x_shell = torch.einsum('ph,bch->bcp', basis, coeff_flat)
            flat_out[:, :, idx] = x_shell

        return flat_out.reshape(batchsize, channels, *self.n_modes)
