
from __future__ import annotations

import math
from typing import List

import torch
from torch import nn


class HarmonicSpectralConv(nn.Module):
    """
    Stronger shell-wise harmonic spectral operator.

    coeffs_per_l[l] is expected to be:
        [B, Cin, Shell, 2l+1]
    output is:
        [B, Cout, Shell, 2l+1]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shell_count: int,
        lmax: int,
        init_scale: float | None = None,
        use_residual: bool = True,
        use_shell_gate: bool = True,
        residual_alpha_init: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shell_count = shell_count
        self.lmax = lmax
        self.use_residual = use_residual
        self.use_shell_gate = use_shell_gate

        if init_scale is None:
            init_scale = 1.0 / math.sqrt(max(in_channels, 1))

        self.main_weights = nn.ParameterList()
        self.res_weights = nn.ParameterList() if use_residual else None
        self.shell_gates = nn.ParameterList() if use_shell_gate else None
        self.biases = nn.ParameterList()

        for l in range(lmax + 1):
            w_main = torch.randn(
                shell_count, out_channels, in_channels, dtype=torch.cfloat
            ) * init_scale
            self.main_weights.append(nn.Parameter(w_main))

            if use_residual:
                if in_channels == out_channels:
                    eye = torch.eye(out_channels, in_channels, dtype=torch.float32)
                    eye = eye.unsqueeze(0).repeat(shell_count, 1, 1).to(torch.cfloat)
                    w_res = eye.clone()
                else:
                    w_res = torch.randn(
                        shell_count, out_channels, in_channels, dtype=torch.cfloat
                    ) * (0.25 * init_scale)
                self.res_weights.append(nn.Parameter(w_res))

            if use_shell_gate:
                self.shell_gates.append(nn.Parameter(torch.zeros(shell_count)))

            bias = torch.zeros(1, out_channels, shell_count, 1, dtype=torch.cfloat)
            self.biases.append(nn.Parameter(bias))

        self.residual_alpha = nn.Parameter(torch.tensor(float(residual_alpha_init)))

    def _apply_linear(self, weight: torch.Tensor, coeffs_l: torch.Tensor) -> torch.Tensor:
        """
        weight:   [Shell, Cout, Cin]
        coeffs_l: [B, Cin, Shell, M]
        return:   [B, Cout, Shell, M]
        """
        return torch.einsum("soi,bism->bosm", weight, coeffs_l)

    def forward(self, coeffs_per_l: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []

        alpha = torch.sigmoid(self.residual_alpha)

        for l, coeffs_l in enumerate(coeffs_per_l):
            y = self._apply_linear(self.main_weights[l], coeffs_l)

            if self.use_residual:
                y_res = self._apply_linear(self.res_weights[l], coeffs_l)
                y = y + alpha * y_res

            if self.use_shell_gate:
                gate = 1.0 + torch.tanh(self.shell_gates[l]).view(1, 1, self.shell_count, 1)
                y = y * gate

            y = y + self.biases[l]
            outputs.append(y)

        return outputs