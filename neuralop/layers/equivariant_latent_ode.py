from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import radius_graph


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContinuousInvariantFeatureODE(nn.Module):
    """
    Continuous-time latent dynamics on scalar anchor features.

    v3:
    - keep flow-aware invariants
    - add global latent context into node updates
    - slightly stronger residual dynamics for stability/capacity

    Assumes batch_size = 1.
    """

    def __init__(
        self,
        hidden_channels: int,
        cond_channels: int = 0,
        radius: float = 0.1,
        max_num_neighbors: int = 32,
        message_channels: int | None = None,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cond_channels = cond_channels
        self.radius = radius
        self.max_num_neighbors = max_num_neighbors
        self.message_channels = message_channels or (2 * hidden_channels)

        # [f_i, f_j, f_i-f_j, dist2, xi·u, xj·u, (xi-xj)·u, cond]
        edge_in_dim = 3 * hidden_channels + 4 + cond_channels
        global_in_dim = hidden_channels + cond_channels
        update_in_dim = hidden_channels + self.message_channels + cond_channels + hidden_channels

        self.node_norm = nn.LayerNorm(hidden_channels)

        self.edge_mlp = _MLP(edge_in_dim, self.message_channels, self.message_channels)
        self.edge_gate = nn.Sequential(
            nn.Linear(self.message_channels, self.message_channels),
            nn.GELU(),
            nn.Linear(self.message_channels, 1),
        )

        self.global_mlp = _MLP(global_in_dim, hidden_channels, hidden_channels)
        self.update_mlp = _MLP(update_in_dim, self.message_channels, hidden_channels)

        self.res_scale = nn.Parameter(torch.tensor(0.15))

    def forward(
        self,
        coords: torch.Tensor,              # [B, M, 3]
        h: torch.Tensor,                   # [B, M, C]
        flow_dir: torch.Tensor,            # [B, 3]
        cond: torch.Tensor | None = None,  # [B, Cc]
        batch_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if coords.dim() != 3 or h.dim() != 3:
            raise RuntimeError(
                f"Expected coords [B,M,3] and h [B,M,C], got {coords.shape} and {h.shape}"
            )
        if coords.size(0) != 1:
            raise RuntimeError("ContinuousInvariantFeatureODE v3 currently supports batch_size=1 only.")

        if flow_dir.dim() == 1:
            flow_dir = flow_dir.unsqueeze(0)
        if flow_dir.dim() != 2 or flow_dir.size(0) != 1 or flow_dir.size(1) != 3:
            raise RuntimeError(f"Expected flow_dir [1,3], got {flow_dir.shape}")

        x = coords[0]                 # [M, 3]
        feat = self.node_norm(h[0])   # [M, C]
        u = flow_dir[0]
        u = u / (u.norm() + 1e-8)

        if batch_index is None:
            batch0 = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        else:
            batch0 = batch_index

        edge_index = radius_graph(
            x,
            r=self.radius,
            batch=batch0,
            loop=False,
            max_num_neighbors=self.max_num_neighbors,
        )
        row, col = edge_index

        if row.numel() == 0:
            return torch.zeros_like(h)

        xi = x[row]
        xj = x[col]
        rel = xi - xj

        dist2 = (rel * rel).sum(dim=-1, keepdim=True)
        xi_dot_u = (xi * u).sum(dim=-1, keepdim=True)
        xj_dot_u = (xj * u).sum(dim=-1, keepdim=True)
        rel_dot_u = (rel * u).sum(dim=-1, keepdim=True)

        fi = feat[row]
        fj = feat[col]
        fd = fi - fj

        if cond is not None:
            cond_node = cond.expand(feat.size(0), -1)
            cond_edge = cond.expand(row.numel(), -1)
            cond_global = cond
        else:
            cond_node = feat.new_zeros((feat.size(0), 0))
            cond_edge = feat.new_zeros((row.numel(), 0))
            cond_global = feat.new_zeros((1, 0))

        edge_input = torch.cat(
            [fi, fj, fd, dist2, xi_dot_u, xj_dot_u, rel_dot_u, cond_edge],
            dim=-1,
        )
        edge_hidden = self.edge_mlp(edge_input)
        gate = torch.sigmoid(self.edge_gate(edge_hidden))
        msg = gate * edge_hidden

        agg = feat.new_zeros((feat.size(0), self.message_channels))
        agg.index_add_(0, row, msg)

        deg = feat.new_zeros((feat.size(0), 1))
        deg.index_add_(0, row, torch.ones((row.numel(), 1), device=x.device, dtype=feat.dtype))
        agg = agg / deg.clamp_min(1.0)

        global_feat = feat.mean(dim=0, keepdim=True)  # [1, C]
        global_ctx = self.global_mlp(torch.cat([global_feat, cond_global], dim=-1))  # [1, C]
        global_node = global_ctx.expand(feat.size(0), -1)

        update_input = torch.cat([feat, agg, cond_node, global_node], dim=-1)
        dh = self.update_mlp(update_input)

        dh = torch.tanh(self.res_scale) * dh
        return dh.unsqueeze(0)
