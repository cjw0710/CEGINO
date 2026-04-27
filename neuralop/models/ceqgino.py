from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import fps

from ..layers.equivariant_latent_ode import ContinuousInvariantFeatureODE


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


class InvariantAnchorLift(nn.Module):
    """
    Learned invariant lift from full mesh -> anchor features.

    v3:
    - keep flow-aware invariants
    - add global node summary
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        cond_channels: int = 0,
        k_neighbors: int = 128,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cond_channels = cond_channels
        self.k_neighbors = k_neighbors

        self.node_encoder = _MLP(in_channels, hidden_channels, hidden_channels)

        edge_in_dim = hidden_channels + 4 + cond_channels
        out_in_dim = hidden_channels + hidden_channels + cond_channels

        self.edge_mlp = _MLP(edge_in_dim, hidden_channels, hidden_channels)
        self.edge_gate = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
        )
        self.global_mlp = _MLP(hidden_channels + cond_channels, hidden_channels, hidden_channels)
        self.out_mlp = _MLP(out_in_dim, hidden_channels, hidden_channels)

    def forward(
        self,
        input_coords: torch.Tensor,    # [1, N, 3]
        input_x: torch.Tensor,         # [1, N, Cin]
        anchor_coords: torch.Tensor,   # [1, M, 3]
        flow_dir: torch.Tensor,        # [1, 3]
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = input_coords[0]
        fin = input_x[0]
        a = anchor_coords[0]
        u = flow_dir[0]
        u = u / (u.norm() + 1e-8)

        node_feat = self.node_encoder(fin)  # [N, H]

        d2 = torch.cdist(a, x) ** 2
        k = min(self.k_neighbors, x.size(0))
        d2_k, idx = torch.topk(d2, k=k, dim=-1, largest=False)

        neigh_feat = node_feat[idx]

        node_dot_u = (x * u).sum(dim=-1)
        anchor_dot_u = (a * u).sum(dim=-1, keepdim=True)

        node_dot_u_k = node_dot_u[idx].unsqueeze(-1)
        anchor_dot_u_k = anchor_dot_u.unsqueeze(1).expand(-1, k, -1)

        rel = a.unsqueeze(1) - x[idx]
        rel_dot_u = (rel * u.view(1, 1, 3)).sum(dim=-1, keepdim=True)

        if cond is not None:
            cond_edge = cond.unsqueeze(1).expand(a.size(0), k, -1)
            cond_anchor = cond.expand(a.size(0), -1)
            cond_global = cond
        else:
            cond_edge = neigh_feat.new_zeros((a.size(0), k, 0))
            cond_anchor = neigh_feat.new_zeros((a.size(0), 0))
            cond_global = neigh_feat.new_zeros((1, 0))

        edge_input = torch.cat(
            [neigh_feat, d2_k.unsqueeze(-1), anchor_dot_u_k, node_dot_u_k, rel_dot_u, cond_edge],
            dim=-1,
        )
        edge_hidden = self.edge_mlp(edge_input)
        logits = self.edge_gate(edge_hidden).squeeze(-1) - d2_k
        weights = torch.softmax(logits, dim=-1)
        agg = (weights.unsqueeze(-1) * edge_hidden).sum(dim=1)  # [M, H]

        global_feat = node_feat.mean(dim=0, keepdim=True)
        global_ctx = self.global_mlp(torch.cat([global_feat, cond_global], dim=-1))  # [1, H]
        global_anchor = global_ctx.expand(a.size(0), -1)

        out_input = torch.cat([agg, global_anchor, cond_anchor], dim=-1)
        anchor_feat = self.out_mlp(out_input)
        return anchor_feat.unsqueeze(0)


class InvariantCrossDecoder(nn.Module):
    """
    Learned invariant decoder from anchors -> query outputs.

    v3:
    - two-scale local aggregation (small + large neighborhood)
    - global anchor context
    - flow-aware invariants in both branches
    """

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        cond_channels: int = 0,
        k_small: int = 16,
        k_large: int = 48,
        query_chunk_size: int = 8192,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.cond_channels = cond_channels
        self.k_small = k_small
        self.k_large = k_large
        self.query_chunk_size = query_chunk_size

        edge_in_dim = hidden_channels + 4 + cond_channels
        out_in_dim = hidden_channels + hidden_channels + hidden_channels + cond_channels + 1

        self.edge_mlp_small = _MLP(edge_in_dim, hidden_channels, hidden_channels)
        self.edge_gate_small = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
        )

        self.edge_mlp_large = _MLP(edge_in_dim, hidden_channels, hidden_channels)
        self.edge_gate_large = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
        )

        self.global_mlp = _MLP(hidden_channels + cond_channels, hidden_channels, hidden_channels)
        self.out_mlp = _MLP(out_in_dim, hidden_channels, out_channels)

    def _aggregate_branch(
        self,
        q_chunk: torch.Tensor,
        a: torch.Tensor,
        h: torch.Tensor,
        u: torch.Tensor,
        cond: torch.Tensor | None,
        k: int,
        edge_mlp: nn.Module,
        edge_gate: nn.Module,
    ) -> torch.Tensor:
        d2 = torch.cdist(q_chunk, a) ** 2
        d2_k, idx = torch.topk(d2, k=k, dim=-1, largest=False)

        neigh_feat = h[idx]

        query_dot_u = (q_chunk * u).sum(dim=-1, keepdim=True)
        query_dot_u_k = query_dot_u.unsqueeze(1).expand(-1, k, -1)

        anchor_dot_u = (a * u).sum(dim=-1)
        anchor_dot_u_k = anchor_dot_u[idx].unsqueeze(-1)

        rel = q_chunk.unsqueeze(1) - a[idx]
        rel_dot_u = (rel * u.view(1, 1, 3)).sum(dim=-1, keepdim=True)

        if cond is not None:
            cond_edge = cond.unsqueeze(1).expand(q_chunk.size(0), k, -1)
        else:
            cond_edge = neigh_feat.new_zeros((q_chunk.size(0), k, 0))

        edge_input = torch.cat(
            [neigh_feat, d2_k.unsqueeze(-1), query_dot_u_k, anchor_dot_u_k, rel_dot_u, cond_edge],
            dim=-1,
        )
        edge_hidden = edge_mlp(edge_input)
        logits = edge_gate(edge_hidden).squeeze(-1) - d2_k
        weights = torch.softmax(logits, dim=-1)
        agg = (weights.unsqueeze(-1) * edge_hidden).sum(dim=1)
        return agg

    def forward(
        self,
        anchor_coords: torch.Tensor,    # [1, M, 3]
        anchor_feat: torch.Tensor,      # [1, M, H]
        query_coords: torch.Tensor,     # [1, Q, 3]
        flow_dir: torch.Tensor,         # [1, 3]
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        a = anchor_coords[0]
        h = anchor_feat[0]
        q = query_coords[0]
        u = flow_dir[0]
        u = u / (u.norm() + 1e-8)

        k_small = min(self.k_small, a.size(0))
        k_large = min(self.k_large, a.size(0))

        if cond is not None:
            cond_query = cond.expand(1, -1)
            cond_global = cond
        else:
            cond_query = h.new_zeros((1, 0))
            cond_global = h.new_zeros((1, 0))

        global_feat = h.mean(dim=0, keepdim=True)
        global_ctx = self.global_mlp(torch.cat([global_feat, cond_global], dim=-1))  # [1, H]

        outputs = []

        for start in range(0, q.size(0), self.query_chunk_size):
            q_chunk = q[start:start + self.query_chunk_size]

            agg_small = self._aggregate_branch(
                q_chunk, a, h, u, cond, k_small, self.edge_mlp_small, self.edge_gate_small
            )
            agg_large = self._aggregate_branch(
                q_chunk, a, h, u, cond, k_large, self.edge_mlp_large, self.edge_gate_large
            )

            q_dot_u = (q_chunk * u).sum(dim=-1, keepdim=True)
            cond_q = cond_query.expand(q_chunk.size(0), -1)
            global_q = global_ctx.expand(q_chunk.size(0), -1)

            out_input = torch.cat([agg_small, agg_large, global_q, cond_q, q_dot_u], dim=-1)
            out = self.out_mlp(out_input)
            outputs.append(out)

        return torch.cat(outputs, dim=0).unsqueeze(0)


class CEqGINO(nn.Module):
    """
    Continuous-Equivariant GINO v3
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        gno_coord_dim: int,
        gno_radius: float,
        fno_n_modes,
        fno_hidden_channels: int,
        fno_n_layers: int,
        fno_norm,
        fno_in_channels: int,
        latent_feature_channels: int,
        ode_steps: int = 4,
        ode_max_neighbors: int = 32,
        interp_scale: float = 0.05,   # kept for API compatibility
        anchor_count: int = 512,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = fno_hidden_channels
        self.latent_feature_channels = latent_feature_channels
        self.ode_steps = ode_steps
        self.anchor_count = anchor_count

        self.anchor_lift = InvariantAnchorLift(
            in_channels=in_channels,
            hidden_channels=self.hidden_channels,
            cond_channels=latent_feature_channels,
            k_neighbors=min(128, anchor_count),
        )

        self.dynamics = ContinuousInvariantFeatureODE(
            hidden_channels=self.hidden_channels,
            cond_channels=latent_feature_channels,
            radius=gno_radius,
            max_num_neighbors=ode_max_neighbors,
            message_channels=2 * self.hidden_channels,
        )

        self.query_decoder = InvariantCrossDecoder(
            hidden_channels=self.hidden_channels,
            out_channels=out_channels,
            cond_channels=latent_feature_channels,
            k_small=min(16, anchor_count),
            k_large=min(48, anchor_count),
            query_chunk_size=8192,
        )

    def _pool_condition(self, latent_features: torch.Tensor | None) -> torch.Tensor | None:
        if latent_features is None:
            return None
        if latent_features.dim() == 5:
            return latent_features.mean(dim=(1, 2, 3))
        if latent_features.dim() == 3:
            return latent_features.mean(dim=1)
        if latent_features.dim() == 2:
            return latent_features
        raise RuntimeError(f"Unsupported latent_features shape: {latent_features.shape}")

    def _select_anchors(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.size(0) != 1:
            raise RuntimeError("CEqGINO v3 currently supports batch_size=1 only.")

        x = coords[0]
        n = x.size(0)
        m = min(self.anchor_count, n)
        batch0 = torch.zeros(n, dtype=torch.long, device=x.device)

        ratio = max(float(m) / float(n), 1.0 / float(n))
        idx = fps(x, batch=batch0, ratio=ratio, random_start=False)

        if idx.numel() > m:
            idx = idx[:m]
        elif idx.numel() < m:
            base = torch.linspace(0, n - 1, steps=m, device=x.device).long()
            idx = torch.unique(torch.cat([idx, base], dim=0))[:m]

        return coords[:, idx, :]

    def _integrate(
        self,
        coords: torch.Tensor,
        h: torch.Tensor,
        flow_dir: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dt = 1.0 / float(self.ode_steps)
        for _ in range(self.ode_steps):
            dh = self.dynamics(coords, h, flow_dir=flow_dir, cond=cond, batch_index=None)
            h = h + dt * dh
        return h

    def forward(
        self,
        input_geom: torch.Tensor,
        latent_queries: torch.Tensor | None = None,
        latent_features: torch.Tensor | None = None,
        output_queries: torch.Tensor | None = None,
        x: torch.Tensor | None = None,
        flow_dir: torch.Tensor | None = None,
        batch_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x is None:
            raise RuntimeError("CEqGINO.forward requires x")
        if flow_dir is None:
            raise RuntimeError("CEqGINO.forward requires flow_dir")

        if input_geom.dim() == 2:
            input_geom = input_geom.unsqueeze(0)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if flow_dir.dim() == 1:
            flow_dir = flow_dir.unsqueeze(0)

        if output_queries is None:
            output_queries = input_geom
        elif output_queries.dim() == 2:
            output_queries = output_queries.unsqueeze(0)

        cond = self._pool_condition(latent_features)

        anchor_coords = self._select_anchors(input_geom)
        anchor_feat = self.anchor_lift(
            input_coords=input_geom,
            input_x=x,
            anchor_coords=anchor_coords,
            flow_dir=flow_dir,
            cond=cond,
        )

        anchor_feat = self._integrate(
            coords=anchor_coords,
            h=anchor_feat,
            flow_dir=flow_dir,
            cond=cond,
        )

        out = self.query_decoder(
            anchor_coords=anchor_coords,
            anchor_feat=anchor_feat,
            query_coords=output_queries,
            flow_dir=flow_dir,
            cond=cond,
        )
        return out