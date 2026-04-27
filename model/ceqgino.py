import pickle
import torch
import lightning.pytorch as pl

from sklearn.metrics import r2_score
from torch.nn import Linear, Sequential
from torch.utils.data import Dataset

from neuralop.models.ceqgino import CEqGINO
from utils import (
    apply_radial_basis,
    compute_metrics,
    get_rotation_matrix,
)


def preprocess(data, args):
    rotation_matrix = get_rotation_matrix(args.aug_type, device=data.x.device)
    data.x = data.x - data.x.mean(dim=0, keepdim=True)
    data.x = torch.matmul(data.x, rotation_matrix.T)
    data.inlet_vel_direction = torch.matmul(data.inlet_vel_direction, rotation_matrix.T)

    if "3d_ab_wss" in args.tgt_y:
        data.y_wallShearStress = torch.matmul(data.y_wallShearStress, rotation_matrix.T)
        data.y = data.y_wallShearStress
    elif "3d_ab_p" in args.tgt_y:
        data.y = data.y_p
    elif "3d_ab_k" in args.tgt_y:
        data.y = data.y_k
    elif "3d_ab_omega" in args.tgt_y:
        data.y = torch.log(data.y_omega + 1e-8)
    elif "3d_ab_nut" in args.tgt_y:
        data.y = torch.log(data.y_nut + 1e-8)
    elif "3d_snc_press" in args.tgt_y:
        data.conds_feat = torch.ones((data.x.shape[0], 1), device=data.x.device).float()
        data.y = data.y.unsqueeze(-1).float()

    data.conds_feat = data.conds_feat.expand(data.x.shape[0], -1)
    data.inlet_vel_direction = data.inlet_vel_direction.repeat(data.x.shape[0], 1)

    u = data.inlet_vel_direction
    u = u / (torch.norm(u, dim=-1, keepdim=True) + 1e-8)

    flow_proj = (data.x * u).sum(dim=-1, keepdim=True)           # x · u
    r2 = (data.x * data.x).sum(dim=-1, keepdim=True)             # ||x||^2
    cross2 = (r2 - flow_proj.pow(2)).clamp_min(0.0)              # ||x||^2 - (x·u)^2

    data.node_attr = torch.cat(
        [data.conds_feat, flow_proj, r2, cross2],
        dim=-1,
    )
    return data


class LargeDataset(Dataset):
    def __init__(self, dataset_files, basepath, args):
        self.dataset_files = dataset_files
        self.basepath = basepath
        self.args = args

    def __len__(self):
        return len(self.dataset_files)

    def __getitem__(self, idx):
        file_path = self.dataset_files[idx]
        with open(f"{self.basepath}/{file_path}", "rb") as f:
            data = pickle.load(f)
        data = preprocess(data, self.args)
        data.shape_id = file_path[:-4]
        return data


class EQGINO(pl.LightningModule):
    """
    Keep the class name EQGINO so your current main.py loader keeps working.
    The actual operator here is CEqGINO.
    """

    def __init__(self, raw_sample_data, args):
        super().__init__()
        self.args = args
        raw_sample_data = preprocess(raw_sample_data, args)

        coord_dim_node = raw_sample_data.x.shape[1]
        input_dim_node = raw_sample_data.node_attr.shape[1]
        cond_dim = raw_sample_data.conds_feat.shape[1]
        output_dim = raw_sample_data.y.shape[1]

        self.in_channels = input_dim_node
        self.out_channels = self.args.hidden_dim
        self.fno_n_mode = self.args.fno_n_mode
        self.fno_n_modes = (self.fno_n_mode, self.fno_n_mode, self.fno_n_mode)
        self.fno_hidden_channels = self.args.hidden_dim
        self.gno_coord_dim = coord_dim_node
        self.gno_radius = self.args.gno_radius
        self.fno_n_layers = self.args.fno_n_layers
        self.fno_norm = None

        self.operator = CEqGINO(
            in_channels=self.in_channels,
            out_channels=output_dim,
            gno_coord_dim=self.gno_coord_dim,
            gno_radius=self.gno_radius,
            fno_n_modes=self.fno_n_modes,
            fno_hidden_channels=self.fno_hidden_channels,
            fno_n_layers=self.fno_n_layers,
            fno_norm=self.fno_norm,
            fno_in_channels=self.in_channels,
            latent_feature_channels=cond_dim,
            ode_steps=self.args.ode_steps,
            ode_max_neighbors=self.args.ode_max_neighbors,
            interp_scale=self.args.interp_scale,
            anchor_count=self.args.anchor_count,
        )

        self.decoder = Sequential(
            Linear(args.hidden_dim, args.hidden_dim),
            Linear(args.hidden_dim, output_dim),
        )

    def loss(self, pred, inputs):
        labels = inputs.y
        mae = torch.mean(torch.abs(labels - pred))
        error = torch.sum((labels - pred) ** 2, dim=1)
        rmse = torch.sqrt(torch.mean(error))
        r2 = r2_score(labels.detach().cpu().numpy(), pred.detach().cpu().numpy())
        return rmse, mae, r2

    def _maybe_subsample(self, coord, data):
        if self.args.mesh_subsample_rate != 1 and self.trainer.training:
            sampling_rate = 1 / self.args.mesh_subsample_rate
        elif self.args.mesh_subsample_rate_valid != 1 and (self.trainer.validating or self.trainer.testing):
            sampling_rate = 1 / self.args.mesh_subsample_rate_valid
        else:
            return coord, data

        n_nodes = coord.size(0)
        n_samples = int(n_nodes * sampling_rate)
        idx = torch.randperm(n_nodes, device=coord.device)[:n_samples]

        coord = coord[idx, :]
        data.x = data.x[idx, :]
        data.y = data.y[idx, :]
        data.node_attr = data.node_attr[idx, :]
        data.inlet_vel_direction = data.inlet_vel_direction[idx, :]
        data.conds_feat = data.conds_feat[idx, :]
        if hasattr(data, "batch") and data.batch is not None:
            data.batch = data.batch[idx]

        return coord, data

    def _build_cond_token(self, data):
        return data.conds_feat[:1].unsqueeze(1)   # [1, 1, C]

    def forward(self, data):
        coord = data.x
        coord = coord - coord.mean(dim=0, keepdim=True)
        max_dist = torch.max(torch.norm(coord, dim=1)) + 1e-8
        coord = coord / max_dist

        coord, data = self._maybe_subsample(coord, data)
        cond_token = self._build_cond_token(data).to(coord.device)
        flow_dir = data.inlet_vel_direction[:1].to(coord.device)  # [1, 3]

        pred = self.operator(
            input_geom=coord.unsqueeze(0),
            latent_queries=None,
            latent_features=cond_token,
            output_queries=coord.unsqueeze(0),
            x=data.node_attr.unsqueeze(0),
            flow_dir=flow_dir,
            batch_index=None,
        ).squeeze(0)

        if pred.shape[1] == 3:
            pred = apply_radial_basis(pred, data.x, data.inlet_vel_direction)

        return pred, []

    def training_step(self, batch, batch_idx):
        pred, _ = self(batch)
        rmse, mae, r2 = self.loss(pred, batch)
        self.log("Train RMSE", rmse, prog_bar=True, batch_size=1)
        self.log("Train MAE", mae, prog_bar=True, batch_size=1)
        self.log("Train R2", r2, prog_bar=True, batch_size=1)
        return mae

    def validation_step(self, batch, batch_idx):
        pred, _ = self(batch)
        batch_index = batch.batch if hasattr(batch, "batch") else None
        metrics = compute_metrics(pred, batch.y, batch_index)
        for k, v in metrics.items():
            self.log(f"Valid {k}", v, prog_bar=True, batch_size=1, sync_dist=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        pred, _ = self(batch)
        batch_index = batch.batch if hasattr(batch, "batch") else None
        metrics = compute_metrics(pred, batch.y, batch_index)
        for k, v in metrics.items():
            self.log(f"Test {k}", v, prog_bar=True, batch_size=1, sync_dist=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return {"optimizer": optimizer}

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        sd = checkpoint.get("state_dict", checkpoint)
        if "_metadata" in sd:
            sd.pop("_metadata")