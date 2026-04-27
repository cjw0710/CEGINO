import copy
import pickle
import torch
import lightning.pytorch as pl
import trimesh

from sklearn.metrics import r2_score
from torch.nn import Linear, Sequential
from torch.utils.data import Dataset

from neuralop.models.eqgino import EqGINO
from utils import (
    apply_radial_basis,
    compute_metrics,
    get_rotation_matrix,
    relative_equivariance_error,
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

    # 兼容 AhmedBody / ShapeNetCar 的条件特征逻辑
    if not hasattr(data, "conds_feat") or data.conds_feat is None:
        data.conds_feat = torch.ones((1, 1), device=data.x.device).float()

    data.conds_feat = data.conds_feat.expand(data.x.shape[0], -1)
    data.inlet_vel_direction = data.inlet_vel_direction.repeat(data.x.shape[0], 1)
    data.node_attr = data.conds_feat

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

        self.operator = EqGINO(
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
        )

        # 保留 decoder，兼容你当前仓库风格
        self.decoder = Sequential(
            Linear(args.hidden_dim, args.hidden_dim),
            Linear(args.hidden_dim, output_dim),
        )

    def generate_bounding_latent_queries(self, grid_size):
        x = torch.linspace(-1, 1, grid_size[0], device=self.device)
        y = torch.linspace(-1, 1, grid_size[1], device=self.device)
        z = torch.linspace(-1, 1, grid_size[2], device=self.device)
        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
        latent_queries = torch.stack((X, Y, Z), dim=-1)
        return latent_queries.unsqueeze(0)

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
        subsampled_node_idx = torch.randperm(n_nodes, device=coord.device)[:n_samples]

        coord = coord[subsampled_node_idx, :]
        data.x = data.x[subsampled_node_idx, :]
        data.y = data.y[subsampled_node_idx, :]
        data.node_attr = data.node_attr[subsampled_node_idx, :]
        data.inlet_vel_direction = data.inlet_vel_direction[subsampled_node_idx, :]
        data.conds_feat = data.conds_feat[subsampled_node_idx, :]

        if hasattr(data, "batch") and data.batch is not None:
            data.batch = data.batch[subsampled_node_idx]

        return coord, data

    def _build_latent_features(self, data, latent_queries):
        cond_feat = data.conds_feat[:1]
        latent_feature_dim = cond_feat.shape[1]
        batch_size = latent_queries.shape[0]

        latent_features = cond_feat.view(batch_size, 1, 1, 1, latent_feature_dim)
        latent_features = latent_features.expand(
            batch_size, *latent_queries.shape[1:-1], latent_feature_dim
        )
        return latent_features

    def _equivariance_penalty(self, batch, pred):
        """
        Lightweight continuous-rotation regularizer.

        Key tricks:
        1. detach pred on the target side
        2. use a stronger subsample for the rotated branch
        """
        rotation = get_rotation_matrix("arbitrary", device=batch.x.device)

        batch_rot = copy.deepcopy(batch)
        batch_rot.x = torch.matmul((batch.x - batch.x.mean(dim=0, keepdim=True)), rotation.T)
        batch_rot.inlet_vel_direction = torch.matmul(batch.inlet_vel_direction, rotation.T)

        if hasattr(batch_rot, "node_attr"):
            batch_rot.node_attr = batch.node_attr.clone()
        if hasattr(batch_rot, "conds_feat"):
            batch_rot.conds_feat = batch.conds_feat.clone()
        if hasattr(batch_rot, "y") and batch_rot.y is not None and batch_rot.y.shape[-1] == 3:
            batch_rot.y = torch.matmul(batch.y, rotation.T)

        eq_subsample_rate = getattr(self.args, "eq_loss_subsample_rate", 4)
        idx = None
        if eq_subsample_rate > 1:
            n_nodes = batch_rot.x.size(0)
            n_samples = max(512, int(n_nodes / eq_subsample_rate))
            idx = torch.randperm(n_nodes, device=batch_rot.x.device)[:n_samples]

            batch_rot.x = batch_rot.x[idx]
            batch_rot.y = batch_rot.y[idx]
            batch_rot.node_attr = batch_rot.node_attr[idx]
            batch_rot.inlet_vel_direction = batch_rot.inlet_vel_direction[idx]
            batch_rot.conds_feat = batch_rot.conds_feat[idx]

            if hasattr(batch_rot, "batch") and batch_rot.batch is not None:
                batch_rot.batch = batch_rot.batch[idx]

        pred_rot, _ = self(batch_rot)

        if pred.shape[-1] == 3:
            target_rot = torch.matmul(pred.detach(), rotation.T)
        else:
            target_rot = pred.detach()

        if idx is not None:
            target_rot = target_rot[idx]

        eq_loss = torch.sqrt(torch.mean((pred_rot - target_rot) ** 2))
        return eq_loss

    def _log_equivariance_metric(self, batch, pred, split="Valid"):
        if not getattr(self.args, "use_equivariance_metric", False):
            return

        rotation = get_rotation_matrix("arbitrary", device=batch.x.device)

        batch_rot = copy.deepcopy(batch)
        batch_rot.x = torch.matmul((batch.x - batch.x.mean(dim=0, keepdim=True)), rotation.T)
        batch_rot.inlet_vel_direction = torch.matmul(batch.inlet_vel_direction, rotation.T)

        if hasattr(batch_rot, "node_attr"):
            batch_rot.node_attr = batch.node_attr.clone()
        if hasattr(batch_rot, "conds_feat"):
            batch_rot.conds_feat = batch.conds_feat.clone()
        if hasattr(batch_rot, "y") and batch_rot.y is not None and batch_rot.y.shape[-1] == 3:
            batch_rot.y = torch.matmul(batch.y, rotation.T)

        pred_rot, _ = self(batch_rot)
        eq_err = relative_equivariance_error(pred, pred_rot, rotation)

        self.log(
            f"{split} EqErr",
            eq_err,
            prog_bar=True,
            batch_size=1,
            sync_dist=True,
            on_epoch=True,
        )

    def forward(self, data):
        coord = data.x
        coord = coord - coord.mean(dim=0, keepdim=True)
        max_dist = torch.max(torch.norm(coord, dim=1)) + 1e-8
        coord = coord / max_dist

        coord, data = self._maybe_subsample(coord, data)

        latent_queries = self.generate_bounding_latent_queries(
            (self.fno_n_mode, self.fno_n_mode, self.fno_n_mode)
        ).to(coord.device)

        latent_features = self._build_latent_features(data, latent_queries).to(coord.device)

        pred = self.operator(
            input_geom=coord.unsqueeze(0),
            latent_queries=latent_queries,
            latent_features=latent_features,
            output_queries=coord,
            x=data.node_attr.unsqueeze(0),
        ).squeeze(0)

        if pred.shape[1] == 3:
            pred = apply_radial_basis(pred, data.x, data.inlet_vel_direction)

        return pred, []

    def training_step(self, batch, batch_idx):
        pred, _ = self(batch)
        rmse, mae, r2 = self.loss(pred, batch)

        loss = mae
        eq_loss_weight = getattr(self.args, "eq_loss_weight", 0.0)
        if eq_loss_weight > 0:
            eq_loss = self._equivariance_penalty(batch, pred)
            loss = loss + eq_loss_weight * eq_loss
            self.log("Train EqLoss", eq_loss, prog_bar=True, batch_size=1)

        self.log("Train RMSE", rmse, prog_bar=True, batch_size=1)
        self.log("Train MAE", mae, prog_bar=True, batch_size=1)
        self.log("Train R2", r2, prog_bar=True, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
        pred, _ = self(batch)
        batch_index = batch.batch if hasattr(batch, "batch") else None
        metrics = compute_metrics(pred, batch.y, batch_index)

        for k, v in metrics.items():
            self.log(f"Valid {k}", v, prog_bar=True, batch_size=1, sync_dist=True, on_epoch=True)

        self._log_equivariance_metric(batch, pred, split="Valid")

    def test_step(self, batch, batch_idx):
        pred, _ = self(batch)
        batch_index = batch.batch if hasattr(batch, "batch") else None
        metrics = compute_metrics(pred, batch.y, batch_index)

        for k, v in metrics.items():
            self.log(f"Test {k}", v, prog_bar=True, batch_size=1, sync_dist=True, on_epoch=True)

        self._log_equivariance_metric(batch, pred, split="Test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return {"optimizer": optimizer}

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        sd = checkpoint.get("state_dict", checkpoint)
        if "_metadata" in sd:
            sd.pop("_metadata")