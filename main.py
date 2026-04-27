import os
import copy
import pickle
import random
import importlib
import argparse

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch_geometric.loader import DataLoader as PyGDataLoader

torch.set_float32_matmul_precision("high")


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    L.seed_everything(seed, workers=True)


class CFDDataModule(L.LightningDataModule):
    def __init__(self, train_dataset, valid_dataset, test_dataset, batch_size=1, num_workers=0):
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return PyGDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=False if self.num_workers == 0 else True,
        )

    def val_dataloader(self):
        return PyGDataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=False if self.num_workers == 0 else True,
        )

    def test_dataloader(self):
        return PyGDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=False if self.num_workers == 0 else True,
        )


def main(args):
    seed_everything(args.seed)

    print("    # ========================================================================================== #")
    print(f"    # Model: {args.model}")
    print(f"    # Summary: {args.model}_{args.data_fname}")
    print(f"    # Current Time: {args.run_tag}")
    print(f"    # Dataset: {args.data_fname}")
    print(f"    # Seed: {args.seed}, Total: 1/1")
    print("    # ========================================================================================== #")

    module = importlib.import_module(f"model.{args.model}")
    model_class = getattr(module, "EQGINO")
    dataset_class = getattr(module, "LargeDataset")

    train_basepath = os.path.join("./data", args.data_fname, "train")
    test_basepath = os.path.join("./data", args.data_fname, "test")

    train_data_list = sorted(os.listdir(train_basepath))
    test_data_list = sorted(os.listdir(test_basepath))

    # 当前代码约定：train 目录中前 413 用作 train，后 45 用作 valid
    # 对 AhmedBody 这和论文里的 413 train + 45 valid 正好对齐
    if args.data_fname == "ahmedbody":
        split_idx = 413
        train_files = train_data_list[:split_idx]
        valid_files = train_data_list[split_idx:]
    else:
        # 对其他数据，默认 9:1 划分
        split_idx = int(len(train_data_list) * 0.9)
        train_files = train_data_list[:split_idx]
        valid_files = train_data_list[split_idx:]

    train_dataset = dataset_class(train_files, train_basepath, args)
    valid_dataset = dataset_class(valid_files, train_basepath, args)
    test_dataset = dataset_class(test_data_list, test_basepath, args)

    # 取一个样本初始化模型
    with open(os.path.join(train_basepath, train_files[0]), "rb") as f:
        sample_data = pickle.load(f)

    model = model_class(sample_data, args)

    datamodule = CFDDataModule(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    logger = WandbLogger(
        project=args.wandb_project,
        name=f"{args.model}_{args.data_fname}_{args.run_tag}",
        offline=(os.environ.get("WANDB_MODE", "").lower() in ["disabled", "offline"]),
    )

    ckpt_dir = os.path.join("checkpoints", f"{args.model}_{args.data_fname}_{args.run_tag}")
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_top_k=1,
        monitor="Valid Rel_L2",
        mode="min",
        filename="{epoch:02d}-{Valid_Rel_L2:.4f}",
        auto_insert_metric_name=False,
    )

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices if torch.cuda.is_available() else 1,
        strategy="auto" if (not torch.cuda.is_available() or args.devices == 1) else "ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=args.max_epochs,
        log_every_n_steps=1,
        enable_progress_bar=True,
        num_sanity_val_steps=2,
        default_root_dir=".",
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument("--model", type=str, default="eqgino", choices=["eqgino", "harmonicgino", "ceqgino"])
    parser.add_argument("--data_fname", type=str, default="ahmedbody")
    parser.add_argument("--tgt_y", type=str, default="3d_ab_p")
    parser.add_argument("--aug_type", type=str, default="arbitrary")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_tag", type=str, default="0414-eqgino")
    parser.add_argument("--wandb_project", type=str, default="eqgino")
    parser.add_argument("--use_equivariance_metric", action="store_true")

    # optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=100)

    # data loading
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--devices", type=int, default=1)

    # model size / operator
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--fno_n_mode", type=int, default=32)
    parser.add_argument("--fno_n_layers", type=int, default=4)
    parser.add_argument("--gno_radius", type=float, default=0.1)

    # mesh subsampling
    parser.add_argument("--mesh_subsample_rate", type=int, default=1)
    parser.add_argument("--mesh_subsample_rate_valid", type=int, default=1)

    # harmonic params
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--radial_bins", type=int, default=16)
    parser.add_argument("--ode_steps", type=int, default=4)
    
    parser.add_argument("--anchor_count", type=int, default=512)
    parser.add_argument("--ode_max_neighbors", type=int, default=64)
    parser.add_argument("--interp_scale", type=float, default=0.05)
    # new: weak equivariance regularizer
    parser.add_argument("--eq_loss_weight", type=float, default=0.0)
    parser.add_argument("--eq_loss_subsample_rate", type=int, default=4)    
    args = parser.parse_args()
    main(args)