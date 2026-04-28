# CEGINO: Continuous Equivariant Geometry-Informed Neural Operators for 3D PDEs

This repository provides the implementation of **CEGINO** for learning solution operators of 3D PDEs on irregular 3D geometries.

---

## Environment

The code has been tested with the following environment:

```text
Python 3.10
CUDA 12.8
PyTorch 2.11.0+cu128
PyTorch Geometric
Lightning
scikit-learn
trimesh
wandb
tensorly
```

Create a conda environment:

```bash
conda create -n cegino python=3.10 -y
conda activate cegino
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install PyTorch Geometric and required extensions:

```bash
pip install torch_geometric
pip install --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
```

Install other dependencies:

```bash
pip install lightning scikit-learn trimesh wandb tensorly
```

---

## Data Preparation

Place datasets under the `data/` directory:

```text
CEGINO/
└── data/
    ├── ahmedbody/
    ├── shapenetcar/
    └── DeepJEB/
```

AhmedBody and ShapeNetCar should follow the processed format used by the training scripts.

DeepJEB should be downloaded from the official DeepJEB dataset source. After downloading and extracting the dataset, place it under:

```text
data/DeepJEB/
```

The expected DeepJEB folder structure is:

```text
data/DeepJEB/
├── Field/
├── FieldMesh/
├── Scalar/
├── SurfaceMesh/
└── Metadata/
```

If the official dataset provides additional folders such as `VolumeMesh`, keep them under the same `DeepJEB` directory.

Example of a correct layout:

```text
data/DeepJEB/FieldMesh/6_247.h5
data/DeepJEB/FieldMesh/6_351.h5
```

Avoid nested duplicated folders such as:

```text
data/DeepJEB/FieldMesh/FieldMesh/6_247.h5
```

---

## Running CEGINO

### AhmedBody Pressure

```bash
python main.py \
  --model ceqgino \
  --data_fname ahmedbody \
  --tgt_y 3d_ab_p \
  --aug_type arbitrary \
  --hidden_dim 128 \
  --gno_radius 0.08 \
  --ode_steps 4 \
  --ode_max_neighbors 24 \
  --interp_scale 0.05 \
  --anchor_count 1024 \
  --mesh_subsample_rate_valid 4 \
  --max_epochs 100 \
  --run_tag cegino_pressure
```

For full-resolution validation and testing, use:

```bash
--mesh_subsample_rate_valid 1
```

Example full-resolution command:

```bash
python main.py \
  --model ceqgino \
  --data_fname ahmedbody \
  --tgt_y 3d_ab_p \
  --aug_type arbitrary \
  --hidden_dim 128 \
  --gno_radius 0.08 \
  --ode_steps 4 \
  --ode_max_neighbors 24 \
  --interp_scale 0.05 \
  --anchor_count 1024 \
  --mesh_subsample_rate_valid 1 \
  --max_epochs 100 \
  --run_tag cegino_pressure_fullres
```

---

### AhmedBody Wall Shear Stress

```bash
python main.py \
  --model ceqgino \
  --data_fname ahmedbody \
  --tgt_y 3d_ab_wss \
  --aug_type arbitrary \
  --hidden_dim 128 \
  --gno_radius 0.08 \
  --ode_steps 4 \
  --ode_max_neighbors 24 \
  --interp_scale 0.05 \
  --anchor_count 1024 \
  --mesh_subsample_rate_valid 4 \
  --max_epochs 100 \
  --run_tag cegino_wss
```

---

### AhmedBody Other Targets

AhmedBody supports the following targets:

```text
3d_ab_p       pressure
3d_ab_wss     wall shear stress
3d_ab_k       turbulent kinetic energy
3d_ab_omega   omega
3d_ab_nut     turbulent viscosity
```

Example command for turbulent kinetic energy:

```bash
python main.py \
  --model ceqgino \
  --data_fname ahmedbody \
  --tgt_y 3d_ab_k \
  --aug_type arbitrary \
  --hidden_dim 128 \
  --gno_radius 0.08 \
  --ode_steps 4 \
  --ode_max_neighbors 24 \
  --interp_scale 0.05 \
  --anchor_count 1024 \
  --mesh_subsample_rate_valid 4 \
  --max_epochs 100 \
  --run_tag cegino_k
```

---

## Runtime Notes

To disable W&B logging on Linux or macOS:

```bash
export WANDB_MODE=disabled
```

On Windows CMD:

```bat
set WANDB_MODE=disabled
set WANDB_DISABLED=true
set PYTHONUTF8=1
```

Use `--max_epochs` to control the number of training epochs:

```bash
--max_epochs 100
```

Use `--run_tag` to name the run:

```bash
--run_tag my_experiment_name
```

---

## Important Files

```text
main.py
model/ceqgino.py
neuralop/models/ceqgino.py
neuralop/layers/equivariant_latent_ode.py
utils.py
```

---

## Citation

If you use this code, please cite our paper.
