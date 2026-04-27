# CEGINO: Continuous-Equivariant Geometry-Informed Neural Operators for 3D PDEs

CEGINO is a continuous-equivariant geometry-informed neural operator for surrogate modeling of 3D PDE fields on irregular 3D geometries.

The model is designed to learn PDE solution operators on complex meshes while improving robustness under geometric transformations. It uses a latent anchor representation and continuous invariant dynamics to support scalar and vector field prediction on irregular geometries.

The current implementation includes:

- invariant lifting from irregular mesh nodes to latent anchor points;
- flow-aware continuous latent dynamics;
- invariant cross-decoding from anchors back to query points;
- support for scalar targets such as pressure;
- preliminary support for vector targets such as wall shear stress;
- planned support for DeepJEB deflection and stress prediction.

The main tested dataset at the current stage is **AhmedBody**. DeepJEB support is being prepared after downloading and organizing the full dataset.

---

## 1. Repository Structure

```text
CEGINO/
├── main.py
├── utils.py
├── model/
│   ├── ceqgino.py
│   └── harmonicgino.py
├── neuralop/
│   ├── models/
│   │   ├── ceqgino.py
│   │   ├── harmonicgino.py
│   │   └── __init__.py
│   └── layers/
│       ├── equivariant_latent_ode.py
│       ├── harmonic_spectral_convolution.py
│       ├── fno_block.py
│       ├── spectral_convolution.py
│       └── ...
├── data/
│   ├── ahmedbody/
│   └── DeepJEB/
│       ├── Field/
│       ├── FieldMesh/
│       ├── Scalar/
│       ├── SurfaceMesh/
│       └── Metadata/
├── checkpoints/
├── best_models/
├── wandb/
├── logs/
└── README.md
```

Important CEGINO-related files:

```text
model/ceqgino.py
neuralop/models/ceqgino.py
neuralop/layers/equivariant_latent_ode.py
```

---

## 2. Main Model

### CEGINO

Current continuous-equivariant model.

```bash
--model ceqgino
```

CEGINO uses:

```text
irregular mesh input
latent anchor points
continuous invariant feature dynamics
flow-aware invariant messages
anchor-to-query invariant decoding
```

---

## 3. Environment Setup

### 3.1 Recommended Windows Environment

The current local Windows environment has been tested with:

```text
OS: Windows
GPU: RTX 5090 32GB
RAM: 96GB
CUDA: 12.8
Python: 3.10
PyTorch: 2.11.0+cu128
```

Create a conda environment:

```bat
conda create -n cegino python=3.10 -y
conda activate cegino
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch with CUDA 12.8:

```bat
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install PyTorch Geometric:

```bat
pip install torch_geometric
pip install --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
```

`torch_spline_conv` is not required for the current CEGINO path.

Install other dependencies:

```bat
pip install lightning scikit-learn trimesh wandb tensorly
```

If PyG extensions fail with `WinError 127`, install the latest Microsoft Visual C++ Redistributable x64.

---

## 4. Environment Check

Run:

```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
python -c "import torch_geometric; print('pyg ok')"
python -c "import torch_cluster; print('cluster ok')"
python -c "import torch_scatter; print('scatter ok')"
python -c "import torch_sparse; print('sparse ok')"
```

Expected PyTorch output:

```text
2.11.0+cu128 True 12.8
```

---

## 5. Windows Runtime Settings

Before running training on Windows, set:

```bat
conda activate cegino
set WANDB_MODE=disabled
set WANDB_DISABLED=true
set PYTHONUTF8=1
set PYTORCH_CUDA_ALLOC_CONF=
```

Do **not** set the following on Windows:

```bat
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

It may be unsupported and can cause CUDA initialization errors.

---

## 6. Data Layout

### 6.0 Dataset Download

The anonymized processed dataset package is available on Zenodo:

```text
DOI: https://doi.org/10.5281/zenodo.19809778
Record page: https://zenodo.org/records/19809778
```

Download `data.zip` from the Zenodo record and extract it under the project root so that the dataset folders follow the layout below:

```text
CEGINO/
└── data/
    ├── ahmedbody/
    ├── DeepJEB/
    └── shapenetcar/
```

### 6.1 AhmedBody

Expected directory:

```text
data/
└── ahmedbody/
    └── ...
```

Current supported targets:

```text
3d_ab_p       pressure
3d_ab_wss     wall shear stress
3d_ab_k       turbulent kinetic energy
3d_ab_omega   omega
3d_ab_nut     turbulent viscosity
```

---

### 6.2 DeepJEB

Expected directory after downloading and extracting or syncing:

```text
data/
└── DeepJEB/
    ├── Field/
    ├── FieldMesh/
    ├── Scalar/
    ├── SurfaceMesh/
    └── Metadata/
```

Recommended folder meanings:

```text
Field       likely deflection / vector-field data
Scalar      likely stress / scalar-field data
FieldMesh   mesh corresponding to field data
SurfaceMesh surface geometry for visualization
Metadata    sample metadata and split information
```

If stress or other fields require volume meshes, also add:

```text
data/
└── DeepJEB/
    └── VolumeMesh/
```

Recommended download method:

```text
Use Google Drive for desktop sync instead of browser folder download.
```

Browser folder downloads may generate many temporary zip chunks such as:

```text
FieldMesh-20260423T133656Z-3-027.zip
FieldMesh-20260423T133656Z-3-032.zip
```

These are temporary Google Drive packaging files, not original dataset files. Syncing directly downloads the original `.h5` files and is much more stable.

---

## 7. Training Commands

### 7.1 AhmedBody Pressure

Current stable CEGINO pressure command:

```bat
python main.py --model ceqgino --data_fname ahmedbody --tgt_y 3d_ab_p --aug_type arbitrary --hidden_dim 128 --gno_radius 0.08 --ode_steps 4 --ode_max_neighbors 24 --interp_scale 0.05 --anchor_count 1024 --mesh_subsample_rate_valid 4 --max_epochs 100 --run_tag win_pressure_h128_a1024
```

Full-resolution validation / test version:

```bat
python main.py --model ceqgino --data_fname ahmedbody --tgt_y 3d_ab_p --aug_type arbitrary --hidden_dim 128 --gno_radius 0.08 --ode_steps 4 --ode_max_neighbors 24 --interp_scale 0.05 --anchor_count 1024 --mesh_subsample_rate_valid 1 --max_epochs 100 --run_tag win_pressure_h128_a1024_fullres
```

---

### 7.2 AhmedBody Wall Shear Stress

WSS is a vector field and is harder than pressure.

Baseline WSS command:

```bat
python main.py --model ceqgino --data_fname ahmedbody --tgt_y 3d_ab_wss --aug_type arbitrary --hidden_dim 128 --gno_radius 0.08 --ode_steps 4 --ode_max_neighbors 24 --interp_scale 0.05 --anchor_count 1024 --mesh_subsample_rate_valid 4 --max_epochs 100 --run_tag win_wss_h128_a1024
```

Larger WSS model:

```bat
python main.py --model ceqgino --data_fname ahmedbody --tgt_y 3d_ab_wss --aug_type arbitrary --hidden_dim 192 --gno_radius 0.08 --ode_steps 4 --ode_max_neighbors 24 --interp_scale 0.05 --anchor_count 1024 --mesh_subsample_rate_valid 4 --max_epochs 100 --run_tag win_wss_h192_a1024
```

Even larger WSS model:

```bat
python main.py --model ceqgino --data_fname ahmedbody --tgt_y 3d_ab_wss --aug_type arbitrary --hidden_dim 256 --gno_radius 0.08 --ode_steps 4 --ode_max_neighbors 24 --interp_scale 0.05 --anchor_count 1024 --mesh_subsample_rate_valid 4 --max_epochs 100 --run_tag win_wss_h256_a1024
```

---

## 8. Background Training on Windows

Use:

```bat
start /b cmd /c "set WANDB_MODE=disabled && set WANDB_DISABLED=true && set PYTHONUTF8=1 && set PYTORCH_CUDA_ALLOC_CONF= && python -u main.py --model ceqgino --data_fname ahmedbody --tgt_y 3d_ab_wss --aug_type arbitrary --hidden_dim 192 --gno_radius 0.08 --ode_steps 4 --ode_max_neighbors 24 --interp_scale 0.05 --anchor_count 1024 --mesh_subsample_rate_valid 4 --max_epochs 100 --run_tag win_wss_h192_a1024 > win_wss_h192_a1024.log 2>&1"
```

View training log:

```bat
powershell -command "Get-Content win_wss_h192_a1024.log -Wait -Tail 80"
```

Show final results:

```bat
powershell -command "Get-Content win_wss_h192_a1024.log -Tail 80"
```

---

## 9. Main Hyperparameters

### 9.1 `hidden_dim`

Controls latent feature width.

Common values:

```text
128   current default
192   recommended for WSS
256   larger-capacity WSS trial
```

Increasing `hidden_dim` increases model capacity. WSS is a vector field and may benefit more from larger hidden dimensions than pressure.

---

### 9.2 `anchor_count`

Number of latent anchor points.

Current stable value:

```text
1024
```

Earlier experiments showed:

```text
512   too low
1024  stable and effective
1536  did not consistently improve performance
```

---

### 9.3 `gno_radius`

Radius for the latent graph.

Stable value:

```text
0.08
```

Larger values such as `0.10` did not consistently improve pressure results.

---

### 9.4 `ode_steps`

Number of latent ODE integration steps.

Stable value:

```text
4
```

Increasing to `6` did not consistently improve pressure results.

---

### 9.5 `ode_max_neighbors`

Maximum number of neighbors in the latent ODE graph.

Stable value:

```text
24
```

---

### 9.6 `mesh_subsample_rate_valid`

Validation / test subsampling rate.

For fast development:

```text
--mesh_subsample_rate_valid 4
```

For final evaluation:

```text
--mesh_subsample_rate_valid 1
```

---

## 10. Evaluation Protocols

The project supports three common evaluation settings.

### 10.1 In-distribution

```text
Train: Canonical
Test: Canonical
```

### 10.2 Zero-shot rotated test

```text
Train: Canonical
Test: Rotated
```

### 10.3 Continuous rotated setting

```text
Train: Continuously rotated
Test: Continuously rotated
```

CEGINO should be evaluated under both canonical and rotated settings because the current implementation contains numerical approximations such as:

```text
anchor sampling
FPS
radius graph construction
top-k local decoding
floating-point numerical effects
vector-field post-processing
```

---

## 11. Visualization Experiments

Qualitative visualization usually includes:

```text
Input geometry
Ground truth field
Prediction
Absolute error
```

For rotated consistency visualization:

```text
Top row: canonical input
Bottom row: rotated input
```

For scalar fields such as pressure:

```text
Error = |Ground Truth - Prediction|
```

For vector fields such as WSS or DeepJEB deflection:

```text
Error = ||Ground Truth - Prediction||
```

Recommended workflow:

```text
1. Load one sample.
2. Run canonical inference.
3. Rotate geometry by 180 degrees.
4. Run rotated inference.
5. Save ground truth, prediction, and error.
6. Repeat per sample.
```

Important:

```text
Do not load the entire dataset into memory for visualization.
Process one sample at a time.
```

Example pseudocode:

```python
for sample_id in sample_ids:
    data = load_one_sample(sample_id)
    pred = model(data)

    data_rot = rotate_sample(data, angle=180)
    pred_rot = model(data_rot)

    save_visualization(data, pred)
    save_visualization(data_rot, pred_rot)

    del data, data_rot, pred, pred_rot
    torch.cuda.empty_cache()
```

---

## 12. Current Known Results

### 12.1 AhmedBody Pressure

Typical Windows result for CEGINO pressure setting:

```text
hidden_dim = 128
anchor_count = 1024
gno_radius = 0.08
ode_steps = 4
ode_max_neighbors = 24

Test Rel_L2 ≈ 0.45
Test RMSE   ≈ 137
Test R2     ≈ 0.72
```

Earlier Linux result for the best continuous-equivariant pressure configuration was approximately:

```text
Test Rel_L2 ≈ 0.44
Test RMSE   ≈ 134
Test R2     ≈ 0.73
```

---

### 12.2 AhmedBody WSS

Initial WSS result with `hidden_dim=128`:

```text
Params      ≈ 1.1M
Test Rel_L2 ≈ 0.478
Test RMSE   ≈ 0.652
Test R2     ≈ 0.536
```

WSS is harder than pressure because it is vector-valued.

Recommended next trials:

```text
hidden_dim = 192
hidden_dim = 256
```

---

## 13. Troubleshooting

### 13.1 `ModuleNotFoundError: No module named 'tensorly'`

Install:

```bat
pip install tensorly
```

---

### 13.2 `torch_cluster` or other PyG extensions fail

Install matching PyG wheels:

```bat
pip install --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
```

If it still fails with `WinError 127`, install Microsoft Visual C++ Redistributable x64.

---

### 13.3 `torch_spline_conv` fails to install

This is usually fine for the current CEGINO path.

Do not install it unless needed.

---

### 13.4 `wandb` UnicodeDecodeError on Windows

Set:

```bat
set WANDB_MODE=disabled
set WANDB_DISABLED=true
set PYTHONUTF8=1
```

---

### 13.5 CUDA config error

If you see:

```text
ValueError: Expected 'True' or 'False' ... got 'Truev'
```

clear the variable:

```bat
set PYTORCH_CUDA_ALLOC_CONF=
```

Then rerun.

---

### 13.6 `--epochs` not recognized

This version uses:

```text
--max_epochs
```

not:

```text
--epochs
```

Use:

```bat
--max_epochs 100
```

---

### 13.7 `--model ceqgino` not recognized

Update `main.py` model choices to include:

```python
"ceqgino"
```

Valid choices should include:

```python
["ceqgino", "harmonicgino"]
```

---

### 13.8 `raw_sample_data is None`

If you see:

```text
AttributeError: 'NoneType' object has no attribute 'x'
```

check:

```text
data path
dataset name
sample file loading
whether the selected dataset split contains valid samples
whether data_fname matches the folder name
```

---

## 14. DeepJEB Preparation

Recommended final layout:

```text
D:\CEGINO\data\DeepJEB\
  Field\
  FieldMesh\
  Scalar\
  SurfaceMesh\
  Metadata\
```

Each folder should directly contain dataset files.

Good:

```text
D:\CEGINO\data\DeepJEB\FieldMesh\6_247.h5
D:\CEGINO\data\DeepJEB\FieldMesh\6_351.h5
```

Bad:

```text
D:\CEGINO\data\DeepJEB\FieldMesh\FieldMesh\6_247.h5
```

Check file counts:

```bat
dir D:\CEGINO\data\DeepJEB\FieldMesh\*.h5 /b | find /c /v ""
dir D:\CEGINO\data\DeepJEB\Field\*.h5 /b | find /c /v ""
dir D:\CEGINO\data\DeepJEB\Scalar\*.h5 /b | find /c /v ""
dir D:\CEGINO\data\DeepJEB\SurfaceMesh\*.h5 /b | find /c /v ""
```

Check metadata:

```bat
dir D:\CEGINO\data\DeepJEB\Metadata
```

If Google Drive browser download fails, use Google Drive for desktop and mark folders as:

```text
Available offline
```

or:

```text
可离线使用
```

---

## 15. Recommended Next Experiments

### 15.1 Pressure

Pressure is near a local plateau. Recommended next steps:

```text
1. Full-resolution testing.
2. Canonical vs rotated evaluation.
3. Better lift / decoder architecture.
4. Avoid simply increasing anchor_count beyond 1024 unless necessary.
```

---

### 15.2 WSS

WSS still has room for improvement.

Recommended order:

```text
1. hidden_dim = 192
2. hidden_dim = 256
3. evaluate vector-field rotation consistency
4. improve vector decoder or local basis handling
```

Recommended command:

```bat
python main.py --model ceqgino --data_fname ahmedbody --tgt_y 3d_ab_wss --aug_type arbitrary --hidden_dim 192 --gno_radius 0.08 --ode_steps 4 --ode_max_neighbors 24 --interp_scale 0.05 --anchor_count 1024 --mesh_subsample_rate_valid 4 --max_epochs 100 --run_tag win_wss_h192_a1024
```

---

### 15.3 DeepJEB

After the dataset is ready:

```text
1. Verify Field / Scalar / FieldMesh / SurfaceMesh / Metadata structure.
2. Write DeepJEB dataset loader.
3. Run deflection target.
4. Run stress target.
5. Generate qualitative canonical-vs-rotated visualizations.
```

---

## 16. Notes on Continuous Equivariance

CEGINO is designed to improve prediction consistency under 3D rotations and coordinate changes.

The current model uses invariant quantities such as distances and flow-aware geometric relations to construct latent dynamics and decoding. This makes it suitable for testing rotation consistency and continuous transformation generalization on irregular 3D geometries.

The current implementation still contains numerical approximations, so canonical and rotated evaluations should both be reported.

---

## 17. Hardware Notes

Current local hardware:

```text
RAM  96GB
VRAM 32GB
GPU  RTX 5090
```

This is sufficient for:

```text
AhmedBody pressure training
AhmedBody WSS training
single-sample visualization
canonical-vs-rotated qualitative figures
DeepJEB visualization after data is ready
```

For visualization, process one sample at a time.

---

## 18. Status

Current status:

```text
AhmedBody pressure: working
AhmedBody WSS: working, needs stronger model
DeepJEB: dataset preparation in progress
Visualization: planned after dataset organization
```

CEGINO is an active experimental branch for continuous-equivariant operator learning on irregular 3D geometries.
