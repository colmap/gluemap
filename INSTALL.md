# Installation

GLUEMAP is a Python package with a C++/pybind11 extension (`pygluemap`)
that links against Ceres, Eigen, Boost, and OpenMP. It also depends on
several feed-forward models that are vendored as git submodules.

## 1. Clone with submodules

```
git clone https://github.com/colmap/gluemap.git
cd gluemap
git submodule update --init --recursive
```

The submodules under `thirdparty/` (pi3, vggt, mapanything,
doppelgangers-plusplus, salad) are required at runtime — they are added
to `sys.path` by `thirdparty/path_to_thirdparty.py`. Skipping this step
will cause `ImportError` on first run.

## 2. Install C/C++ build dependencies

You need development headers + CMake configs for:

- CMake ≥ 3.18
- A C++17 compiler
- Eigen3
- Ceres (built with CUDA — see note below)
- METIS (Ceres pulls it in transitively)
- Boost (headers only)
- OpenMP
- pybind11 ≥ 3.0 (also fetched automatically as a build dep, but it's
  fine to have it in the env)

### Option A — conda / micromamba (recommended)

Versions below are pinned to the known-good `gluemap` env so installs
stay reproducible alongside the pinned Python deps in
`pyproject.toml`. GLUEMAP requires CUDA at runtime — the GPU PyTorch
build is the only supported configuration:

```
micromamba install -n <env> -c conda-forge \
    eigen=3.4.0 \
    ceres-solver=2.2.0 \
    metis=5.1.0 \
    boost=1.85.0 \
    libstdcxx-ng=15.2.0 \
    pytorch-gpu=2.4.1 \
    torchvision=0.19.1 \
    cuda-version=12.4
```

Installing PyTorch from conda-forge (rather than pip) is strongly
recommended: it links against the same `libstdc++` as conda-forge's
Ceres, which avoids a runtime ABI clash where `import torch` loads the
older system `libstdc++` first and then `import pygluemap` fails with
``version `CXXABI_1.3.15' not found``. See the troubleshooting section
if you hit this.

If you want CMake to discover the C++ deps without setting
`CMAKE_PREFIX_PATH` manually, also install the conda compilers (their
activate scripts export `CMAKE_PREFIX_PATH=$CONDA_PREFIX`):

```
micromamba install -n <env> -c conda-forge compilers
```

### Option B — system packages (Ubuntu/Debian)

```
sudo apt install \
    cmake \
    build-essential \
    libeigen3-dev \
    libceres-dev \
    libmetis-dev \
    libboost-dev \
    pybind11-dev
```

### Note on Ceres + CUDA

`gluemap/pybind/CMakeLists.txt` defines `CERES_HAS_CUDA`, so the Ceres
build you link against needs CUDA support. The default conda-forge
`ceres-solver` is not CUDA-enabled; if you need CUDA, build Ceres from
source against your CUDA toolkit and point `CMAKE_PREFIX_PATH` at it.

## 3. Install GLUEMAP

From the repo root, in your active Python ≥ 3.10 environment:

```
CMAKE_PREFIX_PATH=$CONDA_PREFIX pip install -e .
```

The `CMAKE_PREFIX_PATH=$CONDA_PREFIX` prefix is needed when the conda
compilers are not installed — Ceres' bundled `FindMETIS.cmake` only
honours `CMAKE_PREFIX_PATH`/system paths, so without this it can't
locate METIS even when it's installed in the env.

For a non-editable install drop the `-e`.

### Optional extras

```
pip install -e ".[dev]"   # ruff + pytest
```

## 4. Download model checkpoints

The demo configs (`configs/base.yaml`) expect four checkpoints under
`checkpoints/`. They are not bundled — download them from each model's
upstream release:

| Config key | File | Source |
|---|---|---|
| `path_feedforward` | `pi3.safetensors` | HF `yyfz233/Pi3` |
| `path_retrieval` | `dino_salad.ckpt` | github.com/serizba/salad release `v1.0.0` |
| `path_tracker` | `vggsfm_v2_0_0_track_predictor.bin` | HF `facebook/VGGSfM` (`vggsfm_v2_tracker.pt`, renamed) |
| `path_dg` | `checkpoint-dg+visym.pth` | HF `doppelgangers25/doppelgangers_plusplus` |

Quickest setup from the repo root:

```
mkdir -p checkpoints

# SALAD retrieval
wget -O checkpoints/dino_salad.ckpt \
    https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt

# VGGSfM tracker (renamed to match base.yaml)
wget -O checkpoints/vggsfm_v2_0_0_track_predictor.bin \
    https://huggingface.co/facebook/VGGSfM/resolve/main/vggsfm_v2_tracker.pt

# Pi3 multiview model (huggingface_hub already in env)
hf download yyfz233/Pi3 model.safetensors --local-dir checkpoints
mv checkpoints/model.safetensors checkpoints/pi3.safetensors

# Doppelgangers++
hf download doppelgangers25/doppelgangers_plusplus \
    checkpoint-dg+visym.pth --local-dir checkpoints
```

If you switch `chosen_model` to `vggt` or `map_anything`, fetch the
matching backbone instead of `pi3.safetensors`:

- VGGT: `https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt`
- MapAnything: set `path_feedforward: facebook/map-anything` (it's a HF
  repo ID, not a file path — `MapAnything.from_pretrained` resolves it).

## 5. Verify

```
python -c "import gluemap; import pygluemap; print(pygluemap.__file__)"
gluemap-demo --help
gluemap-demo-lamar --help
gluemap-benchmark --help
pytest tests/
```

## Troubleshooting

- **`Could NOT find Eigen3 / Ceres`** — install the dev packages (step 2)
  and make sure CMake can see them: `CMAKE_PREFIX_PATH=$CONDA_PREFIX`
  or the equivalent install prefix.
- **`Could NOT find METIS (missing: METIS_INCLUDE_DIR METIS_LIBRARY)`** —
  METIS is installed but `CMAKE_PREFIX_PATH` is not set. Re-run with
  `CMAKE_PREFIX_PATH=$CONDA_PREFIX pip install -e .`, or pass hints:
  `CMAKE_ARGS="-DMETIS_INCLUDE_DIR=$CONDA_PREFIX/include -DMETIS_LIBRARY=$CONDA_PREFIX/lib/libmetis.so" pip install -e .`.
- **`No matching distribution found for lightglue`** — `lightglue` is
  installed from `git+https://github.com/cvg/LightGlue.git` (declared
  in `pyproject.toml`). Make sure pip can reach GitHub.
- **`ImportError: pi3 / vggt / mapanything is not initialized`** — you
  forgot `git submodule update --init --recursive`.
- **`ImportError: /lib64/libstdc++.so.6: version 'CXXABI_1.3.15' not
  found (required by .../libceres.so.4)`** — you have PyTorch installed
  from pip, which pulls in the system `libstdc++` at import time, and
  then conda-forge's newer `libceres` can't load against it. The
  durable fix is to install PyTorch from conda-forge so it shares a
  toolchain with Ceres:

  ```
  pip uninstall -y torch torchvision
  micromamba install -n <env> -c conda-forge \
      pytorch-gpu=2.4.1 \
      torchvision=0.19.1 \
      cuda-version=12.4
  ```

  As a per-shell workaround you can also force the conda libstdc++ to
  load first: `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`.
