# Docker and reproducible execution

The image contains the code and its dependencies only. Datasets, results, checkpoints and the PyTorch weight cache are **mounted**, never copied in (`.dockerignore` excludes them, tested in `tests/test_docker.py`).

## Build

```powershell
docker build -t tac-ufld:gpu .                                                       # CUDA 12.6 PyTorch (default)
docker build -t tac-ufld:cpu --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu .
```

Build arguments: `TORCH_INDEX` (PyTorch wheel index: `cu126`, `cu128`, `cpu`, ...), `TORCH_VERSION` (default 2.8.0), `EXTRAS` (default `ui,deploy,dev`). The image uses Python 3.12 (the development machine used 3.14 with PyTorch 2.14; the package supports Python ≥ 3.10).

## Run with Compose

Copy `.env.example` to `.env` and set the host folders:

```ini
ELAS_HOST_DIR=D:/datasets/dataset_elas_v1
# CULANE_HOST_DIR=D:/datasets/CULane
RESULTS_HOST_DIR=./results
```

```powershell
docker compose run --rm doctor          # Python / PyTorch / GPU / dataset checks
docker compose run --rm tests           # test suite (network tests excluded)
docker compose run --rm smoke           # ~10 min plumbing check
docker compose run --rm pilot           # ~2.5 h GPU pilot (configs/elas_pilot.yaml)
docker compose up ui                    # http://localhost:8501
docker compose --profile cpu run --rm tests-cpu
# anything else:
docker compose run --rm doctor python -m tac_ufld run --config configs/elas.yaml --confirm
```

Mounts: `/data/elas`, `/data/culane`, `/data/tusimple`, `/data/openlane` (read-only; disabled datasets point at an empty placeholder), `/app/results` (read-write), a named volume for `/cache/torch` (ImageNet weights, downloaded once). The container environment sets `ELAS_ROOT=/data/elas` etc., which the dataset registry reads. Enable a dataset in `configs/datasets.yaml` before using it; the registry file is part of the image, so rebuild or mount `configs/` after changing it.

`shm_size: 4gb` is required: PyTorch DataLoader workers pass batches through shared memory, and Docker's default (64 MB) makes training crash with bus errors.

## GPU

* **Linux**: NVIDIA driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Check with `docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi`. Compose requests all GPUs through `deploy.resources.reservations.devices`.
* **Windows**: Docker Desktop with the **WSL 2 backend** and a recent NVIDIA Windows driver (the driver exposes the GPU to WSL 2; do **not** install a Linux NVIDIA driver inside WSL). Check with the same `nvidia-smi` command. GPU support is not available with the Hyper-V backend.
* The CUDA version of the PyTorch wheel (`TORCH_INDEX`) must be supported by the host driver (`nvidia-smi` shows the highest supported CUDA version; cu126 needs driver ≥ 560 on Windows / ≥ 525 on Linux).
* Jetson: do not use this image (x86-64). Use NVIDIA's L4T PyTorch containers or a native JetPack install (docs/DEPLOYMENT.md).

## Windows specifics

* Bind-mounting folders from `C:` into a WSL 2 container is slow for many small files. For training, keep the dataset inside the WSL 2 file system (`\\wsl$\...`) or accept slower data loading.
* Paths with non-ASCII characters (`Residência`) work in the container (Linux paths); on native Windows the package reads images with PIL / `np.fromfile` and opens videos via short (8.3) paths for the same reason.
* Line endings: `.gitattributes` forces LF, so the shell scripts under `scripts/` run in the container.

## Validation status

`docker compose config` parses (tested). The CPU image was built and run on the development machine; results are in `docs/TESTING.md`. The GPU image was not run on this machine: Docker Desktop's GPU passthrough was not exercised in this session, so the GPU container path is **not validated**.
