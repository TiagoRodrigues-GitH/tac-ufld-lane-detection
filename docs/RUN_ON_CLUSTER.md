# Running the long experiments on a GPU cluster

Two experiments of the article take days on the author's desktop (one 6 GB RTX GPU) and are run on a cluster instead.
Everything below is scripted. The only things to adapt are the lines marked `CLUSTER` in `scripts/cluster/`: the
partition name and the module system.

| Experiment | Config | What | GPU time (A100, estimate) |
|---|---|---|---|
| OpenLane-V | `configs/cluster/openlane_v.yaml` | 10 models (2 baselines, 8 temporal) x 7 seeds x 30 epochs, scored with the RVLD protocol on 23,021 test frames | 7 jobs of ~15-30 h, in parallel |
| ELAS, 22 scenes | `configs/cluster/elas_full22.yaml` | 5 models x 7 seeds x 50 epochs, hyper-parameter search first, then robustness | ~8 h + 6 jobs of ~10-15 h |

The estimates come from the desktop timings scaled to an A100. The cluster smoke test (step 3) and the first epochs
in the logs give the real figures. Each job asks for 48 h and resumes where it stopped if it is resubmitted.

## Training recipe (why these settings)

Both cluster configs follow the official UFLD training recipe (Qin et al., ECCV 2020, github.com/cfzd/Ultra-Fast-Lane-Detection):

- the auxiliary segmentation branch is on (`model.ufld_aux: true`) for every UFLD model, baselines included; it is
  used only in training and is not counted in the parameters or the latency;
- batch 32 with Adam 4e-4, cosine schedule and warm-up.

All temporal models start from the baseline of the same seed. This makes each comparison exactly
"baseline vs baseline + temporal module".

The 7 paired seeds are what the two primary comparisons need: a paired Wilcoxon test with Holm correction cannot reach p < 0.05 with fewer than 6.

## Requirements

- Linux, one NVIDIA GPU per job with at least 16 GB (A100 40 GB recommended), CUDA 12 driver, Python 3.11 or newer.
- Disk: about 10 GB for OpenLane-V (labels plus the cropped 640x384 frames) and 11 GB for ELAS. Results without
  checkpoints take about 2 GB; checkpoints add about 8 GB per experiment.
- Internet on the login (or transfer) node for steps 1-2. The compute jobs need no internet.

## Steps

### 1. Code and environment (login node, ~10 min)

```bash
git clone -b handoff/v0.4 https://github.com/TiagoRodrigues-GitH/tac-ufld-lane-detection.git
cd tac-ufld-lane-detection
bash scripts/cluster/setup_env.sh      # venv, PyTorch, package, ImageNet weights, `doctor`, unit tests
mkdir -p logs                          # SLURM writes the job logs here
```

### 2. Data (login or transfer node)

**OpenLane-V** (~1-3 h, depending on the network). The script downloads the labels and lists from the RVLD
repository (540 MB). It then streams the 20 OpenLane image archives (~103 GB, read over HTTP and never stored)
and keeps only the cropped 640x384 frames of the OpenLane-V segments (~6 GB). It can be rerun safely: finished
archives are skipped.

```bash
export OPENLANE_ROOT=/scratch/$USER/OpenLane          # any folder with ~10 GB free
bash scripts/cluster/prepare_openlane_v.sh
```

**ELAS** (11 GB): a copy of `dataset_elas_v1`, all 22 scenes, comes from Tiago (it is not downloadable in this layout).

```bash
export ELAS_ROOT=/scratch/$USER/dataset_elas_v1
```

Add both `export` lines to `~/.bashrc` (or pass them with `sbatch --export=ALL`), because the job scripts read them.

### 3. Smoke test (one GPU, ~15 min)

This run puts a few frames through every stage with the cluster settings: auxiliary loss, batch 32, all 10 models.
It catches environment, data and GPU-memory problems before the 48 h jobs. Its numbers are not results.

```bash
srun --gres=gpu:1 --cpus-per-task=10 --mem=48G --time=01:00:00 \
     .venv/bin/python -m tac_ufld run --config configs/cluster/openlane_v_smoke.yaml
```

The run is good if it ends with `done: .../results/openlane_v_cluster_smoke/report/REPORT.md`. Its F1 is close to 0,
which is expected: the models see 64 frames for one epoch.

### 4. Submit

**OpenLane-V**: 7 independent seeds, then the report.

```bash
J=$(sbatch --parsable scripts/cluster/openlane_v.slurm)                          # array 1-7
sbatch --dependency=afterok:$J --export=ALL,REPORT=1 --array=0 scripts/cluster/openlane_v.slurm
```

**ELAS**: seed 1 runs the hyper-parameter search, and seeds 2-7 reuse it. Then the report and the robustness evaluation run.

```bash
J1=$(sbatch --parsable --array=1 scripts/cluster/elas.slurm)
J2=$(sbatch --parsable --array=2-7 --dependency=afterok:$J1 scripts/cluster/elas.slurm)
sbatch --dependency=afterok:$J2 --export=ALL,REPORT=1 --array=0 scripts/cluster/elas.slurm
```

The two experiments are independent and can run at the same time.

### 5. Follow, and resume after a time-out or a failure

- `squeue -u $USER` shows the jobs; `tail -f logs/tacufld-olv-<job>_<seed>.out` shows the epochs of one seed.
- Every job resumes. A trained model is skipped, an interrupted one restarts from its last epoch, and a finished
  seed is not evaluated again. Resubmitting the same seed is all it takes:
  `sbatch --array=<seed> scripts/cluster/openlane_v.slurm`.
- The parallel seeds write the same shared files (split lists, the provisional report). The final `REPORT=1` job
  rewrites all of them from every seed, so only its report counts.

### 6. What to send back

Send everything except the checkpoints:

```bash
tar czf tacufld_cluster_results.tgz --exclude='*.pt' --exclude='tensorboard' \
    results/openlane_v_cluster results/elas_full22_cluster logs
```

What the archive holds:

- `report/REPORT.md` and the CSVs behind the article's tables: aggregates, paired tests, efficiency, and the RVLD
  metrics `aggregate_native_*.csv`;
- `seed_*/results.csv`, `seed_*/per_frame/`, and the training histories;
- `config_resolved.yaml` and `environment.json` (the exact versions and the GPU).

Keep the checkpoints on the cluster until the article is accepted, in case a reviewer asks for a figure or a re-evaluation.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `CUDA out of memory` in the smoke test | The GPU has less than ~16 GB. Use `train.batch_size: 16` in the cluster config and report it: it changes the official recipe. |
| `FileNotFoundError: OpenLane-V: ... not found` | `OPENLANE_ROOT` is not exported in the job, or step 2 did not finish. Rerun `prepare_openlane_v.sh`; it exits with code 0 only when every frame is cached. |
| `URLError` / download in a compute job | The ImageNet weights were not cached: rerun `setup_env.sh` on the login node, which has internet. |
| `FullRunNotConfirmed` | Missing `--confirm`; the job scripts already pass it. |
| A job ends at 48 h | Resubmit the same array index; it resumes. |
