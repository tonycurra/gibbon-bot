# Cloud / containerized deployment

The whole training stack runs in a Docker container with one GPU passed
through. Same image works on your laptop's RTX 4070 and on any cloud GPU
instance (RunPod, Lambda, Hetzner, AWS p3/g5, GCP A100, etc.).

## What's in the container

- CUDA 12.6 + cuDNN runtime (NVIDIA base image)
- Python 3.12, JAX-cuda 0.10.0, JAXlib 0.10.0
- mujoco 3.8.0, mujoco-mjx 3.8.0
- flax, optax, numpy, mlflow, imageio
- All `train_mjx.py`, `view.py`, `scripts/`, `envs/`, `data/` source

Image size: ~5 GB (mostly cuDNN + JAX-cuda wheels). Headless — no GUI viewer
inside (use `scripts/snapshot.py` for inspection).

## Persistent state

Three host directories are mounted as volumes:

| Host path | Container path | What |
|---|---|---|
| `./runs/` | `/app/runs/` | checkpoints, train.log, keepers/ |
| `./mlruns/` | `/app/mlruns/` | legacy MLflow filesystem store |
| `./mlflow.db` | `/app/mlflow.db` | MLflow sqlite (current) |

Plus `./data/`, `./envs/`, `./scripts/` are bind-mounted so you can edit
locally and the container picks up changes (no rebuild).

## Local run (RTX 4070, WSL2 + nvidia-docker)

```bash
docker compose build trainer

# train one task
docker compose run --rm trainer \
    python train_mjx.py --task stand --no-resume --auto-push --no-watch

# full curriculum
docker compose run --rm trainer python scripts/curriculum.py

# MLflow UI in another terminal
docker compose up -d mlflow_ui
# → http://localhost:5000
```

## Cloud run (any provider)

1. Pick a GPU instance (T4, A10, A100, H100). Minimum: 8 GB VRAM (4070-class).
2. SSH in. Make sure `nvidia-docker` / `nvidia-container-toolkit` is set up.
3. Clone repo:
   ```bash
   git clone ssh://git@codeberg.org/tony_olivabot/gibbonBot-RL.git
   cd gibbonBot-RL
   ```
4. Build + run:
   ```bash
   docker compose build trainer
   docker compose run --rm trainer python scripts/curriculum.py
   ```

That's it. `runs/` and `mlflow.db` accumulate on the cloud host's disk.

### Pulling training data back to your laptop

After cloud training completes, sync the artifacts:

```bash
# from your laptop:
rsync -avz cloud-host:/path/to/gibbonBot-RL/runs/ ./runs/
rsync -avz cloud-host:/path/to/gibbonBot-RL/mlflow.db ./mlflow.db
```

Or just `git pull` if the container had `--auto-push` enabled (it ships to
Codeberg automatically).

## Provider-specific notes

### RunPod
- Pick the "PyTorch 2.x" or "Cuda 12.4" community template. CUDA 12+ required.
- Runs `docker compose` natively. Persistent volume mount for `runs/`.

### Hetzner GPU (German EU)
- They have RTX 4090 instances at ~€40/mo or hourly. EU-jurisdiction, matches
  user's Codeberg backup region.
- nvidia-docker pre-installed on the GPU images.

### Lambda Labs
- Quick on-demand A10/A100. No nvidia-docker setup needed; Lambda images come
  ready.

### AWS / GCP
- `g5.xlarge` (A10), `p3.2xlarge` (V100), `a2-highgpu-1g` (A100). Use the
  Deep Learning AMI / Image; Docker + nvidia-runtime preinstalled.

## Database / state transfer

`mlflow.db` is a single SQLite file — `scp` it anywhere. To load on a new host:

```bash
scp mlflow.db newhost:/path/to/gibbonBot-RL/
docker compose up -d mlflow_ui   # browse at http://newhost:5000
```

For a more durable setup, point MLflow at a managed Postgres later (P3 in
BACKLOG.md) — but at current data sizes (1 MB sqlite, 100 MB checkpoints),
the file-based approach is plenty.

## Limitations / gotchas

- **No live viewer in cloud.** `view.py` requires X11/Wayland; cloud GPUs are
  headless. Use `scripts/snapshot.py --frames N` to dump PNGs and copy them
  back, or run the viewer locally pointed at a remote-pulled `runs/` dir.
- **Driver version**: container needs CUDA 12 *driver* on host (not just
  toolkit). On WSL2 this means Windows GeForce driver 535+; on Linux server,
  `nvidia-driver-535` or newer.
- **First run pulls JAX-cuda wheels** (~3 GB) into the image. After that
  the layer is cached.
- **mjx_*.npz checkpoint format**: shape-locked to OBS_DIM=88. If the env
  obs schema changes, old checkpoints can't be resumed. The `--resume-from`
  path is shape-checked.
