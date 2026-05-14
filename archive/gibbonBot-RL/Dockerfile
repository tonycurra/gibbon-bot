# gibbonBot-RL training container — CUDA 12.4 + JAX-cuda + MuJoCo MJX + Warp.
#
# Build:
#   docker build -t gibbonbot:latest .
#
# Run (local GPU):
#   docker run --rm --gpus all \
#     -v $(pwd)/runs:/app/runs \
#     -v $(pwd)/mlruns:/app/mlruns \
#     -v $(pwd)/mlflow.db:/app/mlflow.db \
#     gibbonbot:latest \
#     python train_mjx.py --task stand --no-resume --no-watch
#
# Or use docker-compose: see docker-compose.yml
#
# Image size ~5GB (mostly cuDNN + JAX-cuda wheels). Headless only — no GUI viewer
# (use scripts/snapshot.py to inspect via PNG dumps when running in cloud).

# CUDA 12.4 base — chosen to satisfy mujoco-warp's `wp.is_conditional_graph_supported()`
# check (CUDA >= 12.4 enables conditional graph nodes and reliable graph capture). CUDA
# forward-compatibility means a 12.4 user-space runs fine on the host's WSL2 driver
# 537.28 (CUDA 12.2 class). Cloud H100 hosts on driver 535+ also work.
# JAX has no jax[cuda13] wheels yet, so we stay on cuda12 extras.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# System deps: Python 3.12, git, ssh (for git push to backups), curl
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates curl git openssh-client \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default `python`/`python3`
RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && curl -sSL https://bootstrap.pypa.io/get-pip.py | python

WORKDIR /app

# Install JAX with CUDA 12 support FIRST so the layer caches across code changes.
# Pinning to match local environment versions.
# `--default-timeout=300 --retries=10`: jaxlib + mujoco wheels are 100+ MB each;
# resilience against slow / flaky connections so a single stalled chunk doesn't
# kill a 10-minute build.
# `imageio` deliberately omitted — only used by scripts/snapshot.py, which runs
# from the host venv, not inside the trainer container.
RUN pip install --no-cache-dir \
        --default-timeout=300 --retries=10 \
        "jax[cuda12]==0.10.0" \
        jaxlib==0.10.0 \
        flax==0.12.7 \
        optax==0.2.8 \
        numpy==2.4.4 \
        mujoco==3.8.1 \
        mujoco-mjx==3.8.1 \
        mujoco-warp>=3.8.0 \
        mlflow==3.12.0 \
        pandas \
    && pip cache purge

# Copy source. .dockerignore keeps .venv, runs/, mlruns/, .git, _snapshots/ out.
COPY . /app

# Default workdir is /app; runs/, mlruns/, mlflow.db meant to be mounted.
RUN mkdir -p /app/runs /app/mlruns

# Headless: no viewer, no display.
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV PYTHONUNBUFFERED=1

# Default command — train stand (current focus). Override with `docker run ... <cmd>`.
CMD ["python", "train_mjx.py", "--task", "stand", "--auto-push", "--no-watch", "--label", "containerized run"]
