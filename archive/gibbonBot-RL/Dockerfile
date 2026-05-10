# gibbonBot-RL training container — CUDA 12 + JAX-cuda + MuJoCo MJX.
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

# CUDA 12.2 base — matches WSL2 driver passthrough on the user's RTX 4070
# (driver 537.28 → CUDA 12.2). JAX-cuda 0.10.0 wheels are forward-compatible
# with newer drivers, so cloud H100 hosts on driver 535+ also work.
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

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
RUN pip install --no-cache-dir \
        "jax[cuda12]==0.10.0" \
        jaxlib==0.10.0 \
        flax==0.12.7 \
        optax==0.2.8 \
        numpy==2.4.4 \
        mujoco==3.8.0 \
        mujoco-mjx==3.8.0 \
        mlflow==3.12.0 \
        imageio==2.37.3 \
        pandas \
    && pip cache purge

# Copy source. .dockerignore keeps .venv, runs/, mlruns/, .git, _snapshots/ out.
COPY . /app

# Default workdir is /app; runs/, mlruns/, mlflow.db meant to be mounted.
RUN mkdir -p /app/runs /app/mlruns

# Headless: no viewer, no display.
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV PYTHONUNBUFFERED=1

# Default command — train walk. Override with `docker run ... <cmd>`.
CMD ["python", "train_mjx.py", "--task", "walk", "--auto-push", "--no-watch", "--label", "containerized run"]
