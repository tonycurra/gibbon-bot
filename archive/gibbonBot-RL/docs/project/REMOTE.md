# Remote training — SSH setup & workflow

We can train on a remote GPU server while watching live on the local machine.
This doc covers what we tried, what worked, and how to extend it.

---

## Servers tested

| Host | GPU | RAM | Use |
|------|-----|-----|-----|
| `root@51.159.147.155` | NVIDIA H100 PCIe (80 GB) | 24 cores | GPU training (MJX) |
| `root@163.172.142.212` | (CPU instance) | — | Backup / SB3 CPU training |

---

## Speed up SSH (essential!)

Without multiplexing, every `ssh` call opens a new TCP+TLS connection (~1s overhead).
With it, all subsequent calls reuse one persistent connection — ~10× faster.

Add to `~/.ssh/config`:

```
Host 51.159.147.155
    ControlMaster auto
    ControlPath ~/.ssh/cm/%r@%h:%p
    ControlPersist 10m
    ServerAliveInterval 30
```

`mkdir -p ~/.ssh/cm` once. After this, `ssh root@51.159.147.155 "echo hi"` is ~0.1s instead of 0.7s.

---

## Initial server setup

One-time, for any new GPU box:

```bash
# from local
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='runs/' \
  /home/tony/code/gb2.1/ root@SERVER:/root/gb2.1/

# on server
ssh root@SERVER
apt-get update -q && apt install -y python3.12-venv
cd /root/gb2.1
python3 -m venv .venv
source .venv/bin/activate
pip install mujoco mujoco-mjx stable-baselines3 gymnasium 'jax[cuda12]' flax optax
```

Verify GPU is visible to JAX:
```bash
python3 -c "import jax; print(jax.devices())"
# Expected: [CudaDevice(id=0)]
```

---

## Bidirectional sync workflow

`remote_sync.sh` runs locally in a loop. Every 30 seconds it:

1. **Pushes** `envs/hints.py` → server (so editing hints locally steers training live)
2. **Pulls** `runs/*.zip` `runs/*.npz` `runs/lineage.json` ← server

The local `tools/watch.py` polls `runs/` and hot-swaps the policy when new checkpoints arrive.

```bash
# terminal 1 — start sync
./remote_sync.sh

# terminal 2 — watch the policy as it evolves
python tools/watch.py --task stand
```

To steer training: edit `envs/hints.py` locally → next sync pushes → next episode picks up new reward weights (already hot-reloaded by `StandEnv._reload_hints`).

---

## Running training remotely

Background-launch with `nohup` so the process survives SSH disconnect:

```bash
ssh root@51.159.147.155 "cat > /root/restart.sh << 'EOF'
#!/bin/bash
kill \$(pgrep -f train_mjx) 2>/dev/null
sleep 2
cd /root/gb2.1
source .venv/bin/activate
nohup python3 train_mjx.py --envs 4096 > /root/gb2.1/train_mjx.log 2>&1 &
echo PID:\$!
EOF
chmod +x /root/restart.sh"

ssh root@51.159.147.155 "bash /root/restart.sh"
```

Check progress:
```bash
ssh root@51.159.147.155 "tail -20 /root/gb2.1/train_mjx.log"
ssh root@51.159.147.155 "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
```

Stop:
```bash
ssh root@51.159.147.155 "kill \$(pgrep -f train_mjx)"
```

---

## Gotchas we hit

- **`scene.xml` had absolute meshdir**: `/home/tony/code/gb2.1/roarm_ws/...` — fixed to relative `roarm_ws/roarm_description/`.
- **`python3.12-venv` not installed by default** on Ubuntu 24.04 — apt-install before `python3 -m venv`.
- **SSH connection drops on `pkill`/`kill`**: writing the kill+restart commands into a server-side script (`/root/restart.sh`) and invoking that avoids the issue.
- **JAX JIT compilation takes 2–5 min** for large env counts (4096+). First "steps=" line appears late.
- **GAE vmap** needs `in_axes=(1, 1, 1, 0)` because `last_val` is 1D `(N_ENVS,)` while rollout buffers are 2D `(T, N_ENVS)`.
- **MJX done-env reset broadcasting**: use `done_mask.reshape(done_mask.shape + (1,) * (n.ndim - 1))` to handle arbitrary rank fields in MJX `Data` pytree.

---

## Possible ways to extend

### Sharing training data
- **Codeberg/GitHub LFS** — push checkpoints to git. Pros: easy rollback. Cons: 100MB file limit, LFS storage cost.
- **`rsync` to S3 / Scaleway Object Storage** — cheaper for large checkpoints, but you give up git diff/history.
- **WandB or TensorBoard.dev** — push only TensorBoard event files, view metrics in browser, keep checkpoints local.

### Live remote viewer
Currently we sync checkpoints and view locally. Two alternatives:

- **VirtualGL / X11 forwarding**: `ssh -X` + `mujoco.viewer.launch_passive` on server. Needs OpenGL on remote. Often laggy.
- **Headless render → MP4 → scp** — periodic 10-second clips of the latest policy, pulled back to local. Good for sharing, not real-time.
- **WebRTC streaming**: serve frames over WebRTC from the GPU box. Highest fidelity, most engineering.

The current approach (sync checkpoints, view locally) is the simplest and matches our existing `watch.py` flow exactly.

### Multi-server training
- Run `train_mjx.py` on H100 + `train.py --task reach --envs 32` on the CPU box → two parallel skill streams, merge with `--inherit` later.
- Use `--inherit` across machines: pull the H100's `best_model.zip`, copy weights into a CPU-trained reach policy.

### Auto-restart on crash
Wrap the launch in `while true; do python3 train_mjx.py ... ; sleep 5; done` inside `restart.sh`.
Or use `systemd --user` for proper service management.

### Spot/preemptible GPUs
If using cloud spot instances, the job can be killed at any time. Use `--inherit` on each restart and a small checkpoint frequency (`SAVE_EVERY = 100_000`).
