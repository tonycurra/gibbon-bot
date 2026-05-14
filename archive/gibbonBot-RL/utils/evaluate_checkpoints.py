#!/usr/bin/env python3
"""
Evaluate all past checkpoints on the current scene.
Ranks them by stability and returns the best N.

Usage:
    python -m utils.evaluate_checkpoints --task stand --top 5
    python -m utils.evaluate_checkpoints --all-tasks --top 3
"""

import argparse
import pathlib
import sys

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

OBS_DIM = 88
ACT_DIM = 24

def build_net():
    """Match view.py network: just the actor, returns mean."""
    import flax.linen as nn
    
    class Net(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.tanh(nn.Dense(256)(x))
            x = nn.tanh(nn.Dense(256)(x))
            x = nn.tanh(nn.Dense(128)(x))
            mean = nn.Dense(ACT_DIM)(x)
            log_std = self.param("log_std", nn.initializers.zeros, (ACT_DIM,))
            value = nn.Dense(1)(x).squeeze(-1)
            return mean, log_std, value
    return Net()


def load_checkpoint(ckpt_path):
    """Load .npz checkpoint, match view.py approach."""
    net = build_net()
    init_params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS_DIM)))
    
    data = np.load(str(ckpt_path))
    keys = sorted(data.files, key=lambda k: int(k.split("_")[-1]))
    flat = [jnp.asarray(data[k]) for k in keys]
    init_flat, treedef = jax.tree_util.tree_flatten(init_params)
    
    if any(a.shape != b.shape for a, b in zip(flat, init_flat)):
        raise ValueError(f"Shape mismatch in {ckpt_path.name}")
    
    params = jax.tree_util.tree_unflatten(treedef, flat)
    net.apply = jax.jit(net.apply)
    return params, net


def evaluate_checkpoint(ckpt_path, scene_path="data/scene.xml", num_steps=200):
    """Evaluate a single checkpoint."""
    try:
        params, net = load_checkpoint(ckpt_path)
        
        # Load scene via mujoco directly (like view.py)
        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)
        
        # Find torso body index
        torso_idx = None
        for i in range(model.nbody):
            if model.body(i).name == "torso":
                torso_idx = i
                break
        
        # Reset simulation
        mujoco.mj_resetData(model, data)
        
        total_reward = 0.0
        torso_heights = []
        
        for step in range(num_steps):
            # Build observation (simplified - just use qpos and qvel)
            obs = np.concatenate([data.qpos, data.qvel])
            if len(obs) < OBS_DIM:
                obs = np.pad(obs, (0, OBS_DIM - len(obs)))
            elif len(obs) > OBS_DIM:
                obs = obs[:OBS_DIM]
            
            # Get action from policy
            mean, _, _ = net.apply(params, jnp.array(obs[np.newaxis, :]))
            action = np.array(jnp.tanh(mean).squeeze(0))
            
            # Apply action to mujoco
            # This is a simplified version - we're not using the full env
            # Just step the physics forward
            data.ctrl[:] = action
            mujoco.mj_step(model, data)
            
            # Simple reward: torso height
            if torso_idx is not None:
                torso_z = data.body(torso_idx).xpos[2]
            else:
                torso_z = data.qpos[2] if len(data.qpos) > 2 else 0
            torso_heights.append(float(torso_z))
            total_reward += max(0, torso_z - 0.3)  # Simple reward for height > 0.3m
            
            # Check termination (torso too low = fallen)
            if torso_z < 0.1:
                break
        
        steps = len(torso_heights)
        return {
            "path": str(ckpt_path),
            "total_reward": float(total_reward),
            "avg_torso_height": float(np.mean(torso_heights)) if torso_heights else 0,
            "height_std": float(np.std(torso_heights)) if len(torso_heights) > 1 else 0,
            "steps_survived": steps,
            "is_stable": steps == num_steps,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"path": str(ckpt_path), "error": str(e)[:100], "total_reward": -float("inf")}


def find_all_checkpoints(task=None):
    """Find all .npz checkpoints."""
    runs_dir = pathlib.Path("runs")
    ckpts = []
    
    if task:
        task_dirs = [runs_dir / f"ape_{task}_mjx"]
    else:
        task_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and "ape_" in d.name]
    
    for task_dir in task_dirs:
        if not task_dir.exists():
            continue
        for npz_file in sorted(task_dir.glob("mjx_*.npz"), key=lambda p: p.stat().st_mtime, reverse=True):
            ckpts.append({
                "path": npz_file,
                "task": task_dir.name.replace("ape_", "").replace("_mjx", ""),
                "step": int(npz_file.stem.split("_")[1]),
            })
    return ckpts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="stand")
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--scene", default="data/scene.xml")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()
    
    task = None if args.all_tasks else args.task
    ckpts = find_all_checkpoints(task)
    
    if not ckpts:
        print("No checkpoints found", file=sys.stderr)
        sys.exit(1)
    
    print(f"Evaluating {len(ckpts)} checkpoints...")
    
    results = []
    for i, c in enumerate(ckpts):
        print(f"[{i+1}/{len(ckpts)}] {c['path'].name}...", end=" ", flush=True)
        r = evaluate_checkpoint(c["path"], args.scene, args.steps)
        r["task"] = c["task"]
        r["step"] = c["step"]
        results.append(r)
        print("OK" if "error" not in r else "FAIL")
    
    results.sort(key=lambda x: -x["total_reward"])
    
    print(f"\n{'#':<4} {'Task':<10} {'Step':<10} {'Reward':<12} {'Height':<10} {'Steps':<8} {'Path':<30}")
    print("-" * 90)
    for i, r in enumerate(results[:args.top]):
        err = r.get("error", "")
        if err:
            print(f"{i+1:<4} {r.get('task','?'):<10} {r.get('step',0):<10} {'ERROR':<12} {'ERROR':<10} {'ERROR':<8} {r['path']:<30}")
        else:
            print(f"{i+1:<4} {r.get('task','?'):<10} {r.get('step',0):<10} {r['total_reward']:<12.1f} {r['avg_torso_height']:<10.3f} {r['steps_survived']:<8} {r['path']:<30}")
    
    if args.output:
        import json
        pathlib.Path(args.output).write_text(json.dumps(results[:args.top], indent=2))
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
