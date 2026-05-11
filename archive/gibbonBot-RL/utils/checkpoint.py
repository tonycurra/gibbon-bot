"""Lightweight checkpoint I/O for ape-brain and PPO policy state.

Uses safetensors as the on-disk format:
  - single binary file, no zip overhead (smaller than .npz)
  - mmap-able, zero-copy load
  - dtype/shape in header, no pickle risk
  - portable: PyTorch, JAX, HF Hub all read it

The flax param PyTree is flattened to a flat dict of leaf tensors. The tree
structure is stored as a sidecar `.treedef.json` next to the safetensors file
so we can reconstruct without needing an init_params reference.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
from safetensors.numpy import save_file, load_file


def _flatten_with_keys(tree: Any) -> tuple[dict[str, np.ndarray], list[str]]:
    """Flatten a PyTree to {dotted_key: ndarray}, preserve key order."""
    flat, treedef = jax.tree_util.tree_flatten_with_path(jax.device_get(tree))
    out: dict[str, np.ndarray] = {}
    keys: list[str] = []
    for path, leaf in flat:
        k = ".".join(str(p.key) if hasattr(p, "key") else str(p.idx) for p in path)
        out[k] = np.asarray(leaf)
        keys.append(k)
    return out, keys


def save(params: Any, path: str | pathlib.Path) -> pathlib.Path:
    """Save params to <path>.safetensors + <path>.treedef.json.

    `path` may be given with or without extension; the .safetensors suffix is
    enforced. Returns the resolved safetensors path.
    """
    p = pathlib.Path(path)
    if p.suffix != ".safetensors":
        p = p.with_suffix(".safetensors")
    p.parent.mkdir(parents=True, exist_ok=True)
    tensors, key_order = _flatten_with_keys(params)
    save_file(tensors, str(p))
    p.with_suffix(".treedef.json").write_text(json.dumps({"keys": key_order}))
    return p


def load(path: str | pathlib.Path, init_params: Any) -> tuple[Any, bool]:
    """Load safetensors weights into a flax param tree shaped like init_params.

    Returns (new_params, ok). ok=False on shape mismatch.
    """
    p = pathlib.Path(path)
    if p.suffix != ".safetensors":
        p = p.with_suffix(".safetensors")
    tensors = load_file(str(p))
    sidecar = p.with_suffix(".treedef.json")
    if sidecar.exists():
        keys = json.loads(sidecar.read_text())["keys"]
    else:
        keys = sorted(tensors.keys())
    init_flat, treedef = jax.tree_util.tree_flatten(init_params)
    new_flat = [jnp.asarray(tensors[k]) for k in keys]
    if len(new_flat) != len(init_flat) or any(a.shape != b.shape for a, b in zip(new_flat, init_flat)):
        return None, False
    return jax.tree_util.tree_unflatten(treedef, new_flat), True


def save_npz_compat(params: Any, path: str | pathlib.Path) -> pathlib.Path:
    """Back-compat: also write the old .npz layout (arr_0, arr_1, ...).

    Useful during the safetensors migration so existing tools that read .npz
    keep working.
    """
    p = pathlib.Path(path)
    if p.suffix != ".npz":
        p = p.with_suffix(".npz")
    flat, _ = jax.tree_util.tree_flatten(jax.device_get(params))
    np.savez(str(p), *flat)
    return p
