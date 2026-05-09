"""Per-limb dynamics diagnostic.

The robot has 4 morphologically identical limbs of 6 DOF each:
  FL: actuators[0:6]    (front-left)
  FR: actuators[6:12]   (front-right)
  BL: actuators[12:18]  (back-left)
  BR: actuators[18:24]  (back-right)

This module aggregates joint-level Mujoco state into per-limb metrics:
  - actuator torque sum  (data.qfrc_actuator)
  - bias force sum       (data.qfrc_bias = gravity + coriolis)
  - kinetic energy       (0.5 * qvel^2 summed)
  - position drift       (|target - actual| summed)

Use it (a) to verify URDF symmetry — same trajectory on each limb should
give same torques; (b) to read out per-limb gait phase during inference;
(c) to add per-limb terms to a reward.
"""
import numpy as np


LIMBS = ("FL", "FR", "BL", "BR")
JOINTS_PER_LIMB = 6
LIMB_RANGES = {
    "FL": slice(0,  6),
    "FR": slice(6,  12),
    "BL": slice(12, 18),
    "BR": slice(18, 24),
}


def per_limb(data, field: str = "qfrc_actuator", reduce: str = "sum"):
    """Extract `field` from a mujoco `data` (or mjx Data), slice per limb,
    apply `reduce` (sum/mean/abs_sum/rms/max).

    Robot joints in qfrc_* arrays start at offset 6 (free-joint takes [0:6]).
    So actuator[0..23] live at qfrc[6..29].
    """
    arr = np.asarray(getattr(data, field))
    assert arr.shape[-1] >= 6 + 24, f"unexpected shape for {field}: {arr.shape}"
    joint_arr = arr[..., 6:6 + 24]
    out = {}
    for limb, sl in LIMB_RANGES.items():
        sub = joint_arr[..., sl]
        if reduce == "sum":
            out[limb] = float(np.sum(sub))
        elif reduce == "mean":
            out[limb] = float(np.mean(sub))
        elif reduce == "abs_sum":
            out[limb] = float(np.sum(np.abs(sub)))
        elif reduce == "rms":
            out[limb] = float(np.sqrt(np.mean(sub * sub)))
        elif reduce == "max":
            out[limb] = float(np.max(np.abs(sub)))
        else:
            raise ValueError(f"unknown reduce: {reduce}")
    return out


def symmetry_score(per_limb_dict: dict) -> dict:
    """Compute pairwise asymmetry: |FL - FR|, |BL - BR|, |front - back|.
    Lower = more symmetric. For checking gait quality and URDF correctness."""
    fl, fr, bl, br = (per_limb_dict[k] for k in LIMBS)
    return {
        "lr_front":      abs(fl - fr),
        "lr_back":       abs(bl - br),
        "fb_left":       abs(fl - bl),
        "fb_right":      abs(fr - br),
        "total_asym":    abs(fl - fr) + abs(bl - br),
    }


def energy(data) -> float:
    """Sum of squared actuator torques — proxy for energy use this step."""
    arr = np.asarray(getattr(data, "qfrc_actuator"))[6:6 + 24]
    return float(np.sum(arr * arr))


if __name__ == "__main__":
    # Quick demo when run directly: load scene, step a few times, print balance.
    import mujoco
    import pathlib
    model = mujoco.MjModel.from_xml_path(str(pathlib.Path("data/scene.xml")))
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"{'metric':<22}  " + "  ".join(f"{l:>8}" for l in LIMBS))
    for field, reduce in [("qfrc_actuator", "abs_sum"),
                          ("qfrc_bias",     "abs_sum"),
                          ("qvel",          "rms")]:
        try:
            d = per_limb(data, field, reduce)
            print(f"{field+'.'+reduce:<22}  " + "  ".join(f"{d[l]:>8.4f}" for l in LIMBS))
        except Exception as e:
            print(f"{field+'.'+reduce:<22}  ERROR: {e}")
    print(f"\ntotal energy proxy: {energy(data):.4f}")
