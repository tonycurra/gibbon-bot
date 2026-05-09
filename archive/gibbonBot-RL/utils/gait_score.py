"""Gait coherence score — quantifies how trot-like the policy's gait is.

A trotting quadruped has FL+BR limbs synchronized (in-phase) and FL vs FR
opposite-phase. We measure this from joint velocity patterns:

    coherence = corr(FL_shoulder_vel, BR_hip_vel) - |corr(FL, FR)|

Pure trot → +2. No pattern → 0. Pacing (FL+FR sync) → negative.

This is a cheap proxy for "biological locomotion" without rendering or
running a 2D pose estimator. Apes mostly trot/walk; getting coherence > 0
is a strong sign the policy learned something quadrupedal.
"""
import numpy as np


# Joint indices: 0-5=FL, 6-11=FR, 12-17=BL, 18-23=BR.
# Shoulder pitch / hip pitch is index 1 within each 6-joint limb block.
SHOULDER_PITCH = {"FL": 1, "FR": 7, "BL": 13, "BR": 19}


def trot_coherence(joint_vel_history) -> float:
    """joint_vel_history: shape (T, 24). Returns scalar in roughly [-2, 2].

    >0 = trot-like (diagonal pairs synced)
    <0 = pacing or chaotic
    Near 0 = no rhythmic pattern."""
    arr = np.asarray(joint_vel_history)
    if arr.shape[0] < 8:
        return 0.0  # too short
    fl = arr[:, SHOULDER_PITCH["FL"]]
    fr = arr[:, SHOULDER_PITCH["FR"]]
    bl = arr[:, SHOULDER_PITCH["BL"]]
    br = arr[:, SHOULDER_PITCH["BR"]]

    def corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / denom) if denom > 1e-9 else 0.0

    diag    = corr(fl, br) + corr(fr, bl)        # diagonal pairs (trot)
    same_l  = abs(corr(fl, fr))                  # same-side fronts (pacing)
    same_r  = abs(corr(bl, br))                  # same-side backs
    return diag - 0.5 * (same_l + same_r)


def stride_period(joint_vel_history, dt_action: float = 0.032) -> float:
    """Estimate stride period (s) via FFT peak of FL shoulder pitch velocity.
    Returns 0 if no clear peak."""
    arr = np.asarray(joint_vel_history)
    if arr.shape[0] < 16:
        return 0.0
    sig = arr[:, SHOULDER_PITCH["FL"]] - arr[:, SHOULDER_PITCH["FL"]].mean()
    if np.abs(sig).max() < 1e-3:
        return 0.0
    spec = np.abs(np.fft.rfft(sig))
    spec[0] = 0  # ignore DC
    peak = int(np.argmax(spec))
    if peak == 0:
        return 0.0
    freq = peak / (len(sig) * dt_action)  # Hz
    return float(1.0 / freq) if freq > 1e-3 else 0.0


if __name__ == "__main__":
    # Demo on a synthetic trot pattern
    T = 100
    t = np.arange(T) * 0.032
    f = 1.5
    v = np.zeros((T, 24))
    v[:, SHOULDER_PITCH["FL"]] = np.sin(2 * np.pi * f * t)
    v[:, SHOULDER_PITCH["BR"]] = np.sin(2 * np.pi * f * t)        # in-phase with FL → trot
    v[:, SHOULDER_PITCH["FR"]] = np.sin(2 * np.pi * f * t + np.pi)
    v[:, SHOULDER_PITCH["BL"]] = np.sin(2 * np.pi * f * t + np.pi)
    print(f"synthetic trot — coherence: {trot_coherence(v):+.3f}  stride: {stride_period(v):.2f}s")

    # Random pattern → near 0
    rng = np.random.default_rng(0)
    v_rand = rng.standard_normal((T, 24))
    print(f"random         — coherence: {trot_coherence(v_rand):+.3f}  stride: {stride_period(v_rand):.2f}s")

    # Pacing → negative
    v_pace = np.zeros((T, 24))
    v_pace[:, SHOULDER_PITCH["FL"]] = np.sin(2 * np.pi * f * t)
    v_pace[:, SHOULDER_PITCH["FR"]] = np.sin(2 * np.pi * f * t)        # same phase = pacing
    v_pace[:, SHOULDER_PITCH["BL"]] = np.sin(2 * np.pi * f * t + np.pi)
    v_pace[:, SHOULDER_PITCH["BR"]] = np.sin(2 * np.pi * f * t + np.pi)
    print(f"synthetic pace — coherence: {trot_coherence(v_pace):+.3f}  stride: {stride_period(v_pace):.2f}s")
