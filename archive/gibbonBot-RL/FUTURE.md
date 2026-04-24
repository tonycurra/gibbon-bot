# Future Implementation Ideas

## AMP — Adversarial Motion Priors

**What it is**: A technique from DeepMind (Peng et al., 2021) where instead of hand-crafting a reward function, you show the agent reference motion clips and train a discriminator to judge whether the agent's motion "looks like" the reference. The agent is rewarded for fooling the discriminator — i.e. for moving like the reference animal.

**Why it's powerful**: The agent doesn't need a precise joint-by-joint match to the reference. It learns the *style* of movement — the rhythm, the weight shifts, the follow-through. This is how most of the impressive animal locomotion videos (quadruped galloping, backflips, parkour) are made.

**How it works**:
1. Collect reference motion clips (keyframe sequences of joint positions/velocities)
2. Train a discriminator D(s, s') that scores whether a state transition looks like the reference
3. Agent reward = task reward + λ × discriminator score
4. Discriminator and policy train together adversarially (like a GAN)

**Paper**: "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control" — Peng et al., SIGGRAPH 2021.

---

## Pipeline A — Manual Keyframe Reference Motion

**Complexity**: Low (1–2 days)

Pose the robot manually through a desired motion (e.g. standing up) using MuJoCo's viewer or a small script, save `qpos` at each frame, use it as a reference trajectory with a phase-matching imitation reward.

### Steps
1. Write `record_motion.py` — interactive script that lets you scrub through joint angles and save keyframes to a `.npy` file.
2. Add `ImitationReward` term to `stand_env.py`:
   - Track phase `φ = step / episode_length`
   - At each step, find nearest reference frame by phase
   - Reward += `exp(-||qpos - ref_qpos||²)`
3. Combine with existing height/upright rewards.

### Tools needed
- Just NumPy + MuJoCo (already have both)

---

## Pipeline B — Video Landmark Tracking (MediaPipe)

**Complexity**: Medium (3–5 days)

Record or download video of an animal performing the target behaviour. Use MediaPipe Pose or BlazePose to extract 2D/3D landmark positions per frame. Map landmarks to robot joint angles via inverse kinematics. Use the resulting trajectory as AMP reference motion.

### Steps
1. **Extract landmarks** from video:
   ```
   pip install mediapipe opencv-python
   ```
   MediaPipe Holistic gives 33 body landmarks at ~30 fps from any webcam or video file.

2. **Retarget** to robot skeleton:
   - Map human/ape landmark pairs (shoulder→elbow, hip→knee) to robot link lengths
   - Scale by ratio of limb lengths
   - Solve IK for each frame to get robot `qpos` sequence

3. **Build reference clip** — save as `.npy` array `(T, 31)` matching `qpos` layout.

4. **Add AMP discriminator** — small MLP trained on reference clips vs agent rollouts.

### Tools needed
- `mediapipe` — landmark extraction
- `opencv-python` — video I/O
- Custom IK retargeting (or use `mujoco.mj_step` + gradient-free IK)

### Good source videos
- BBC Earth, National Geographic ape footage
- Any quadruped animal getting up from the ground

---

## Pipeline C — SLEAP / DeepLabCut (Research-grade tracking)

**Complexity**: High (1–2 weeks)

For more accurate animal pose estimation than MediaPipe (which is trained on humans). Requires labelling a few hundred video frames with joint positions, training a custom pose model, then following Pipeline B from step 2 onward.

- **SLEAP**: https://sleap.ai — multi-animal pose tracking, active development
- **DeepLabCut**: https://deeplabcut.github.io — widely used in neuroscience/biomechanics

Best used when you have specific animal footage and MediaPipe's human-skeleton assumptions cause poor retargeting.

---

## Pipeline D — MuJoCo MJX (GPU Physics)

**Complexity**: Medium (1–2 days to migrate)

Run thousands of parallel MuJoCo environments entirely on the GPU using JAX. Transforms training from CPU-bound to GPU-bound, giving 50–100× speedup on the RTX 4070.

### Steps
1. Replace `mujoco.MjModel` / `MjData` with `mujoco.mjx.put_model` / `put_data`
2. Replace `mj_step` with `mjx.step` (JAX JIT-compiled)
3. Replace SB3 PPO with a JAX-native RL library: **Brax**, **PureJaxRL**, or **Rlax**
4. Vectorise environments with `jax.vmap`

### Expected speedup
- Current: ~1M steps/hour (1 env, CPU-bound)
- With MJX + 4070: ~50–100M steps/hour

### Reference
- MuJoCo MJX docs: https://mujoco.readthedocs.io/en/stable/mjx.html
- PureJaxRL: https://github.com/luchris429/purejaxrl

---

## Recommended order

| Priority | Task | Payoff | Effort |
|----------|------|--------|--------|
| 1 | More CPU envs (`--envs 32`) | 32× faster, zero code | 0 |
| 2 | Manual keyframe reference (Pipeline A) | Agent has a blueprint | 1–2 days |
| 3 | MJX migration (Pipeline D) | 50–100× faster | 1–2 days |
| 4 | MediaPipe video tracking (Pipeline B) | Bioinspired motion style | 3–5 days |
| 5 | Full AMP discriminator | Style transfer from video | 1 week |
| 6 | SLEAP custom tracking (Pipeline C) | Research-grade accuracy | 1–2 weeks |
