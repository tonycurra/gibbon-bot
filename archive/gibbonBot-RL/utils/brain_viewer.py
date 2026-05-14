import argparse
import pathlib
import sys

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Brain Viewer")
    parser.add_argument("--task", default="stand")
    parser.add_argument("--scene", default="data/scene.xml")
    parser.add_argument("--db-path", default="runs/ape_policy_db.sqlite")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--brain", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--policy-limit", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--num-steps", type=int, default=None)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)

    # Collision filtering
    for i in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name and "floor" in geom_name:
            continue
        if (
            geom_name
            and geom_name.startswith("t")
            and geom_name[1:].split("_")[0].isdigit()
        ):
            continue
        if geom_name:
            model.geom(i).contype = 2
            model.geom(i).conaffinity = 1

    policy = None
    router = None

    if args.checkpoint:
        from utils.policy_router import PolicyRouter

        router = PolicyRouter(db_path=args.db_path)
        params = router.load_policy_params(args.checkpoint)
        policy = lambda obs: jnp.array(
            router.net.apply(params, jnp.array(obs[np.newaxis, :]))[0]
        )
        print(f"Loaded checkpoint: {args.checkpoint}")

    elif args.brain:
        from utils.policy_router import PolicyRouter

        router = PolicyRouter(db_path=args.db_path)

        if router.db.count() == 0:
            print("Importing checkpoints from runs/...")
            imported = router.import_from_runs()
            print(f"Imported {imported} checkpoints")

        selected = router.select_best_policy(
            task=args.task,
            scene_path=args.scene,
            evaluate=args.evaluate,
            num_steps=args.eval_steps,
            limit=args.policy_limit,
        )

        if selected:
            params = selected.params
            policy = lambda obs: jnp.array(
                router.net.apply(params, jnp.array(obs[np.newaxis, :]))[0]
            )
            print(f"Selected: {selected.checkpoint_path} (score={selected.score:.1f})")
        else:
            print("No policies found in database")

    else:
        import flax.linen as nn

        from utils.best import find_best

        OBS_DIM = 88
        ACT_DIM = 24

        class Net(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = nn.tanh(nn.Dense(256)(x))
                x = nn.tanh(nn.Dense(256)(x))
                x = nn.tanh(nn.Dense(128)(x))
                mean = nn.Dense(ACT_DIM)(x)
                return mean

        net = Net()
        init_params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS_DIM)))
        try:
            results = find_best(task=args.task, top=1)
            if results:
                ckpt_path = results[0][0]
                data_np = np.load(str(ckpt_path))
                keys = sorted(data_np.files, key=lambda k: int(k.split("_")[-1]))
                flat = [jnp.asarray(data_np[k]) for k in keys]
                init_flat, treedef = jax.tree_util.tree_flatten(init_params)
                params = jax.tree_util.tree_unflatten(treedef, flat)
                policy = lambda obs: net.apply(params, jnp.array(obs[np.newaxis, :]))
                print(f"Loaded best: {ckpt_path}")
        except Exception as e:
            print(f"Failed to load best checkpoint: {e}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Find torso body index
        torso_idx = None
        for i in range(model.nbody):
            if mujoco.mj_id2name(model, 0, i) == "torso":
                torso_idx = i
                break

        # Wait for torso to be settled on floor for ~1 second before starting policy
        import time

        startup_delay = 1.0  # seconds
        settled_threshold = 0.05  # velocity threshold m/s
        startup_done = False
        settled_start_time = None

        while viewer.is_running():
            mujoco.mj_step(model, data)

            # Check if torso is settled (low velocity, near floor)
            if torso_idx is not None and not startup_done:
                torso_z = data.body(torso_idx).xpos[2]
                torso_vel = np.linalg.norm(data.body(torso_idx).cvel[:3])
                # Torso is settled if it's near floor and moving slowly
                if torso_z < 0.1 and torso_vel < settled_threshold:
                    if settled_start_time is None:
                        settled_start_time = time.time()
                    elif time.time() - settled_start_time >= startup_delay:
                        startup_done = True
                        print("Startup delay complete. Policy active.")
                else:
                    settled_start_time = None

            if not startup_done:
                data.ctrl[:] = 0  # Zero out controls during startup delay
                viewer.sync()
                continue

            # Policy takes over after startup delay
            if policy:
                try:
                    obs = np.concatenate([data.qpos, data.qvel])
                    if len(obs) > 88:
                        obs = obs[:88]
                    elif len(obs) < 88:
                        obs = np.pad(obs, (0, 88 - len(obs)))
                    mean = policy(obs)
                    data.ctrl[:] = np.tanh(mean)
                except Exception as e:
                    print(f"Policy error: {e}")
                    data.ctrl[:] = 0
            else:
                data.ctrl[:] = 0
            viewer.sync()


if __name__ == "__main__":
    main()
