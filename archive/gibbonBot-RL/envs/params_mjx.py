# MJX training parameters — tune these before scaling up to many envs.
# Edit values here or run this file directly to open the slider UI.
# Reward/motor/learning params hot-reload between rollouts.
# PHYSICS_STEPS_PER_ACTION requires a training restart (baked into JAX compile).

# --- timing ---
PHYSICS_STEPS_PER_ACTION = 16    # physics steps per policy action (16 × 0.002s = 0.032s)
VIEWER_SPEED = 17.000000  # how fast the viewer plays back: 1=real-time, 10=10× faster

# --- motor ---
MOTOR_MAX_SPEED = 0.190000  # max normalised joint travel per action

# --- goals ---
TARGET_Z = 0.300000  # torso height target (metres)
TARGET_TOL = 0.100000            # Gaussian reward width (sigma): small=tight, large=forgiving

# --- reward weights ---
W_HEIGHT = 10.000000  # scale of the height reward
W_UPRIGHT = 5.000000  # scale of the upright orientation reward

# --- arm spacing ---
ARM_SAFETY_MARGIN = 0.050000  # arms closer than this (metres) incur a penalty
W_PROXIMITY = 3.000000  # how strongly the arm-spacing penalty is applied

# --- termination ---
FLIP_PENALTY = 200.000000  # one-time penalty when robot flips upside-down

# --- learning (hot-reloadable between rollouts) ---
LEARNING_RATE = 0.000330  # Adam learning rate — reduce if training is unstable
ENT_COEF = 0.050000  # entropy bonus: 0=exploit, 0.02=explore more
CLIP_EPS = 0.320000  # PPO clip range: how much policy can change per update

# --- smoothness penalties (set to 0 to disable) ---
W_SMOOTH = 0.500000  # penalises motor movement each step — reduces twitching
W_JVEL = 0.050000  # penalises fast joint velocities — reduces flailing
W_ENERGY = 0.010000  # penalises sum(ctrl²) — discourages high motor effort
W_ALIVE   = 0.050000             # small per-step bonus for staying alive (not flipped)


# ── slider UI (run this file directly to open) ────────────────────────────────
if __name__ == "__main__":
    import pathlib
    import tkinter as tk

    _FILE = pathlib.Path(__file__)

    # Each item is either a category header string or a param tuple:
    # (name, lo, hi, step, is_float, description)
    _PARAM_DEFS = [
        "Viewer",
        (
            "VIEWER_SPEED", 1.0, 100.0, 1.0, True,
            "How fast the viewer plays back the robot.\n"
            "1 = real time (slow, smooth, good for studying movement).\n"
            "10 = 10× faster. 100 = as fast as CPU allows.\n"
            "Set high during training, low when you want to watch carefully.",
        ),
        "Motor",
        (
            "MOTOR_MAX_SPEED", 0.01, 0.50, 0.01, True,
            "Maximum joint movement per policy action — the robot's 'joint speed'.\n"
            "Too low (< 0.05): robot barely moves, looks frozen.\n"
            "Too high (> 0.35): jerky, unstable, hard to control.\n"
            "0.2 is a good starting point.",
        ),
        "Height Goal",
        (
            "TARGET_Z", 0.05, 0.60, 0.01, True,
            "Target torso height off the ground (metres).\n"
            "0.1 = 10 cm — nearly lying down.\n"
            "0.2 = 20 cm — crouching low.\n"
            "0.4 = 40 cm — partially upright.\n"
            "Start low and increase as the robot learns.",
        ),
        (
            "TARGET_TOL", 0.005, 0.20, 0.005, True,
            "How precisely the robot must hit TARGET_Z to get rewarded (Gaussian sigma).\n"
            "Small (0.01): only rewarded right at the target — very hard.\n"
            "Large (0.15): rewarded anywhere in a wide band — easier.\n"
            "Start wide (0.1+), narrow as training matures.",
        ),
        (
            "W_HEIGHT", 0.0, 10.0, 0.1, True,
            "How much the robot cares about reaching TARGET_Z.\n"
            "0 = height doesn't matter.\n"
            "5 = strong pull toward target height.\n"
            "If robot ignores height, increase this.",
        ),
        (
            "W_UPRIGHT", 0.0, 5.0, 0.05, True,
            "How much the robot is penalised for tilting sideways or forward.\n"
            "0 = tilting is fine, robot can lean freely.\n"
            "2 = moderate penalty, encourages staying level.\n"
            "If robot reaches height but constantly tips over, increase this.",
        ),
        "Arm Spacing",
        (
            "ARM_SAFETY_MARGIN", 0.05, 0.50, 0.01, True,
            "Minimum distance (metres) arms should keep from each other.\n"
            "Arms closer than this get a growing penalty.\n"
            "Reduce if you want arms to move more freely and cross.",
        ),
        (
            "W_PROXIMITY", 0.0, 20.0, 0.5, True,
            "How strongly the arm-spacing penalty is applied.\n"
            "0 = arms can touch and cross freely.\n"
            "3 = moderate — arms tend to stay apart.\n"
            "10+ = arms strongly repel each other.\n"
            "If arms tangle or collide often, increase this.",
        ),
        "Survival",
        (
            "FLIP_PENALTY", 0.0, 200.0, 1.0, True,
            "One-time reward penalty when the robot flips upside-down.\n"
            "This also ends the episode immediately.\n"
            "Higher = stronger discouragement of flipping.\n"
            "If the robot keeps flipping on purpose, increase this.",
        ),
        (
            "W_ALIVE", 0.0, 0.5, 0.005, True,
            "Small reward every step the robot stays alive (not flipped).\n"
            "0 = no bonus. 0.05 = light encouragement to survive longer.\n"
            "Helps prevent the robot from giving up early in training.",
        ),
        "Smoothness",
        (
            "W_SMOOTH", 0.0, 0.5, 0.005, True,
            "Penalty for motor movement each step — directly reduces twitching.\n"
            "0 = no penalty, robot can twitch freely.\n"
            "0.05 = moderate smoothing (good starting point).\n"
            "0.2+ = heavy smoothing, robot moves slowly and carefully.\n"
            "Increase if the robot shakes or vibrates.",
        ),
        (
            "W_JVEL", 0.0, 0.05, 0.001, True,
            "Penalty for fast joint velocities — reduces flailing and sudden jerks.\n"
            "0 = no penalty. 0.001 = light damping. 0.01 = heavy damping.\n"
            "Increase if limbs spin or oscillate wildly.",
        ),
        (
            "W_ENERGY", 0.0, 0.01, 0.0001, True,
            "Penalty for total motor effort sum(ctrl²) — rewards efficient movement.\n"
            "0 = no penalty. 0.0005 = light. 0.005 = heavy.\n"
            "Increase if you want the robot to use minimal force.",
        ),
        "Learning",
        (
            "LEARNING_RATE", 1e-5, 1e-3, 1e-5, True,
            "Step size for the Adam optimiser — how fast the network weights change.\n"
            "3e-4 (0.0003) is the standard starting point.\n"
            "Too high: reward oscillates wildly, training unstable.\n"
            "Too low: learning is very slow. Reduce if training looks chaotic.\n"
            "Tip: type the exact value in the box (e.g. 0.0001).",
        ),
        (
            "ENT_COEF", 0.0, 0.05, 0.001, True,
            "Entropy bonus — rewards the policy for trying varied actions (exploration).\n"
            "0 = pure exploitation, policy sticks to what it knows.\n"
            "0.005 = slight exploration (good default).\n"
            "0.02+ = heavy exploration, policy tries many things.\n"
            "Increase early in training, reduce later to lock in good behaviour.",
        ),
        (
            "CLIP_EPS", 0.05, 0.50, 0.01, True,
            "PPO clip range — limits how much the policy can change in one update.\n"
            "0.2 = standard (policy changes by at most ±20% per update).\n"
            "Too high (> 0.4): large unstable updates, training can collapse.\n"
            "Too low (< 0.1): very conservative, slow learning.\n"
            "Reduce if you see sudden reward crashes.",
        ),
    ]

    def _read():
        vals = {}
        for line in _FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            v = v.split("#")[0].strip()
            try:
                vals[k.strip()] = float(v)
            except ValueError:
                pass
        return vals

    def _write(name, value, is_float):
        lines = _FILE.read_text().splitlines()
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith(name) and "=" in s:
                indent = line[: len(line) - len(line.lstrip())]
                comment = ("  " + s[s.index("#"):]) if "#" in s else ""
                val_str = f"{value:.6f}" if is_float else str(int(round(value)))
                lines[i] = f"{indent}{name} = {val_str}{comment}"
                break
        _FILE.write_text("\n".join(lines) + "\n")

    current = _read()
    root = tk.Tk()
    root.title("MJX Training Params")
    root.resizable(False, True)

    # Font scale: 120% of base sizes
    F_NAME  = ("Courier", 12, "bold")   # param name
    F_DESC  = ("TkDefaultFont", 10)     # description text
    F_ENTRY = ("Courier", 11)           # value entry box
    F_CAT   = ("TkDefaultFont", 12, "bold")  # category header

    container = tk.Frame(root)
    container.pack(fill="both", expand=True, padx=8, pady=4)
    canvas = tk.Canvas(container, width=620)
    scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    inner = tk.Frame(canvas)
    canvas.create_window((0, 0), window=inner, anchor="nw")
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    widgets = {}   # name -> (slider, entry_var, is_float)
    _last_mtime = [_FILE.stat().st_mtime]
    _user_moving = [False]

    def _commit(name, raw, is_float, slider):
        try:
            fv = float(raw)
        except ValueError:
            return
        fv = max(float(slider.cget("from")), min(float(slider.cget("to")), fv))
        _user_moving[0] = True
        slider.set(fv)
        _write(name, fv, is_float)
        _last_mtime[0] = _FILE.stat().st_mtime
        _user_moving[0] = False

    for item in _PARAM_DEFS:
        if isinstance(item, str):
            # Category header
            tk.Label(inner, text=item.upper(), anchor="w", font=F_CAT,
                     fg="#ffffff", bg="#444444").pack(
                fill="x", padx=4, pady=(10, 2), ipady=3
            )
            continue

        name, lo, hi, step, is_float, desc = item
        val = current.get(name, lo)
        section = tk.Frame(inner, bd=1, relief="groove")
        section.pack(fill="x", pady=2, padx=4)
        tk.Label(section, text=name, anchor="w", font=F_NAME).pack(
            fill="x", padx=6, pady=(5, 0)
        )
        tk.Label(section, text=desc, anchor="w", justify="left",
                 font=F_DESC, fg="#444444", wraplength=580).pack(fill="x", padx=6)
        row = tk.Frame(section)
        row.pack(fill="x", padx=6, pady=(2, 5))

        slider = tk.Scale(row, from_=lo, to=hi, resolution=step,
                          orient="horizontal", length=440, showvalue=False)
        slider.set(val)
        slider.pack(side="left")

        entry_var = tk.StringVar(value=f"{val:.4g}" if is_float else str(int(val)))
        entry = tk.Entry(row, textvariable=entry_var, width=9, font=F_ENTRY,
                         justify="right")
        entry.pack(side="left", padx=(8, 0))
        widgets[name] = (slider, entry_var, is_float)

        def _on_slider(v, _name=name, _var=entry_var, _is_float=is_float):
            _user_moving[0] = True
            fv = float(v)
            _var.set(f"{fv:.4g}" if _is_float else str(int(fv)))
            _write(_name, fv, _is_float)
            _last_mtime[0] = _FILE.stat().st_mtime
            _user_moving[0] = False

        def _on_entry(event, _name=name, _var=entry_var, _is_float=is_float, _sl=slider):
            _commit(_name, _var.get(), _is_float, _sl)

        slider.configure(command=_on_slider)
        entry.bind("<Return>", _on_entry)
        entry.bind("<FocusOut>", _on_entry)

    def _sync_from_file():
        try:
            mtime = _FILE.stat().st_mtime
            if mtime != _last_mtime[0] and not _user_moving[0]:
                _last_mtime[0] = mtime
                vals = _read()
                for name, (slider, entry_var, is_float) in widgets.items():
                    if name in vals:
                        fv = vals[name]
                        slider.set(fv)
                        entry_var.set(f"{fv:.4g}" if is_float else str(int(fv)))
        except Exception:
            pass
        root.after(1000, _sync_from_file)

    root.after(1000, _sync_from_file)
    root.mainloop()
