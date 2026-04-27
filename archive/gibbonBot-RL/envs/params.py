# Hot-reloadable training parameters.
# Edit values here or run this file directly to open the slider UI.
# Changes take effect within 1 episode.

# --- motor ---
MOTOR_MAX_SPEED = 0.200000  # max normalized joint movement per step

# --- timing ---
PHYSICS_STEPS_PER_ACTION = 170  # steps per policy action (240 × 0.002s = 0.48s)
REALTIME_FACTOR = 1.000000  # viewer speed multiplier: 1=real-time, 10=10× faster

# --- goals ---
TARGET_Z = 0.200000  # torso height target (metres)
TARGET_TOL = 0.100000  # Gaussian sigma for height reward

# --- reward weights ---
W_HEIGHT = 5.000000  # Gaussian height reward scale
W_UPRIGHT = 2.000000  # upright orientation reward (torso level)

# --- arm spacing ---
ARM_SAFETY_MARGIN = 0.390000  # soft penalty ramps up inside this distance (metres)
W_PROXIMITY = 3.000000  # scale of the soft proximity penalty

# --- termination penalties ---
FLIP_PENALTY = 100.000000  # one-time penalty for flipping upside-down


# ── slider UI (run this file directly to open) ────────────────────────────────
if __name__ == "__main__":
    import pathlib
    import tkinter as tk

    _FILE = pathlib.Path(__file__)

    _PARAM_DEFS = [
        # (name, lo, hi, step, is_float, description)
        (
            "REALTIME_FACTOR",
            1.0,
            100.0,
            1.0,
            True,
            "Viewer speed multiplier — training always runs full speed.\n"
            "1 = real-time  |  10 = 10× faster  |  100 = as fast as possible",
        ),
        (
            "PHYSICS_STEPS_PER_ACTION",
            50,
            300,
            10,
            False,
            "How many physics steps the sim runs before the policy picks a new action.\n"
            "Higher = more time per decision (smoother but slower to train).",
        ),
        (
            "MOTOR_MAX_SPEED",
            0.01,
            0.20,
            0.01,
            True,
            "Max joint movement per action (in normalised units).\n"
            "Too low = frozen robot.  Too high = jerky/shaking motion.",
        ),
        (
            "TARGET_Z",
            0.05,
            0.60,
            0.01,
            True,
            "Target torso height in metres. 0.3 ≈ torso ~30 cm off the ground.",
        ),
        (
            "TARGET_TOL",
            0.005,
            0.10,
            0.005,
            True,
            "How tight the height reward is (Gaussian sigma).\n"
            "Small = only rewarded exactly at target.  Large = rewarded over a wider band.",
        ),
        (
            "W_HEIGHT",
            0.0,
            5.0,
            0.1,
            True,
            "How much the height goal matters. Higher = robot prioritises reaching TARGET_Z.",
        ),
        (
            "W_UPRIGHT",
            0.0,
            2.0,
            0.05,
            True,
            "How much staying level matters. Higher = robot penalised more for tilting.",
        ),
        (
            "ARM_SAFETY_MARGIN",
            0.05,
            0.40,
            0.01,
            True,
            "Soft exclusion zone around each arm (metres).\n"
            "Arms entering this range incur a growing quadratic penalty.",
        ),
        (
            "W_PROXIMITY",
            0.0,
            20.0,
            0.5,
            True,
            "Strength of the soft arm-spacing penalty.\n"
            "0 = disabled. Higher = robot keeps arms further apart.",
        ),
        (
            "FLIP_PENALTY",
            0.0,
            100.0,
            1.0,
            True,
            "One-time reward penalty when the robot flips upside-down.",
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
                comment = ("  " + s[s.index("#") :]) if "#" in s else ""
                val_str = f"{value:.6f}" if is_float else str(int(round(value)))
                lines[i] = f"{indent}{name} = {val_str}{comment}"
                break
        _FILE.write_text("\n".join(lines) + "\n")

    current = _read()
    root = tk.Tk()
    root.title("Training Params")
    root.resizable(False, True)

    container = tk.Frame(root)
    container.pack(fill="both", expand=True, padx=8, pady=4)
    canvas = tk.Canvas(container, width=500)
    scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    inner = tk.Frame(canvas)
    canvas.create_window((0, 0), window=inner, anchor="nw")
    inner.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    sliders = {}  # name -> (slider, val_var, is_float)
    _last_mtime = [_FILE.stat().st_mtime]
    _user_moving = [False]

    for name, lo, hi, step, is_float, desc in _PARAM_DEFS:
        val = current.get(name, lo)

        section = tk.Frame(inner, bd=1, relief="groove")
        section.pack(fill="x", pady=4, padx=4)

        tk.Label(section, text=name, anchor="w", font=("Courier", 10, "bold")).pack(
            fill="x", padx=4, pady=(4, 0)
        )
        tk.Label(
            section,
            text=desc,
            anchor="w",
            justify="left",
            font=("TkDefaultFont", 8),
            fg="#555555",
            wraplength=470,
        ).pack(fill="x", padx=4)

        row = tk.Frame(section)
        row.pack(fill="x", padx=4, pady=(2, 4))

        slider = tk.Scale(
            row,
            from_=lo,
            to=hi,
            resolution=step,
            orient="horizontal",
            length=380,
            showvalue=False,
        )
        slider.set(val)
        slider.pack(side="left")

        val_var = tk.StringVar(value=f"{val:.4g}" if is_float else str(int(val)))
        tk.Label(
            row, textvariable=val_var, width=7, anchor="e", font=("Courier", 9)
        ).pack(side="left", padx=(4, 0))

        sliders[name] = (slider, val_var, is_float)

        def _on_change(v, _name=name, _var=val_var, _is_float=is_float):
            _user_moving[0] = True
            fv = float(v)
            _var.set(f"{fv:.4g}" if _is_float else str(int(fv)))
            _write(_name, fv, _is_float)
            _last_mtime[0] = _FILE.stat().st_mtime
            _user_moving[0] = False

        slider.configure(command=_on_change)

    def _sync_from_file():
        try:
            mtime = _FILE.stat().st_mtime
            if mtime != _last_mtime[0] and not _user_moving[0]:
                _last_mtime[0] = mtime
                vals = _read()
                for name, (slider, val_var, is_float) in sliders.items():
                    if name in vals:
                        fv = vals[name]
                        slider.set(fv)
                        val_var.set(f"{fv:.4g}" if is_float else str(int(fv)))
        except Exception:
            pass
        root.after(1000, _sync_from_file)

    root.after(1000, _sync_from_file)
    root.mainloop()
