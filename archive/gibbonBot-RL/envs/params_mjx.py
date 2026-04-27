# MJX training parameters — tune these before scaling up to many envs.
# Edit values here or run this file directly to open the slider UI.
# Reward/motor/learning params hot-reload between rollouts.
# PHYSICS_STEPS_PER_ACTION requires a training restart (baked into JAX compile).

# --- timing ---
PHYSICS_STEPS_PER_ACTION = 200  # physics steps per policy action (16 × 0.002s = 0.032s)
VIEWER_SPEED = 1.000000  # how fast the viewer plays back: 1=real-time, 10=10× faster

# --- motor ---
MOTOR_MAX_SPEED = 0.070000  # max normalised joint travel per action

# --- goals ---
TARGET_Z = 0.300000  # torso height target (metres)
TARGET_TOL = 0.100000  # Gaussian reward width (sigma): small=tight, large=forgiving

# --- reward weights ---
W_HEIGHT = 10.000000  # scale of the height reward
W_UPRIGHT = 5.000000  # scale of the upright orientation reward

# --- arm spacing ---
ARM_SAFETY_MARGIN = 0.050000  # arms closer than this (metres) incur a penalty
W_PROXIMITY = 1.500000  # how strongly the arm-spacing penalty is applied

# --- termination ---
FLIP_PENALTY = 200.000000  # one-time penalty when robot flips upside-down

# --- learning (hot-reloadable between rollouts) ---
LEARNING_RATE = 0.000900  # Adam learning rate — reduce if training is unstable
ENT_COEF = 0.016000  # entropy bonus: 0=exploit, 0.02=explore more
CLIP_EPS = 0.150000  # PPO clip range: how much policy can change per update

# --- smoothness penalties (set to 0 to disable) ---
ACTION_SMOOTH_ALPHA = 0.300000  # action low-pass filter: filt = α·raw + (1-α)·prev. 1=no filter, 0.1=heavy
W_SMOOTH = 0.050000  # penalises action delta each step (filtered) — reduces twitching
W_JVEL = 0.001000  # penalises fast joint velocities — reduces flailing
W_ENERGY = 0.000500  # penalises sum(ctrl²) — discourages high motor effort
W_ALIVE = 0.050000  # small per-step bonus for staying alive (not flipped)


# ── slider UI (run this file directly to open) ────────────────────────────────
if __name__ == "__main__":
    import json
    import pathlib
    import tkinter as tk

    _FILE = pathlib.Path(__file__)
    _LIVE_STATS = _FILE.parent.parent / "runs" / "ape_stand_mjx" / "live_stats.json"

    # Each param tuple: (name, lo, hi, step, is_float, restart_required, live_key, description)
    # live_key (or None) names a key in the live_stats JSON to drive a feedback bar.
    # Categories are an ordered list of (name, default_open, [params...])
    _CATEGORIES = [
        ("⚡  Motion", True, [
            ("PHYSICS_STEPS_PER_ACTION", 4, 300, 1, False, True, None,
             "How many physics steps per policy decision (each step = 0.002s).\n"
             "Higher = more time per decision = smoother, more deliberate motion.\n"
             "16 = 0.032s/decision (fast train). 200 = 0.4s/decision (smooth).\n"
             "⚙ RESTART REQUIRED — baked into JIT compile."),
            ("MOTOR_MAX_SPEED", 0.01, 1.0, 0.01, True, False, "max_action_delta",
             "Max joint travel per action (normalised). Hard cap on motor speed.\n"
             "0.07 = slow/deliberate. 0.5 = fast/jerky.\n"
             "Live bar = actual max joint travel this action (bounded by this value).\n"
             "If bar is much lower than slider, policy isn't using full speed budget."),
            ("ACTION_SMOOTH_ALPHA", 0.05, 1.0, 0.05, True, False, None,
             "Action low-pass filter: filt = α·raw + (1−α)·prev.\n"
             "1.0 = raw policy. 0.3 = heavy smoothing (forces smooth motion)."),
            ("VIEWER_SPEED", 1.0, 100.0, 1.0, True, False, None,
             "Display speed only — does NOT affect physics or training.\n"
             "1 = real time. 10 = 10× faster. 100 = as fast as possible."),
        ]),
        ("🧊  Smoothness Penalties", True, [
            ("W_SMOOTH", 0.0, 2.0, 0.01, True, False, "smooth_delta",
             "Penalty per step on action change: sum((filt − prev_filt)²).\n"
             "0.05 = light, 0.5 = moderate, 2.0 = heavy.\n"
             "Live bar = mean |filt − prev_filt| this step."),
            ("W_JVEL", 0.0, 0.5, 0.001, True, False, "jvel_sq",
             "Penalty on fast joint velocities. Reduces flailing.\n"
             "0.001 = light. 0.05 = heavy.\n"
             "Live bar = mean(qvel²) this step."),
            ("W_ENERGY", 0.0, 0.05, 0.0001, True, False, "ctrl_sq",
             "Penalty on motor effort sum(ctrl²). Rewards efficient movement.\n"
             "5e-4 = light. 0.01 = heavy.\n"
             "Live bar = mean(ctrl²) this step."),
        ]),
        ("🎯  Goal Shaping", False, [
            ("TARGET_Z", 0.05, 0.60, 0.01, True, False, "torso_z",
             "Target torso height (metres).\n"
             "Live bar = current torso z."),
            ("TARGET_TOL", 0.005, 0.20, 0.005, True, False, None,
             "Gaussian sigma — width of the height-reward peak."),
            ("W_HEIGHT", 0.0, 10.0, 0.1, True, False, None,
             "Scale of the height reward (main objective)."),
            ("W_UPRIGHT", 0.0, 5.0, 0.05, True, False, "upright",
             "Reward for staying level (proportional to quat w²).\n"
             "Live bar = quat w (1.0 = perfectly upright)."),
        ]),
        ("💀  Survival", False, [
            ("W_ALIVE", 0.0, 0.5, 0.005, True, False, None,
             "Small per-step bonus for not flipping. Prevents early give-up."),
            ("FLIP_PENALTY", 0.0, 200.0, 1.0, True, False, None,
             "One-time penalty + episode termination when robot flips upside-down."),
        ]),
        ("🤝  Arm Safety", False, [
            ("ARM_SAFETY_MARGIN", 0.05, 0.50, 0.01, True, False, None,
             "Soft penalty fires when arms come within this distance (metres)."),
            ("W_PROXIMITY", 0.0, 20.0, 0.5, True, False, None,
             "Strength of the arm-spacing penalty."),
        ]),
        ("🧠  Learning", False, [
            ("LEARNING_RATE", 1e-5, 1e-3, 1e-5, True, False, None,
             "Adam optimiser step size. Hot-reloadable (rebuilds optimiser)."),
            ("ENT_COEF", 0.0, 0.05, 0.001, True, False, None,
             "Entropy bonus — rewards exploration."),
            ("CLIP_EPS", 0.05, 0.50, 0.01, True, False, None,
             "PPO clip range — limits per-update policy change."),
        ]),
    ]

    # Presets — each writes multiple params at once.
    _PRESETS = [
        ("Smooth", "#1a7f1a", {
            "PHYSICS_STEPS_PER_ACTION": 200,
            "MOTOR_MAX_SPEED":          0.07,
            "ACTION_SMOOTH_ALPHA":      0.30,
            "W_SMOOTH":                 0.05,
            "W_JVEL":                   0.001,
            "W_ENERGY":                 0.0005,
        }),
        ("Fast train", "#b87333", {
            "PHYSICS_STEPS_PER_ACTION": 16,
            "MOTOR_MAX_SPEED":          0.5,
            "ACTION_SMOOTH_ALPHA":      1.0,
            "W_SMOOTH":                 0.0,
            "W_JVEL":                   0.0,
            "W_ENERGY":                 0.0,
        }),
        ("Reset defaults", "#666666", {
            "PHYSICS_STEPS_PER_ACTION": 16,
            "VIEWER_SPEED":             1.0,
            "MOTOR_MAX_SPEED":          0.25,
            "ACTION_SMOOTH_ALPHA":      0.3,
            "TARGET_Z":                 0.30,
            "TARGET_TOL":               0.10,
            "W_HEIGHT":                 10.0,
            "W_UPRIGHT":                5.0,
            "ARM_SAFETY_MARGIN":        0.05,
            "W_PROXIMITY":              1.5,
            "FLIP_PENALTY":             200.0,
            "LEARNING_RATE":            0.0009,
            "ENT_COEF":                 0.016,
            "CLIP_EPS":                 0.15,
            "W_SMOOTH":                 0.05,
            "W_JVEL":                   0.001,
            "W_ENERGY":                 0.0005,
            "W_ALIVE":                  0.05,
        }),
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
    root.geometry("720x900")

    # Fonts (120% of standard sizes)
    F_NAME    = ("Courier", 12, "bold")
    F_DESC    = ("TkDefaultFont", 9)
    F_ENTRY   = ("Courier", 11)
    F_CAT     = ("TkDefaultFont", 13, "bold")
    F_PRESET  = ("TkDefaultFont", 10, "bold")
    F_STATS   = ("Courier", 11)
    F_STATS_L = ("TkDefaultFont", 9)

    # ── Header: preset buttons ───────────────────────────────────────────────
    header = tk.Frame(root, bg="#222222")
    header.pack(fill="x")
    tk.Label(header, text="Presets:", font=F_PRESET,
             fg="#cccccc", bg="#222222").pack(side="left", padx=(8, 4), pady=6)

    # ── Scrollable params area ───────────────────────────────────────────────
    container = tk.Frame(root)
    container.pack(fill="both", expand=True, padx=8, pady=4)
    canvas = tk.Canvas(container, width=700, highlightthickness=0)
    scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    inner = tk.Frame(canvas)
    canvas.create_window((0, 0), window=inner, anchor="nw")
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    # Mouse-wheel scroll on the params area
    def _on_wheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    canvas.bind_all("<MouseWheel>", _on_wheel)
    canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
    canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

    widgets = {}      # name -> (slider, entry_var, is_float)
    live_bars = {}    # name -> (canvas, fill_rect, hi, label_var)
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

    def _make_live_bar(parent, hi):
        """Returns (frame_with_canvas_and_label, canvas, fill_rect, label_var)."""
        frame = tk.Frame(parent)
        c = tk.Canvas(frame, width=120, height=10, bg="#dddddd",
                      highlightthickness=0)
        c.pack(side="left")
        rect = c.create_rectangle(0, 0, 0, 10, fill="#888888", width=0)
        var = tk.StringVar(value="—")
        tk.Label(frame, textvariable=var, font=F_STATS_L,
                 fg="#555555", width=8, anchor="w").pack(side="left", padx=(4, 0))
        return frame, c, rect, var

    def _make_param_row(parent, item):
        name, lo, hi, step, is_float, restart, live_key, desc = item
        val = current.get(name, lo)

        section = tk.Frame(parent, bd=1, relief="groove")
        section.pack(fill="x", pady=2, padx=4)

        title_row = tk.Frame(section)
        title_row.pack(fill="x", padx=6, pady=(5, 0))
        title = name + ("  ⚙" if restart else "")
        tk.Label(title_row, text=title, anchor="w", font=F_NAME,
                 fg=("#a52a2a" if restart else "#000000")).pack(side="left")

        tk.Label(section, text=desc, anchor="w", justify="left",
                 font=F_DESC, fg="#444444", wraplength=680).pack(
            fill="x", padx=6, pady=(0, 2))

        row = tk.Frame(section)
        row.pack(fill="x", padx=6, pady=(0, 5))

        slider = tk.Scale(row, from_=lo, to=hi, resolution=step,
                          orient="horizontal", length=380, showvalue=False)
        slider.set(val)
        slider.pack(side="left")

        entry_var = tk.StringVar(value=f"{val:.4g}" if is_float else str(int(val)))
        entry = tk.Entry(row, textvariable=entry_var, width=9, font=F_ENTRY,
                         justify="right")
        entry.pack(side="left", padx=(8, 4))
        widgets[name] = (slider, entry_var, is_float)

        if live_key:
            bar_frame, bar_canvas, bar_rect, bar_label = _make_live_bar(row, hi)
            bar_frame.pack(side="left", padx=(8, 0))
            live_bars[name] = (bar_canvas, bar_rect, hi, bar_label, live_key)

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

    # ── Build categories with collapsible sections ───────────────────────────
    category_states = {}  # cat_name -> {"open": bool, "body": Frame, "arrow_var": StringVar}

    def _make_category(name, default_open, params):
        wrap = tk.Frame(inner)
        wrap.pack(fill="x", pady=(8, 0))

        header_bar = tk.Frame(wrap, bg="#444444", cursor="hand2")
        header_bar.pack(fill="x", padx=4)
        arrow_var = tk.StringVar(value="▾" if default_open else "▸")
        tk.Label(header_bar, textvariable=arrow_var, font=F_CAT,
                 fg="#ffffff", bg="#444444", width=2).pack(side="left", padx=(8, 0), pady=4)
        tk.Label(header_bar, text=name, anchor="w", font=F_CAT,
                 fg="#ffffff", bg="#444444").pack(side="left", fill="x", expand=True, pady=4)

        body = tk.Frame(wrap)
        if default_open:
            body.pack(fill="x")

        for p in params:
            _make_param_row(body, p)

        state = {"open": default_open, "body": body, "arrow_var": arrow_var}
        category_states[name] = state

        def _toggle(_e=None, _state=state):
            _state["open"] = not _state["open"]
            _state["arrow_var"].set("▾" if _state["open"] else "▸")
            if _state["open"]:
                _state["body"].pack(fill="x")
            else:
                _state["body"].pack_forget()

        for w in (header_bar, *header_bar.winfo_children()):
            w.bind("<Button-1>", _toggle)

    for cat_name, default_open, params in _CATEGORIES:
        _make_category(cat_name, default_open, params)

    # ── Preset buttons (header) ──────────────────────────────────────────────
    def _apply_preset(values):
        _user_moving[0] = True
        for name, val in values.items():
            if name not in widgets:
                continue
            slider, entry_var, is_float = widgets[name]
            slider.set(val)
            entry_var.set(f"{val:.4g}" if is_float else str(int(val)))
            _write(name, val, is_float)
        _last_mtime[0] = _FILE.stat().st_mtime
        _user_moving[0] = False

    for label, color, values in _PRESETS:
        tk.Button(header, text=label, font=F_PRESET, fg="#ffffff", bg=color,
                  activebackground=color, bd=0, padx=10, pady=2,
                  command=lambda v=values: _apply_preset(v)).pack(
            side="left", padx=4, pady=6)

    # ── Bottom stats panel ───────────────────────────────────────────────────
    stats_panel = tk.Frame(root, bg="#222222")
    stats_panel.pack(fill="x", side="bottom")
    tk.Label(stats_panel, text="LIVE  (from watch_mjx)", font=F_PRESET,
             fg="#888888", bg="#222222").pack(anchor="w", padx=8, pady=(6, 0))

    stats_grid = tk.Frame(stats_panel, bg="#222222")
    stats_grid.pack(fill="x", padx=8, pady=(2, 8))

    stats_vars = {}
    def _add_stat(label):
        row = tk.Frame(stats_grid, bg="#222222")
        row.pack(fill="x", pady=1)
        tk.Label(row, text=label, font=F_STATS_L, fg="#aaaaaa", bg="#222222",
                 width=18, anchor="w").pack(side="left")
        var = tk.StringVar(value="—")
        tk.Label(row, textvariable=var, font=F_STATS, fg="#1aff1a",
                 bg="#222222", anchor="w").pack(side="left", fill="x")
        stats_vars[label] = var

    for lbl in ("checkpoint:", "torso_z:", "upright:", "ep_len:",
                "flip_rate (20):", "avg_ep_len (20):"):
        _add_stat(lbl)

    # ── Live stats poller ────────────────────────────────────────────────────
    def _poll_stats():
        try:
            if _LIVE_STATS.exists():
                stats = json.loads(_LIVE_STATS.read_text())

                # Update bottom panel
                stats_vars["checkpoint:"].set(stats.get("ckpt", "—"))
                stats_vars["torso_z:"].set(f"{stats.get('torso_z', 0):.3f} m")
                stats_vars["upright:"].set(f"{stats.get('upright', 0):.3f}")
                stats_vars["ep_len:"].set(f"{stats.get('ep_len', 0)} / 2000")
                fr = stats.get("flip_rate_20", 0.0) * 100
                n  = stats.get("n_eps", 0)
                stats_vars["flip_rate (20):"].set(f"{fr:.0f}%   (n={n})")
                stats_vars["avg_ep_len (20):"].set(f"{stats.get('avg_ep_len_20', 0):.0f}")

                # Update live bars
                for name, (c, rect, hi, label_var, key) in live_bars.items():
                    val = float(stats.get(key, 0.0))
                    frac = max(0.0, min(1.0, val / hi if hi > 0 else 0))
                    width = int(120 * frac)
                    # Colour by saturation
                    if frac < 0.5:
                        col = "#1a7f1a"
                    elif frac < 0.85:
                        col = "#cc8800"
                    else:
                        col = "#cc1a1a"
                    c.coords(rect, 0, 0, width, 10)
                    c.itemconfig(rect, fill=col)
                    label_var.set(f"{val:.3g}")
        except Exception:
            pass
        root.after(200, _poll_stats)

    # ── File-mtime sync (catch external edits) ───────────────────────────────
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
    root.after(200, _poll_stats)
    root.mainloop()
