"""
Standalone params slider UI — launched as a subprocess by watch.py.
Reads/writes src/gibbonbot/envs/params.py directly.
"""

import sys
import pathlib
import tkinter as tk

PARAMS_FILE = pathlib.Path(__file__).parent.parent / "envs" / "params.py"

_PARAM_DEFS = [
    ("MOTOR_MAX_SPEED",          0.001, 0.10,   0.001,  True),
    ("MAX_ACTIVE_JOINTS",        1,     24,     1,      False),
    ("PHYSICS_STEPS_PER_ACTION", 10,    500,    10,     False),
    ("TARGET_Z",                 0.05,  0.60,   0.01,   True),
    ("TARGET_TOL",               0.005, 0.10,   0.005,  True),
    ("LIFTED_Z",                 0.02,  0.40,   0.01,   True),
    ("MIN_HEIGHT",               0.01,  0.15,   0.01,   True),
    ("STABLE_VEL",               0.01,  0.50,   0.01,   True),
    ("W_STABLE",                 0.0,   5.0,    0.1,    True),
    ("W_HEIGHT",                 0.0,   5.0,    0.1,    True),
    ("W_UPRIGHT",                0.0,   2.0,    0.05,   True),
    ("W_GOT_UP",                 0.0,   5.0,    0.1,    True),
    ("W_PLANT_LIFT",             0.0,   2.0,    0.05,   True),
    ("W_ALIVE",                  0.0,   1.0,    0.01,   True),
    ("W_ENERGY",                 0.0,   0.01,   0.0001, True),
    ("W_JVEL",                   0.0,   0.01,   0.0001, True),
    ("W_SMOOTH",                 0.0,   1.0,    0.01,   True),
    ("W_DRAG",                   0.0,   5.0,    0.1,    True),
    ("W_PROXIMITY",              0.0,   10.0,   0.1,    True),
    ("W_JOINTS",                 0.0,   2.0,    0.05,   True),
    ("ARM_SAFETY_MARGIN",        0.01,  0.20,   0.01,   True),
    ("FLIP_PENALTY",             0.0,   50.0,   1.0,    True),
]


def _read_params():
    vals = {}
    try:
        for line in PARAMS_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            v = v.split("#")[0].strip()
            try:
                vals[k.strip()] = float(v)
            except ValueError:
                pass
    except Exception:
        pass
    return vals


def _write_param(name, value, is_float):
    try:
        lines = PARAMS_FILE.read_text().splitlines()
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith(name) and "=" in s:
                indent = line[: len(line) - len(line.lstrip())]
                comment = ("  " + s[s.index("#"):]) if "#" in s else ""
                val_str = f"{value:.6g}" if is_float else str(int(round(value)))
                lines[i] = f"{indent}{name} = {val_str}{comment}"
                break
        PARAMS_FILE.write_text("\n".join(lines) + "\n")
    except Exception as e:
        print(f"[params-ui] write error: {e}", flush=True)


def main():
    current = _read_params()

    root = tk.Tk()
    root.title("Params")
    root.resizable(False, True)

    container = tk.Frame(root)
    container.pack(fill="both", expand=True, padx=8, pady=4)

    canvas = tk.Canvas(container, width=420)
    scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    inner = tk.Frame(canvas)
    canvas.create_window((0, 0), window=inner, anchor="nw")
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    for name, lo, hi, step, is_float in _PARAM_DEFS:
        val = current.get(name, lo)
        row = tk.Frame(inner)
        row.pack(fill="x", pady=1)

        tk.Label(row, text=name, width=26, anchor="w", font=("Courier", 9)).pack(side="left")

        val_var = tk.StringVar(value=f"{val:.4g}" if is_float else str(int(val)))
        tk.Label(row, textvariable=val_var, width=8, anchor="e", font=("Courier", 9)).pack(side="right")

        slider = tk.Scale(row, from_=lo, to=hi, resolution=step,
                          orient="horizontal", length=200, showvalue=False)
        slider.set(val)
        slider.pack(side="left", fill="x", expand=True)

        def _on_change(v, _name=name, _var=val_var, _is_float=is_float):
            fv = float(v)
            _var.set(f"{fv:.4g}" if _is_float else str(int(fv)))
            _write_param(_name, fv, _is_float)

        slider.configure(command=_on_change)

    root.mainloop()


if __name__ == "__main__":
    main()
