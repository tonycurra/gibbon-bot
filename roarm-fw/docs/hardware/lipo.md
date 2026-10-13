# Packs (XT60 LiPo)

**Yes — a healthy 3S LiPo on XT60 is still the right and safe pack** for RoArm-M3 + Bus Servo Driver HAT (A). Waveshare’s HAT input is **9–25 V**; ST3215s are 7–12.6 V. That is this machine.

No reverse-current diode does **not** make a ~11 V pack unsafe. Hand-moved motors (~5 V open-circuit on an unplugged XT60) cannot push into **10–11.1 V**. Recycle the dead leftovers (**5.6 V**, **8.2 V**). Do not plug puffy packs. Store good 3S at ~11.4 V.

HAT has **no BMS**. USB does **not** charge XT60. Each limb is one HAT + one 3S.

**2S is the wrong pack for this HAT** (max 8.4 V, under 9 V). The packs we metered on 2026-09-15 were **3S**.

Durable how-to. Sitting note: `development-logs/2026-09-15-packs.md`.

## Meter before HAT or charger

Rest voltage, pack **off** the robot, **multimeter** on the XT60 (not servo `batt` / USB 5.1 V). Puffy → recycle, never charge.

| Rest (3S) | Per cell | Meaning |
| --- | --- | --- |
| ~11.1–12.4 V | ~3.7–4.1 V | Use. Store at ~11.4 V (~3.8 V/cell) |
| ~9–10.5 V | ~3.0–3.5 V | Low. Balance-charge soon, watch it |
| **8.2 V** (lab, 2026-09-15) | ~2.73 V | Over-discharged. Recycle. Do not charge onto an arm |
| **5.6 V** (lab, 2026-09-15) | ~1.87 V | Dead. Recycle. Do not charge |
| **&lt; 9 V** or **0 V** | — | Same: dead for this robot. No “boost” onto an arm |

Firmware `batt` ignores VIN ≤6 V, so USB-only ~5.1 V looks like pack off. That 5.1 V is the buck on servo VIN, not a 3S.

## Storage kills more packs than regen

Leave a LiPo empty for months and it walks under ~3.0 V/cell: capacity loss, swell, packs that should never go on a charger. Store at ~40–50% (**3S ~11.4 V**), not 0% and not 100%.

**2026-09-15:** four-limb robot was assembled; two 3S short. Multimeter on the two leftovers: **5.6 V** and **8.2 V** (both 3S). That is ~1.9 V/cell and ~2.7 V/cell — storage over-discharge, not 4 V back-EMF charging an 11.1 V 3S.

## Back-EMF on an open XT60

With pack **unplugged**, `tq=0`, moving the arm by hand, XT60 can show **2–4 V** (open-circuit generator voltage). **4 V cannot push into a healthy 11.1 V pack.**

Still true:

- Do not short an empty XT60 while someone backdrives.
- A pack already at ~4 V is a different story (same order as BEMF).
- Faster backdrive raises BEMF; we have not treated that as proven charging.
- USB + pack together can flicker (5 V rail vs XT60). Flash on USB, then **unplug USB**, then pack on for BLE sittings.

Hold-on-plug (torque hold when VIN comes up) is still the right policy so the arm does not collapse. It is not the autopsy of the dead packs.

## Two packs now, four in southern Italy

One pack = one limb. Do not Y-split one 3S across two HATs (14× ST3215 stall will sag). Recycle the 5.6 V and 8.2 V 3S.

**Do not buy the extra two until two-pack lab tests are good enough** (ESTOP, hold/free/nudge on both limbs, VIN ~11 V, body supported). Then two more 3S XT60, then all four on the chassis in **southern Italy**. Plan: `docs/plans/now.md`.

Until those two arrive: flash and test **two limbs**, body **supported**. Do not leave the assembled robot hanging with two unpowered limbs (`tq=0` / no VIN = collapse).

Writes on the chip still cut at pack **10.5 V** (`V_CUTOFF`). That is not a fuel gauge.
