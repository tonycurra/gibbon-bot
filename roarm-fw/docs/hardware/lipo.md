# Packs (XT60 LiPo)

HAT input is **9–25 V**. Each limb is one HAT + one **3S** pack (~11.1–12.6 V rest, gold ~11.4 V). USB does **not** charge XT60. This HAT has **no BMS** and **no reverse-current block** on XT60.

**2S is the wrong pack.** 2S tops out at 8.4 V, under the 9 V HAT minimum. Do not plug it in.

Durable how-to. Sitting note: `development-logs/2026-09-15-packs.md`.

## Meter before HAT or charger

Rest voltage, pack **off** the robot. Puffy → recycle, never charge.

| Rest (3S) | Meaning |
| --- | --- |
| ~11.1–12.4 V | Use. Store at ~11.4 V (~3.8 V/cell) |
| ~9–10.5 V | Low. Balance-charge soon, watch it |
| **&lt; 9 V** or **0 V** | Dead for this robot. Recycle. Do not “boost” onto an arm |

| Rest (2S) | Meaning |
| --- | --- |
| ~8 V | Alive (~4.0 V/cell). Charge as **2S**, store ~7.6 V. **Not for this HAT** |
| ~5 V | Over-discharged. Do not put on XT60 |

Firmware `batt` ignores VIN ≤6 V, so USB-only ~5.1 V looks like pack off. That 5.1 V is the buck on servo VIN, not a 3S.

## Storage kills more packs than regen

Leave a LiPo empty for months and it walks under ~3.0 V/cell: capacity loss, swell, packs that should never go on a charger. Store at ~40–50% (**3S ~11.4 V**), not 0% and not 100%.

**2026-09-15:** four-limb robot was assembled; two packs short. One swollen, others low (5 V / 8 V, including a 2S). That matches sitting uncharged, not 4 V back-EMF charging an 11.1 V 3S.

## Back-EMF on an open XT60

With pack **unplugged**, `tq=0`, moving the arm by hand, XT60 can show **2–4 V** (open-circuit generator voltage). **4 V cannot push into a healthy 11.1 V pack.**

Still true:

- Do not short an empty XT60 while someone backdrives.
- A pack already at ~4 V is a different story (same order as BEMF).
- Faster backdrive raises BEMF; we have not treated that as proven charging.
- USB + pack together can flicker (5 V rail vs XT60). Flash on USB, then **unplug USB**, then pack on for BLE sittings.

Hold-on-plug (torque hold when VIN comes up) is still the right policy so the arm does not collapse. It is not the autopsy of the dead packs.

## Two packs, four limbs

One pack = one limb. Do not Y-split one 3S across two HATs (14× ST3215 stall will sag).

Until replacements arrive: flash and test **two limbs at a time**, body **supported** or limbs **off the chassis**. Do not leave the assembled robot hanging with two unpowered limbs (`tq=0` / no VIN = collapse, and open XT60 BEMF).

Order **3S XT60** packs (plus a spare). Keep the 2S for some other 2S job, or recycle if 5 V / puffy.

Writes on the chip still cut at pack **10.5 V** (`V_CUTOFF`). That is not a fuel gauge.
