# Hand-move and hand limits

USB + XT60. Encoders first (`tq=0`, you backdrove). Then walls. Then I moved the hand inside the soft band. Software: `docs/software.md`. Order: `docs/plans/now.md`.

## Backdrive

`scripts/watch-hand.py`. `pos` and `speed` live. `load` / `moving` stay 0 with torque off. Shoulder 12+13 opposite. Gold: `docs/baselines/hand-move-usb-battery-on.md`. Raw: `docs/baselines/data/hand-move-watch.txt`.

## Walls (you)

Open **609** (cable). Close **2088** (jaws). 569 was past the open wall. Gold: `docs/baselines/hand-open-close.md`.

## Commanded (me)

`hand open` 689, `hand close` 2008, `hand to` up to 2075, `hand free`. Only ID 17. Contact CSV: `docs/baselines/data/hand-contact.csv`. `load` is effort, not a torque cell. Conservative 2008 is not jaw contact; ~2073 is.

## Next

Twist, same pattern. `docs/plans/joint-limits.md`.
