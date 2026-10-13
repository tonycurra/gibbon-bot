# HAT swap

Why this RoArm-M3 Pro is not the stock driver board.

Tony swapped the kit **General Driver Board for Robots** for the **Bus Servo Driver HAT (A)** so a 3S LiPo can plug in on XT60.

- Stock board: https://www.waveshare.com/product/robotics/drivers-sensors/drivers/general-driver-for-robots.htm
- This HAT: https://www.waveshare.com/product/robotics/drivers-sensors/drivers/bus-servo-driver-hat-a.htm — wiki dump: [`SBSD-HAT.md`](SBSD-HAT.md)

**3S on XT60 is still the intended pack** (HAT 9–25 V). Healthy ~11.1 V: use it. Lab 2026-09-15 leftovers **5.6 V** and **8.2 V** — recycle. Packs: [`lipo.md`](lipo.md).

What we lost with that swap (do not look for them on the HAT): user LED, INA219, IMU, OLED. Sitting hardware page: [`../hardware.md`](../hardware.md).
