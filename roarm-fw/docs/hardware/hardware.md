in this roarm m3 pro, the 

## Modifications 
swapped this 
Onboard Interfaces on General Driver Board for Robots (https://www.waveshare.com/product/robotics/drivers-sensors/drivers/general-driver-for-robots.htm?___SID=U) default one that comes with roarm m3 pro

to replace with this: 
roarm-fw/docs/hardware/SBSD-HAT.md 
Serial Bus Servo Driver HAT, Integrates ESP32 and Servo Control Circuit, Suitable for ST / RSBL Series Serial Bus Servos

https://www.waveshare.com/product/robotics/drivers-sensors/drivers/bus-servo-driver-hat-a.htm?___SID=U

so we can attach battery lipo 3s via the XT60 connector.

**3S only.** HAT is 9–25 V; 2S tops out at 8.4 V. Packs, storage, what 5 V / 8 V on the meter means: `lipo.md`.

What we lost with that swap (do not look for them on the HAT): user LED, INA219, IMU, OLED. See `docs/hardware.md`.
 

