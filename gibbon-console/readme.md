Laptop UI for Gibbon. Talks the USB/BLE **text pipe** to a limb chip. Not firmware — that is `roarm-fw` on the ESP32.

Same page later for arm_l / arm_r / leg_l / leg_r. Right now one RoArm on ACM0.

```bash
cd /home/t/code/gibbon-bot/gibbon-console
python3 -m http.server 8080
```

Chrome → http://127.0.0.1:8080 → Connect USB. Close any other ACM0 process first.
