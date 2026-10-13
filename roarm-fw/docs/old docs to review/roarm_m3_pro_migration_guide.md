# RoArm-M3 Pro Custom Migration Guide

**Hardware Reference & Technical Brief:** Waveshare Bus Servo Driver HAT (A)

> **CRITICAL DEPLOYMENT NOTE:** Ensure the onboard physical UART control switch is flipped completely to the ESP32 position. If left on the USB or Pi positions, the data pins will be isolated and no communication will occur.

## 1. Core System Architecture

This document details the configuration requirements for your modified layout using the Waveshare Bus Servo Driver HAT (A) with a native XT60 battery layout powering the ST3215 series serial bus servos.

### Hardware Pin Assignments

- **S_RXD (Receive Line):** GPIO 18
- **S_TXD (Transmit Line):** GPIO 19
- **Bus Baud Rate:** 1,000,000 bps (1 Mbps) half-duplex

### Robotic Arm Servos ID Mapping

The RoArm-M3 Pro populates 7 independent joint nodes mapped to IDs 11 through 17:

- **ID 11:** Base rotation joint
- **ID 12:** Shoulder joint (Primary Drive A)
- **ID 13:** Shoulder joint (Secondary Drive B - Dual-drive mirror)
- **ID 14:** Elbow joint
- **ID 15:** Wrist joint 1 (Pitch/Elevation)
- **ID 16:** Wrist joint 2 (Roll/Rotation)
- **ID 17:** End effector (Gripper claw)

## 2. Verified Deployment Scripts

### Script A: Hardware Bus Scanner

Sweeps the half-duplex network channel to confirm power delivery and connection status across all servo nodes.

```cpp
#include <SCServo.h>

#define S_RXD 18
#define S_TXD 19

SMS_STS st;

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  Serial1.begin(1000000, SERIAL_8N1, S_RXD, S_TXD);
  st.pSerial = &Serial1;

  Serial.println("\n=================================");
  Serial.println("   OFFICIAL PIN ID PIN SCANNER   ");
  Serial.println("=================================");
}

void loop() {
  bool motorFound = false;

  for (int id = 1; id <= 253; id++) {
    int responseID = st.Ping(id);
    if (responseID != -1) {
      Serial.print("[FOUND] Motor responding at ID: ");
      Serial.println(responseID);
      motorFound = true;
    }
  }

  if (!motorFound) {
    Serial.println("[FAIL] Still silence. Verify the battery switch on the board is ON.");
  }

  Serial.println("---------------------------------");
  delay(5000);
}
```

### Script B: Secure Manual Position Controller

Monitors standard serial commands to safely control joint movement only upon explicit user validation.

```cpp
#include <SCServo.h>

#define S_RXD 18
#define S_TXD 19

SMS_STS st;

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  Serial1.begin(1000000, SERIAL_8N1, S_RXD, S_TXD);
  st.pSerial = &Serial1;

  Serial.println("\n=================================");
  Serial.println("   FINAL MOTOR MOVEMENT TEST     ");
  Serial.println("=================================");
  Serial.println("Type 'GO' and press Enter to nudge Motor 11.");
  Serial.println("Type 'BACK' and press Enter to return it.");
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input.equalsIgnoreCase("GO")) {
      Serial.println("Sending movement command: Move to Position 2300");
      st.WritePosEx(11, 2300, 1000, 50);
    }
    else if (input.equalsIgnoreCase("BACK")) {
      Serial.println("Sending movement command: Return to Position 2048");
      st.WritePosEx(11, 2048, 1000, 50);
    }
  }
}
```

### Script C: Telemetry Stream Logger

Queries the active register blocks to print real-time physical telemetry parameters to the Serial Monitor.

```cpp
#include <SCServo.h>

#define S_RXD 18
#define S_TXD 19

SMS_STS st;
int targetID = 11;

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  Serial1.begin(1000000, SERIAL_8N1, S_RXD, S_TXD);
  st.pSerial = &Serial1;
}

void loop() {
  if (st.FeedBack(targetID) != -1) {
    int pos     = st.ReadPos(-1);
    int speed   = st.ReadSpeed(-1);
    int load    = st.ReadLoad(-1);
    int voltage = st.ReadVoltage(-1);
    int temper  = st.ReadTemper(-1);
    int current = st.ReadCurrent(-1);
    int moving  = st.ReadMove(-1);

    Serial.print("ID: "); Serial.print(targetID);
    Serial.print(" | Pos: "); Serial.print(pos);
    Serial.print(" | Speed: "); Serial.print(speed);
    Serial.print(" | Load: "); Serial.print(load);
    Serial.print(" | Volt: "); Serial.print((float)voltage / 10.0); Serial.print("V");
    Serial.print(" | Temp: "); Serial.print(temper); Serial.print("C");
    Serial.print(" | Current: "); Serial.print(current); Serial.print("mA");
    Serial.print(" | Moving: "); Serial.println(moving ? "YES" : "NO");
  }
  delay(1000);
}
```

## 3. Telemetry Register Map & Scaling

| Metric | API Function Target | Expected Range | Scaling Math | System Utility |
| --- | --- | --- | --- | --- |
| Position | `st.ReadPos(-1)` | 0 to 4095 | 360° absolute system | Joint trajectory planning |
| Speed | `st.ReadSpeed(-1)` | Dynamic | Steps / Second | Velocity profile monitoring |
| Load | `st.ReadLoad(-1)` | -1000 to 1000 | Raw Directional Strain | Collision and stall cutoff |
| Voltage | `st.ReadVoltage(-1)` | 90 to 250 | Value ÷ 10 = Volts | Critical LiPo level safety tracking |
| Temperature | `st.ReadTemper(-1)` | 0 to 100 | Direct Degrees Celsius | Overheat circuit trip detection |
| Current | `st.ReadCurrent(-1)` | Dynamic | Milliamps (mA) | Real-time current draw metrics |
| Move Flag | `st.ReadMove(-1)` | 0 or 1 | Boolean State Indicator | Task step sequencer blocking |

## 4. Advanced Safety Logic Implementation

### Low-Voltage LiPo Protection Circuit

Add this safety check to your execution loop to instantly cut torque lock and save your 3S battery cells from discharging below safe structural critical minimums (< 10.5V):

```cpp
int rawVoltage = st.ReadVoltage(11);
float batteryVolts = (float)rawVoltage / 10.0;

if (batteryVolts < 10.5 && rawVoltage != -1) {
    Serial.println("CRITICAL ERROR: LIPO VOLTAGE LOW! RELEASING TORQUE.");
    for(int i = 11; i <= 17; i++) {
        st.EnableTorque(i, 0); // Completely releases structural hold to protect cells
    }
    while(1); // Lock execution path
}
```

### Active Collision Obstruction Cutoff

Monitor real-time torque feedback thresholds to instantly isolate motor power if physical obstruction resistance spikes:

```cpp
int currentLoad = st.ReadLoad(11);
if (abs(currentLoad) > 400) {
    Serial.println("WARN: Structural obstruction encountered! Halting joint sequence.");
    // Insert customized mechanical mitigation routine here
}
```
