Serial Bus Servo Driver HAT, Integrates ESP32 and Servo Control Circuit, Suitable for ST / RSBL Series Serial Bus Servos

https://www.waveshare.com/product/robotics/drivers-sensors/drivers/bus-servo-driver-hat-a.htm?___SID=U

Wiki: www.waveshare.com/wiki/Bus_Servo_Driver_HAT_(A) 



SKU:27577
Part No.:Bus Servo Driver HAT (A)
Brand:Waveshare



Serial Bus Servo Driver HAT, Integrates ESP32 and Servo Control Circuit, Suitable for ST / RSBL Series Serial Bus Servos
Serial bus servo driver HAT

Designed for Serial Bus Servos, Integrates ESP32 and Servo Control Circuit

Allows controlling up to 253 serial bus servos at the same time via ESP32 UART or USB port, can be used as the main controller of the robotics projects
Serial Bus Servo Driver HAT, front view
Features at a glance

    Allows controlling up to 253 ST / RSBL series serial bus servos at the same time
    9~25V wide voltage input (the input voltage and the servo voltage must be matched)
    Based on the ESP32-WROOM-32 module, supports wired and wireless communication
    Provides multiple control demos for different host controllers
    Supports controlling the serial bus servos directly via onboard USB Type-C port
    Supports connecting to Raspberry Pi, powering the Pi via integrated 5V buck regulator circuit and communicating through GPIO UART interface, neat cable management



Specifications
Controller 	ESP32-WROOM-32
Power supply Port 	XT60 connector, 5.5×2.5 mm DC jack
Power Supply 	9~25V (the input voltage and the servo voltage must be matched)
Communication interface 	UART, USB Type-C port × 2
Wireless communication 	2.4G WiFi
Mounting hole diameter 	3 mm
mounting hole spacing 	58×49 mm
PCB size 	65×57 mm

Application examples

    Servos will provide various feedback like position, speed, torque lock, and operating mode (servo mode, servo motor mode, etc.) for advanced projects requiring closed-loop automatic control
    The host controller can control the serial bus servos by sending JSON commands and obtain feedback information in JSON format, more convenient for secondary development.
    Ideal choice for building quadruped robots, hexapod walkers, robotic arms and other robotic projects requiring multiple servos

Serial Bus Servo Driver HAT for building quadruped robots
Serial Bus Servo Driver HAT for building hexapod walkers

* images here are for reference ONLY

Compatible Servos

Supports ST / RSBL series bus servos and selecting power supply input according to the servo used

ST3215 Servo


ST3215-HS
Servo Motor

ST3025 Servo


ST3020 Servo



RSBL45-24


RSBL85-12


RSBL85-24


RSBL120-24

Note: When using the different servo models at the same time, please pay attention to whether the power input can meet the requirements of all the servos.

UART serial bus control

Allows controlling up to 253 serial bus servos at the same time
Serial Bus Servo Driver HAT, supports UART serial bus control

* for reference only, the servos in the picture are NOT included.

Note: though one channel UART can control up to 253 bus servos, due to the high power of the servos, it is necessary to consider whether the power supply solution is enough when using too many servos.

Wired Control Mode

Supports sending JSON commands to ESP32 for controlling the serial bus servos, or directly sending control commands to the servos via USB Type-C port. Comes with Python demo.
Serial Bus Servo Driver HAT, supports Wired Control Mode
Wireless Control Mode

The Bus Servo Driver HAT (A) Will Automatically Establish A Hotspot after Powering on, Allows Users To Connect And Control The Driver board Via Mobile Phones, Tablets And PCs with the Cross-Platform Web Application we provided
Serial Bus Servo Driver HAT, supports Wireless Control Mode

Suitable for Raspberry P 5 / 4B

Supports powering the Pi and communicating through GPIO UART interface, neat cable management
Serial Bus Servo Driver HAT, suitable for Raspberry P 5 / 4B Serial Bus Servo Driver HAT, pin definition

ESP32 Controller, Extensive Resources
Easy Secondary Development

Built-In WiFi And Bluetooth, As Well As ESP-NOW Support, For Remote Control And Servo Debugging
Serial Bus Servo Driver HAT, onboard ESP32 controller
What's On Board
Serial Bus Servo Driver HAT, onboard resources

    ESP32 module
    DCDC 5V buck regulator
    UART control switch
    Reset button
    BOOT button
    for entering downloading mode
    Raspberry Pi 40PIN header
    ESP32 Type-C port

    Serial bus servo Type-C port
    XT60 connector
    5.5×2.5 DC power jack
    Power supply screw terminal
    RS485 serial bus servo header
    TTL serial bus servo headers


Outline dimensions
Serial Bus Servo Driver HAT, outline dimensions
Resources & Services
Wiki: www.waveshare.com/wiki/Bus_Servo_Driver_HAT_(A)



https://www.waveshare.com/wiki/Bus_Servo_Driver_HAT_(A)


Bus Servo Driver HAT (A)
Bus Servo Driver HAT (A)
Bus Servo Driver HAT (A).jpg

UART/USB/TTL/RS485
Introduction

It is a driver board designed specifically for bus servos, integrating ESP32 and bus servo control circuits, and onboard RS485 and TTL bus servo interfaces. We have open-sourced the bus servo control demo, allowing users to connect via Wi-Fi and control the bus servos through a web interface, while also obtaining feedback from the servos (such as torque, speed, position, etc.).

Users can also redevelop it and use it as a lower computer for robot projects.


Specifications

    Controller: ESP32-WROOM-32
    Power supply port: XT60, 5.5×2.5 mm DC
    Power supply: 9~25V (Input voltage must match the servo voltage)
    Communication interface: UART, Type-C × 2
    Wireless communication: 2.4G WiFi
    Mounting hole diameter: 3 mm
    Mounting hole spacing: 58×49 (mm)
    PCB size: 65×57 (mm)


Features

    Allows controlling up to 253 ST / RSBL series serial bus servos at the same time (sufficient power supply required)
    9~25V wide voltage input (the input voltage and the servo voltage must be matched)
    Based on the ESP32-WROOM-32 module, supports wired and wireless communication
    Provides multiple control demos for different host controllers
    Supports controlling the serial bus servos directly via onboard USB Type-C port
    Supports connecting to Raspberry Pi, powering the Pi via integrated 5V buck regulator circuit and communicating through GPIO UART interface, neat cable management


Open Source Project

You can download the related open-source robot models in the product documentation to build your own project.

Servo-Driver-with-ESP32-details-3.jpg
Product Usage
USB Mode

When the driver board is in USB mode, you can use the Python example provided to control the bus servo directly. We will describe the steps to run the Python example in this section.
Python Environment Deployment On Windows

First, go to Python official website to download the latest version of Python installation package. You can download it according to your operating system, I downloaded it here for Windows, the version is 3.11.5.
Python安装.png

After downloading, double-click the installation program python-3.11.5-amd64, click "Customize installation" to enter the "Optional Features" optional feature interface. Keep "Next" checked in the interface to enter the "Advanced Options" interface. Be sure to keep "Add Python to environment variables" checked, and click "Browse" to modify the installation address. Here is the default installation address. Click "Install" to install after setting, and then just wait for the installation to be completed.
Python安装1.png

Once installed, let's move on to the Python project compilation example.
ST Series Servo Python Demo

Download ST serial bus servo control library (Python) demo, unzip, and then input "cmd" in the start menu bar to open the Windows command prompt interface, type "cd folder path" to enter the STServo_Python project folder, where the virtual environment has been created, the name is stservo-env, and then enter:

stservo-env\Scripts\activate.bat

This command is to activate the virtual environment.
Python编译.png

In the STServo_Python project, you can see the "requirements.txt" file, which is used to store the list of installation packages that the project will use. After entering the virtual environment, input:

python <nowiki>-m<nowiki> pip install -r requirements.txt

Use this command to install the installation package used by the project. As you can see, the only installation package used here is "pyserial" and installed.
Python编译1.png

Connect the ST series servo to the bus servo interface on the driver board, and supply it with 9~12.6V power. In the Start menu bar, search for Device Manager to view the newly connected port number.
Python编译-4.png

Next, change the name of the device in the demo you want to run to the access port number, in my case COM11 (different computers have different access port numbers).
Python编译2.png

Once modified you can run it, here is an example of the ping.py file, enter:

python ping.py

You can see it runs successfully and the servo with ID 1 is connected successfully.
Python编译3.png

These is a simple tutorial for the Python demo.
ESP32 Mode
ESP32 Transparent Transmission Mode​

Click Servo transparent transmission mode ESP32 download tool link to download, unzip it after downloading, and double-click to open the "flash_download_tool_3.9.5.exe" program. After opening, two windows will pop up, we need to operate the UI interface of the download tool, and the other window will act as a terminal to show the working status of the download tool.

    In the "DOWNLOAD TOOL MODE" interface, select ChipType as ESP32 and WorkMode as Factory. When calling the binary file using Factory, relative paths will be used, so there is no need for the user to manually input the binary file path, click OK after selection.

300px-STServo1.png

    In this software interface, keep "LockSettings" checked, the right side indicating that programs can be uploaded to 8 boards simultaneously. Then use a USB cable to connect the servo driver board with the computer, click "COM", and select the new COM (the new COM here is COM3); BAUD is used to set the download speed, the higher the speed, the faster it is, ESP32 can use up to 921600.

1000px-STServo2.png

    After selection, click START to start the upload process, after the upload is completed, "IDLE" will change to "FINISH".

STServo3.pngSTServo4.png

After flashing the above firmware, when the driver board is in the ESP32 mode, it can be controlled via the Raspberry Pi serial port. The driver board is alternately connected to the Raspberry Pi RX-TX and TX-RX, and a common ground connection is required.
ST Series Servo Raspberry Pi Demo

Download ST/SC serial bus servo control library (Linux) demo, unzip it, and then remotely upload it to the Raspberry Pi. Next enter "cd folder path" in the Raspberry Pi terminal to enter the SCServo_Linux project folder, then enter:

cmake .

The command is used to generate the Makefile.
600px-舵机1.png

Then enter:

make SCServo

This command generates the libSCServo.a static library file.
600px-舵机2.png

Then enter "cd folder path" to enter the directory of the WritePos demo file, and enter:

cmake .

The command is used to generate the Makefile.
600px-舵机3.png

Then enter according to the port of your device (here take the 40Pin port of Raspberry Pi 4B as an example):

sudo ./WritePos /dev/serial0

This command is used to drive the servo to rotate, and it is necessary to select the appropriate test example according to the servo model.
600px-舵机4.png


Resources
Schematic Diagram

    Bus Servo Driver HAT (A) Schematic

Model

    Bus Servo Driver HAT (A) STEP Model
    Bus Servo Driver HAT (A) 2D Drawing

Demos

    ST serial bus servo control library (Python)
    ST/SC serial bus servo control library (Arduino IDE)
    ST/SC serial bus servo control library (Linux)
    ST series STM32F103 SDK
    SC series servo demo
    ST series servo sample demo
    ESP32 download tool

Software

    Arduino IDE

Open Source Structure

    Open-source Robots

Support


