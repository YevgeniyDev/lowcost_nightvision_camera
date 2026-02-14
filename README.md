---

# Low-Cost Night Vision Camera

### ROBT310 — Image Processing

### Nazarbayev University · SEDS · December 2025

---

## 🛰 PROJECT OVERVIEW

This repository contains a complete, real-time **night-vision enhancement and streaming system** implemented on an NVIDIA Jetson and integrated with Unity on HoloLens 2.

The system includes:

* **Fisrt C++ applications** for capturing & enhancing the camera feed and streaming it with the MJPEG stream server
* **Second C++ application** for showing the image preprocessing pipeline's effect on the initial image via 6 different streams combined into one
* **Unity C# scripts** for streaming into HoloLens 2 via HTTP

---

# 👥 TEAM MEMBERS

1. **Mirat Serik**
2. **Yevgeniy Dikun**
3. **Yermek Zhakupov**
4. **Zaki Al-Farabi**

* Course: **ROBT310 — Image Processing**
* Instructor: Zhanat Kappassov
* Date: **December 2025**

---

# 📷 NIGHT VISION PIPELINE

Both programs implement the same enhancement pipeline:

1. **180° Rotation**
2. **Grayscale Conversion**
3. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
4. **Gaussian Smoothing (3×3)**
5. **3×3 spatial sharpening filter (unsharp-like kernel)**
6. **Brightness Boosting +25%**

This pipeline is tailored for real-time low-light visibility, providing a stable night vision output.

---

# 🚀 PROGRAM 1 — `night_vision.cpp`

### (Final Real-Time Streaming System for HoloLens / Unity)

This is the primary deployment program.

## ✔ Features

* Real-time adaptive night-vision enhancement
* MJPEG streaming server on
* Multi-client support (browser, Unity, HoloLens)
* Thread-safe design (camera loop separated from network threads)

## ✔ Build

```bash
g++ -std=c++17 night_vision.cpp -o night_vision \
`pkg-config --cflags --libs opencv4` -lpthread
```

## ✔ Run

```bash
./night_vision
```

## ✔ View stream

Retrieve Jetson IP:

```bash
ip -4 addr show wlan0 | grep inet | awk '{print $2}' | cut -d/ -f1
```

Then open:

```
http://<jetson-ip>:8080/
```

---

# 🧪 PROGRAM 2 — `night_vision_demo.cpp`

### (Pipeline Visualization / Educational Mode)

This program outputs a **3×2 grid**, showing:

| Grid Cell | Visualization       |
| --------- | ------------------- |
| 1         | Raw grayscale input |
| 2         | CLAHE enhanced      |
| 3         | Denoised            |
| 4         | Sharpened           |
| 5         | Brightness-boosted  |
| 6         | Final output        |

Useful for demonstrations and parameter tuning.

## ✔ Build

```bash
g++ -std=c++17 night_vision_demo.cpp -o night_vision_demo \
`pkg-config --cflags --libs opencv4` -lpthread
```

## ✔ Run

```bash
./night_vision_demo
```

---

# 🌐 MJPEG STREAMING SERVER

### Runs inside both programs

* accessible at: **http://jetson-ip:8080/**
* format: `multipart/x-mixed-replace`
* boundary: `--boundarydonotcross`
* JPEG quality: ~50%
* ~30 FPS output

---

# 🎮 UNITY / HOLOLENS INTEGRATION

Unity reads the MJPEG stream and displays frames inside the scene.

Two scripts are included in the repository:

### **1. `MJPEGHandler.cs`**

* Custom `DownloadHandlerScript`
* Parses MJPEG multipart stream
* Extracts JPEG frames
* Emits frames via `OnFrameComplete` event

### **2. `MJPEGStreamReader.cs`**

* Connects to Jetson MJPEG server
* Receives frames from `MJPEGHandler`
* Converts JPEG bytes to `Texture2D`
* Displays output on a Unity `RawImage`
* Fully compatible with **HoloLens 2 (UWP)**

## ✔ Setup in Unity (Steps)

1. Create a **Canvas**
2. Add a **RawImage** UI element
3. Add `MJPEGStreamReader.cs` to a GameObject
4. Assign the RawImage
5. Set the Jetson stream URL:

```
http://<jetson-ip>:8080/
```

6. Build for **UWP**
7. Deploy to **HoloLens 2 via Visual Studio**

You now receive the real-time night-vision feed inside HoloLens.

---

# 🔧 JETSON NOTES

### Recommended for stability:

Disable USB autosuspend (optional):

```bash
sudo sh -c "echo -1 > /sys/module/usbcore/parameters/autosuspend"
```

Allow MJPEG server port:

```bash
sudo ufw allow 8080
```

Ensure **VPN is OFF** on client devices and Jetson + HoloLens are on the **same Wi-Fi network**.

---

# 📁 PROJECT STRUCTURE

```
/project
 ├── night_vision.cpp              # Main real-time streaming system
 ├── night_vision_demo.cpp         # Visualization / pipeline testing
 ├── MJPEGHandler.cs               # Unity MJPEG parser
 ├── MJPEGStreamReader.cs          # Unity frame renderer
 ├── README.md                     # Documentation
```

---

# 🏁 CONCLUSION

This project integrates:

* Embedded Jetson-based imaging with image preprocessing
* Custom MJPEG networking
* Unity and HoloLens real-time AR visualization

