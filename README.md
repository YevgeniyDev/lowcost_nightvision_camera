# Low-Cost Lightweight Real-Time Night Vision VR System

This repository contains a lightweight **real-time low-light / NIR enhancement** pipeline written in **C++ (OpenCV)**, a simple **MJPEG streaming server** (HTTP multipart), and a **Unity 2020 client** (tested with HoloLens 2) that displays the stream.

The focus is **real-time feasibility on embedded or low-power devices**. The implementation benchmarks classical pipelines (CLAHE, bilateral/NLM+CLAHE, Retinex SSR variants) and two proposed pipelines, and produces:

* **CSV benchmarking results**
* **per-method AVI recordings**
* **combined AVI** (with method label overlay)
* **snapshots** at fixed timestamps (useful for paper figures)
* **live MJPEG stream** for remote visualization (Unity/HoloLens 2)

# Authors

1. **Zaki Al-Farabi**
2. **Yevgeniy Dikun**
3. **Mirat Serik**
4. **Yermek Zhakupov**

PI: **Zhanat Kappassov**

Co-PI: **Ilyas Tursunbek**

## Repository layout

```
.
├── unity/
│   ├── MJPEGStreamReader.cs
│   └── MJPEGHandler.cs
├── samples/
│   ├── results_scene100.csv
│   ├── combined_scene100.avi
│   ├── videos_scene100/
│   ├── snaps_scene100/
│   └── ...
└── night_vision_V2.cpp
└──README.md
```

## Methodology

### C++ code implementation (night_vision_V2.cpp)

### 1) Captures frames from a camera in real-time

* Default camera: `--cam 0`
* Fixed capture size: **640×480**
* Uses `cv::CAP_V4L2` and sets a small buffer to reduce latency.

### 2) Applies enhancement methods in sequence (time-sliced benchmarking)

Each method runs for `--seconds-per-method` seconds. During the first `--warmup` seconds, the program **does not log metrics** (to reduce bias from auto-exposure settling).

Methods implemented:

1. `RawGray`
2. `CLAHE`
3. `Bilateral+CLAHE`
4. `NLM+CLAHE`
5. `RetinexSSR`
6. `RetinexSSR_Pctl`
7. `Proposed`
8. `Proposed_V2`

### 3) Computes quality + runtime metrics

Logged per method:

* `avg_ms_per_frame`
* `fps`
* `entropy`
* `edge_strength` (mean gradient magnitude)
* `laplacian_var` (sharpness proxy)
* `rms_contrast`
* `mean_intensity` (helps detect near-black runs)
* `frames` (number of frames logged)

### 4) Creates videos, snapshots and CSVs

Depending on flags, it writes:

* Per-method videos: `videos_scene<N>/<MethodName>.avi`
* Combined video (with method label overlay): `combined_scene<N>.avi`
* Snapshots: `snaps_scene<N>/snap_<MethodName>_t<X>.png`
* CSV results: `results_scene<N>.csv`

Where N - object distance from camera and X - exact second the snapshot was taken.

### 5) Streams the *current* processed frame via MJPEG over HTTP

* Default server: `http://<host-ip>:8080/`
* Boundary: `boundarydonotcross`
* JPEG quality ≈ 50
* ~30 fps send pacing (simple `usleep(33000)`)

Unity/HoloLens 2 can connect to this stream over Wi-Fi LAN.

## Build

### Dependencies

* Linux
* C++17 compiler (g++ recommended)
* OpenCV (via `pkg-config opencv4`)
* pthread (standard on Linux)

Install necessary dependencies:

```bash
sudo apt update
sudo apt install -y build-essential pkg-config libopencv-dev v4l-utils
```

### Compile (single-file build)

From repo root:

```bash
g++ -std=c++17 night_vision_V2.cpp -o night_vision \
  $(pkg-config --cflags --libs opencv4) -lpthread
```

## Run

### Minimal run (creates defaults)

```bash
./night_vision
```

Command-line options (defaults):

  ```bash
  --seconds-per-method N     Default: 20
                             Duration (in seconds) to run each enhancement method.

  --warmup N                 Default: 2
                             Warm-up seconds per method (metrics are NOT logged during warmup).

  --out results.csv          Output CSV filename for metrics.

  --port 8080                Default: 8080
                             MJPEG streaming server port. Stream URL:
                             http://<host-ip>:8080/

  --cam 0                    Default: 0
                             Camera index (V4L2), e.g., /dev/video0.

  --record-per-method 0/1    Default: 1
                             Save per-method AVI files (one video per method).

  --record-combined 0/1      Default: 1
                             Save a single combined AVI with an on-frame method label.

  --video-dir DIR            Default: videos
                             Output directory for per-method videos.

  --combined-video FILE      Default: combined_run.avi
                             Filename for the combined video.

  --snapshots 0/1            Default: 1
                             Save snapshot PNGs at fixed timestamps per method.

  --snapshot-dir DIR         Default: snaps
                             Output directory for snapshots.

  --preexp 0/1               Default: 0
                             Optional pre-normalization of mean intensity
                             (helps avoid near-black runs under auto-exposure).
```

### Sample run
Example for **100 cm** object distance:

```bash
./night_vision \
  --seconds-per-method 20 \
  --warmup 2 \
  --cam 0 \
  --port 8080 \
  --out results_scene100.csv \
  --video-dir videos_scene100 \
  --combined-video combined_scene100.avi \
  --snapshots 1 \
  --snapshot-dir snaps_scene100 \
  --record-per-method 1 \
  --record-combined 1 \
  --preexp 0
```

## Output files and folders

After a run, you typically get:

### CSV (benchmark table)

Example:

* `results_scene100.csv`

Columns:

```
method,avg_ms_per_frame,fps,entropy,edge_strength,laplacian_var,rms_contrast,mean_intensity,frames
```

### Per-method videos (grayscale MJPEG AVI)

Example directory:

* `videos_scene100/`

  * `RawGray.avi`
  * `CLAHE.avi`
  * `Bilateral_CLAHE.avi`
  * `NLM_CLAHE.avi`
  * `RetinexSSR.avi`
  * `RetinexSSR_Pctl.avi`
  * `Proposed.avi`
  * `ProposedV2.avi`

### Combined run video

Example:

* `combined_scene100.avi`

This file records the processed frame **with an on-frame label**:
`<MethodName> | t=<elapsed>s`

### Snapshots

Example directory:

* `snaps_scene100/`

  * `snap_Proposed_t3.png`
  * `snap_Proposed_t8.png`
  * `snap_Proposed_t15.png`
  * and similarly for each method

Snapshot times are currently hardcoded:

* `{3, 8, 15}` seconds into each method window

## Live streaming (MJPEG)

The program starts an MJPEG server in a background thread.

* Default URL:

  * `http://<your-linux-ip>:8080/`

## Unity 2020 (HoloLens 2) client setup

This repo includes:

* `unity/MJPEGStreamReader.cs`
* `unity/MJPEGHandler.cs`

### 1) Create a simple UI scene

1. Create a new Unity 2020 project.
2. In the Scene:

   * `GameObject -> UI -> Canvas`
   * `GameObject -> UI -> RawImage`
3. Resize the RawImage to fill the Canvas (optional).

### 2) Add scripts

1. Create a folder `Assets/Scripts/`.
2. Copy both C# scripts into it:

   * `MJPEGStreamReader.cs`
   * `MJPEGHandler.cs`

### 3) Add a controller object

1. `GameObject -> Create Empty` and name it e.g. `MJPEGClient`.
2. Attach `MJPEGStreamReader` component to `MJPEGClient`.
3. Drag the `RawImage` object into the `outputImage` field in Inspector.
4. Set `streamURL` to your Linux host address, e.g.:

   * `http://192.168.1.25:8080/`

### 4) Run

* Press Play in Unity Editor (for a quick test on PC).
* For HoloLens 2:

  * Build using UWP workflow (standard HoloLens procedure).
  * Ensure the device is on the same LAN and can reach the Linux host.

## Networking tips (HoloLens 2 ↔ Linux PC)

Most issues are basic connectivity:

* Make sure both devices are on the **same Wi-Fi network**.
* Confirm the Linux device IP:

  ```bash
  ip a
  ```
* Confirm port 8080 is listening:

  ```bash
  ss -lntp | grep 8080
  ```
* If you use a firewall, allow the port (Ubuntu example):

  ```bash
  sudo ufw allow 8080/tcp
  ```
  
## License

Not chosen yet.

## Citation

If you use this code in academic work, please cite the associated paper.
