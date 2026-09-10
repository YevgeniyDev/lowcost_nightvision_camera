# Consumer-Grade Near-Infrared Vision with Real-Time Edge Enhancement for Mixed-Reality Visualization

This repository contains a lightweight **near-infrared / low-light enhancement pipeline** written in **C++ using OpenCV**, a simple **MJPEG streaming server** over HTTP, and a **Unity 2020 client** tested with HoloLens 2 for head-mounted mixed-reality visualization.

The system is designed to evaluate the feasibility of computationally lightweight enhancement on embedded hardware while supporting wireless remote visualization. The implementation benchmarks classical enhancement methods and two lightweight pipelines introduced in the associated paper:

- **LE-CLAHE** — Lightweight Edge-enhancement CLAHE
- **LE-Retinex** — Lightweight Retinex-integrated enhancement

The repository provides:

- **CSV benchmarking results**
- **per-method AVI recordings**
- **combined AVI recordings** with method labels
- **snapshots** at fixed timestamps
- **live MJPEG streaming**
- **Unity/HoloLens 2 visualization scripts**

## Repository layout

```text
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
├── night_vision_V2.cpp
└── README.md
```

## Methodology

### C++ implementation

The main implementation is contained in:

```text
night_vision_V2.cpp
```

### 1. Camera acquisition

The program captures frames from a camera in real time.

- Default camera index: `--cam 0`
- Capture resolution: **640 × 480**
- Capture backend: `cv::CAP_V4L2`
- Capture buffer size: **1 frame**

The small capture buffer is used to reduce accumulation of stale frames.

### 2. Enhancement methods

Each method runs for `--seconds-per-method` seconds. During the first `--warmup` seconds, metrics are not recorded in order to reduce the influence of camera exposure stabilization.

Implemented methods:

1. `RawGray`
2. `CLAHE`
3. `Bilateral+CLAHE`
4. `NLM+CLAHE`
5. `RetinexSSR`
6. `RetinexSSR_Pctl`
7. `Proposed` — corresponds to **LE-CLAHE** in the paper
8. `ProposedV2` — corresponds to **LE-Retinex** in the paper

### 3. LE-CLAHE

LE-CLAHE applies the following sequence:

1. Grayscale conversion
2. CLAHE
   - clip limit: `4.0`
   - tile grid: `8 × 8`
3. Gaussian smoothing
   - kernel: `3 × 3`
4. Sharpening
   - kernel:

```text
 0 -1  0
-1  5 -1
 0 -1  0
```

5. Linear intensity gain
   - gain: `1.25`

The pipeline is designed to improve local contrast while maintaining low computational cost.

### 4. LE-Retinex

LE-Retinex performs lightweight illumination normalization using:

1. Downsampling to `25%` of the original width and height
2. Illumination estimation using a `31 × 31` box filter
3. Bilinear upsampling of the illumination estimate
4. Retinex-like ratio correction
5. Mean/standard-deviation normalization using `k = 2.5`
6. Mild CLAHE
   - clip limit: `2.0`
   - tile grid: `8 × 8`
7. Median filtering
   - kernel: `3 × 3`
8. Detail-gated unsharp enhancement
   - Gaussian sigma: `1.0`
   - detail threshold: `6`
   - sharpening gain: `1.0`

### 5. Runtime and image statistics

For each method, the implementation records:

- `avg_ms_per_frame`
- `fps`
- `entropy`
- `edge_strength`
- `laplacian_var`
- `rms_contrast`
- `mean_intensity`
- `frames`

The reported `fps` value is derived from average processing time:

```text
1000 / avg_ms_per_frame
```

It therefore represents **algorithmic processing throughput**, not the physical camera frame rate or end-to-end streaming rate.

### 6. Video, snapshot, and CSV output

Depending on the selected flags, the program writes:

- Per-method videos: `videos_scene<N>/<MethodName>.avi`
- Combined labeled video: `combined_scene<N>.avi`
- Snapshots: `snaps_scene<N>/snap_<MethodName>_t<X>.png`
- CSV results: `results_scene<N>.csv`

where:

- `N` is the camera-to-target distance
- `X` is the snapshot time within the method window

### 7. MJPEG streaming

The current processed frame is streamed through an HTTP MJPEG server.

- Default port: `8080`
- JPEG quality: approximately `50`
- Transmission pacing: approximately `30 frame/s`
- Multipart boundary: `boundarydonotcross`

The Unity/HoloLens client connects to the stream through a shared Wi-Fi network.

## Build

### Dependencies

- Linux
- C++17 compiler
- OpenCV
- pthread

Install dependencies on Ubuntu:

```bash
sudo apt update
sudo apt install -y build-essential pkg-config libopencv-dev v4l-utils
```

### Compile

From the repository root:

```bash
g++ -std=c++17 night_vision_V2.cpp -o night_vision \
  $(pkg-config --cflags --libs opencv4) -lpthread
```

## Run

### Minimal run

```bash
./night_vision
```

### Command-line options

```text
--seconds-per-method N     Default: 20
                           Duration of each enhancement method.

--warmup N                 Default: 2
                           Warm-up period during which metrics are not logged.

--out results.csv          Output CSV filename.

--port 8080                MJPEG server port.

--cam 0                    Camera index.

--record-per-method 0/1    Default: 1
                           Save one AVI file per enhancement method.

--record-combined 0/1      Default: 1
                           Save a combined labeled AVI.

--video-dir DIR            Output directory for per-method videos.

--combined-video FILE      Combined video filename.

--snapshots 0/1            Default: 1
                           Save snapshots at fixed timestamps.

--snapshot-dir DIR         Snapshot output directory.

--preexp 0/1               Default: 0
                           Optional mean-intensity pre-normalization.
```

### Example run

Example for a target distance of **100 cm**:

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

## Output files

### CSV

Example:

```text
results_scene100.csv
```

Columns:

```text
method,avg_ms_per_frame,fps,entropy,edge_strength,laplacian_var,rms_contrast,mean_intensity,frames
```

### Per-method videos

Example:

```text
videos_scene100/
├── RawGray.avi
├── CLAHE.avi
├── Bilateral_CLAHE.avi
├── NLM_CLAHE.avi
├── RetinexSSR.avi
├── RetinexSSR_Pctl.avi
├── Proposed.avi
└── ProposedV2.avi
```

In the associated paper:

- `Proposed` corresponds to **LE-CLAHE**
- `ProposedV2` corresponds to **LE-Retinex**

### Combined video

Example:

```text
combined_scene100.avi
```

The combined recording contains an on-frame method label of the form:

```text
<MethodName> | t=<elapsed>s
```

### Snapshots

Example:

```text
snaps_scene100/
├── snap_Proposed_t3.png
├── snap_Proposed_t8.png
├── snap_Proposed_t15.png
└── ...
```

Snapshot times are currently fixed at:

```text
3, 8, and 15 seconds
```

within each method window.

## Mixed-reality streaming

The repository includes:

```text
unity/MJPEGStreamReader.cs
unity/MJPEGHandler.cs
```

These scripts provide a Unity client for receiving the MJPEG stream.

The current HoloLens implementation displays the processed NIR stream as a **virtual video frame positioned in front of the wearer**. The image is not spatially registered with the physical environment, so the current prototype is intended as head-mounted mixed-reality / immersive remote visualization rather than a complete augmented night-vision system.

The client-server architecture allows the camera and HoloLens to operate at different physical locations when both devices can communicate over the same network.

During qualitative campus-network trials, the stream was successfully viewed from widely separated locations across the same WLAN. The observed end-to-end delay was approximately **3–5 s** under these network conditions. This value is an observational estimate rather than a synchronized latency measurement.

## Unity 2020 / HoloLens 2 setup

### 1. Create a UI scene

1. Create a Unity 2020 project.
2. Add:
   - `GameObject -> UI -> Canvas`
   - `GameObject -> UI -> RawImage`
3. Resize the `RawImage` as needed.

### 2. Add the scripts

Create:

```text
Assets/Scripts/
```

and copy:

```text
MJPEGStreamReader.cs
MJPEGHandler.cs
```

into the directory.

### 3. Add the client controller

1. Create an empty GameObject.
2. Attach `MJPEGStreamReader`.
3. Assign the `RawImage` to the `outputImage` field.
4. Set `streamURL`, for example:

```text
http://192.168.1.25:8080/
```

### 4. Run

For desktop testing:

```text
Press Play in the Unity Editor.
```

For HoloLens 2:

- build using the standard UWP/HoloLens workflow;
- ensure the headset can reach the streaming host over the network.

## Networking

Both devices must be able to communicate over the same network.

Check the Linux host IP:

```bash
ip a
```

Check whether port `8080` is listening:

```bash
ss -lntp | grep 8080
```

If required, allow the port through the firewall:

```bash
sudo ufw allow 8080/tcp
```

## Reproducibility notes

The enhancement methods in the current benchmark are executed sequentially on live camera input. They therefore do not process identical prerecorded frames. Small differences in scene content and automatic camera exposure may affect method-to-method comparisons.

The exact sensor model and manufacturer specifications of the consumer camera are unavailable because the camera is an unbranded device.

Image-quality statistics such as entropy, gradient magnitude, and Laplacian variance should not be interpreted individually as direct measures of perceptual image quality, because they may also increase due to noise amplification or oversharpening.

The reported processing throughput is derived from per-frame processing time and does not represent physical camera frame rate or end-to-end streaming performance.

## License

A license has not yet been selected.

## Citation

The associated manuscript is currently under anonymous peer review. Citation information will be added after publication.
