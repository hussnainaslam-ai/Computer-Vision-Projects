# Computer Vision Mini Project
## CVZone Computer Vision Toolkit

A collection of real-time computer vision applications built with **OpenCV**, **MediaPipe**, and **CVZone**. This repo includes eye blink detection, face mesh landmark tracking, full-body pose estimation, and a virtual shirt try-on application.

## Features

| Module | Description |
|---|---|
| `EyeBlink.py` | Detects eye blinks in real time using facial landmarks and plots the eye aspect ratio live |
| `FaceMeshModule.py` | Reusable face mesh detector class (468 facial landmarks) built on MediaPipe |
| `PoseModule.py` | Reusable body pose detector class for full-body landmark tracking |
| `Shirts.py` | Virtual try-on app — overlays shirt images on a person's body using pose landmarks, with gesture-based shirt selection |

## Demo Overview

### 1. Eye Blink Counter (`EyeBlink.py`)
- Tracks 12 key landmarks around the left eye
- Computes the vertical/horizontal eye ratio to detect blinks
- Displays a live blink counter and a real-time ratio plot

### 2. Face Mesh Detector (`FaceMeshModule.py`)
- Wraps MediaPipe's Face Mesh solution into an easy-to-use `FaceMeshDetector` class
- Returns pixel coordinates for all detected facial landmarks
- Includes a demo `main()` that runs on a sample video and prints FPS

### 3. Pose Detector (`PoseModule.py`)
- Wraps MediaPipe's Pose solution into a `poseDetector` class
- Detects and draws 33 body landmarks
- Includes a demo `main()` for multi-person video pose tracking

### 4. Virtual Shirt Try-On (`Shirts.py`)
- Uses shoulder landmarks (11 & 12) to scale and position a shirt image on the body
- Includes on-screen gesture buttons — hold your hand over the left/right button to cycle through shirts in the `Shirts/` folder
- Overlay is done using `cvzone.overlayPNG` for transparent PNG shirts

## Requirements

```bash
pip install opencv-python mediapipe cvzone numpy
```

> **Note:** MediaPipe requires Python 3.8–3.11. Check compatibility if using a newer Python version.

## Project Structure

```
.
├── EyeBlink.py
├── FaceMeshModule.py
├── PoseModule.py
├── Shirts.py
├── Source/            # Input videos & button images (not included)
│   ├── 4.mp4
│   ├── 12.mp4
│   ├── shirts_1.mp4
│   └── button.png
└── Shirts/            # Shirt PNG images for try-on (not included)
```

> ⚠️ The `Source/` and `Shirts/` folders and their media files are **not included** in this repo. Add your own videos/images or update the paths to use a webcam (`cv2.VideoCapture(0)`).

## Usage

Run any script directly:

```bash
python EyeBlink.py
python FaceMeshModule.py
python PoseModule.py
python Shirts.py
```

- `EyeBlink.py` uses your webcam (`cv2.VideoCapture(0)`) by default.
- `FaceMeshModule.py` and `PoseModule.py` run on sample videos in `Source/` when executed directly, but their classes (`FaceMeshDetector`, `poseDetector`) are meant to be imported into other scripts.
- `Shirts.py` requires a video and a folder of shirt images with matching aspect ratios.

## Controls (Shirts.py)

| Action | Gesture |
|---|---|
| Next shirt | Hold right wrist near the right button |
| Previous shirt | Hold left wrist near the left button |
| Quit | Press `q` or close the window |

## Known Notes / Fixes Applied
- `FaceMeshModule.py`: `FACE_CONNECTIONS` (deprecated) replaced with `FACEMESH_TESSELATION` for compatibility with newer MediaPipe versions.
- `Shirts.py`: Added a safeguard so `widthOfShirt` never becomes zero or negative before resizing.
