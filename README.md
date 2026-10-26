# Person Detector

A GPU-accelerated person detection and re-identification app built with YOLO segmentation, MobileNet embeddings, a live dashboard, and optional email alerts for new detections.

This project runs as a Dockerized application and is intended to work with one webcam at a time via environment configuration.

## Features

- YOLO segmentation for person detection
- MobileNet-based embeddings for matching known vs new people
- Gallery of stored identities and sample crops
- Live dashboard with recent detections and identity list
- Optional email alerts for newly detected people
- CUDA GPU support via Docker

## Project files

- [detect_and_email.py](detect_and_email.py) — main application logic
- [docker-compose.yml](docker-compose.yml) — container runtime configuration
- [Dockerfile](Dockerfile) — Docker image definition
- [.env](.env) — local runtime configuration (not committed for secrets)
- [.env.example](.env.example) — example environment file
- [requirements.txt](requirements.txt) — Python dependencies

## Prerequisites

- Docker Engine installed
- Docker Compose available
- NVIDIA GPU and NVIDIA Container Toolkit for GPU acceleration
- A webcam device available as `/dev/video0`, `/dev/video1`, etc., or another supported source like a file or RTSP stream

## Quick start

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Edit [.env](.env) to choose the camera source:

```env
SOURCE=0
VIDEO_DEVICE=/dev/video0
```

For a second camera:

```env
SOURCE=2
VIDEO_DEVICE=/dev/video2
```

3. Start the app:

```bash
docker compose up --build -d
```

4. Watch logs:

```bash
docker compose logs -f person-detector
```

5. Open the dashboard:

```text
http://localhost:5000
```

## Environment configuration

The app is designed to be controlled mainly through [.env](.env).

### Required settings

- `SOURCE`: source to read from, such as `0`, `2`, a video file, or an RTSP URL
- `VIDEO_DEVICE`: host device path for the selected webcam, such as `/dev/video0` or `/dev/video2`
- `EMAIL_SENDER`, `EMAIL_PASSWORD`, `EMAIL_RECEIVER`: for email alerts

### Example

```env
SOURCE=0
VIDEO_DEVICE=/dev/video0
MODEL=yolo26n-seg.pt
DEVICE=auto
DASHBOARD=true
DASHBOARD_PORT=5000
```

## Webcam selection

To use webcam 1:

```env
SOURCE=0
VIDEO_DEVICE=/dev/video0
```

To use webcam 2:

```env
SOURCE=2
VIDEO_DEVICE=/dev/video2
```

If the selected camera is not connected, update the values accordingly or use a file/RTSP path instead.

## Docker commands

Build and start:

```bash
docker compose up --build -d
```

Stop:

```bash
docker compose down
```

View logs:

```bash
docker compose logs -f person-detector
```

Rebuild images:

```bash
docker compose build --no-cache
```

## Troubleshooting

### Camera not found

If Docker says the camera device is missing, check the host:

```bash
ls -l /dev/video*
```

Then set the matching value in [.env](.env):

```env
VIDEO_DEVICE=/dev/video0
```

or

```env
VIDEO_DEVICE=/dev/video2
```

### Docker permission errors

Ensure your user can access the Docker socket:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

### GPU not available inside the container

Verify the NVIDIA Container Toolkit is installed and the host GPU is visible:

```bash
nvidia-smi
```

## Notes

- The app stores gallery and sample data under `gallery_segmented/`
- Model weights are mounted from the project directory and can be replaced with your preferred YOLO segmentation model
- The dashboard is served on port `5000` by default

## Recommended long-term structure

Use one canonical documentation file, [README.md](README.md), for the full project setup and keep any Docker-specific note as a short compatibility pointer if needed.
