# Docker Deployment Guide

This guide covers running the Person Detection & Re-identification application using Docker.

## Quick Start

### 1. Prepare Environment

```bash
# Copy example environment file and edit with your values
cp .env.example .env

# Edit .env with your email credentials and preferences
nano .env
```

### 2. CPU Mode (Default)

```bash
# Build and start
docker compose up --build -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

### 3. GPU Mode (NVIDIA GPU Required)

**Prerequisites:**
- NVIDIA GPU with driver >= 550
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed

```bash
# Build and start with GPU profile
docker compose --profile gpu up --build -d

# Or build GPU image separately first
docker compose --profile gpu build
docker compose --profile gpu up -d
```

## Configuration

All configuration is done via environment variables in `.env` file or `docker compose -e` flags.

### Key Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SOURCE` | Video source (camera index, file, RTSP URL) | `0` |
| `MODEL` | YOLO segmentation model file | `yolo26n-seg.pt` |
| `DEVICE` | Compute device (`auto`, `cpu`, `cuda:0`) | `auto` |
| `DASHBOARD` | Enable web dashboard | `true` |
| `HOST` | Dashboard bind address | `0.0.0.0` |
| `DASHBOARD_PORT` | Dashboard port | `5000` |
| `EMAIL_SENDER` | Sender email address | (required for alerts) |
| `EMAIL_PASSWORD` | Email app password | (required for alerts) |
| `EMAIL_RECEIVER` | Alert recipient email | (required for alerts) |
| `THRESHOLD` | Re-ID similarity threshold | `0.70` |
| `FRAME_GAP` | Process every Nth frame | `15` |

See `.env.example` for all available options.

## Model Files

The YOLO model files (`yolo26n-seg.pt`, `yolo26n.pt`) are **not** included in the Docker image due to size. You have two options:

### Option 1: Mount from Host (Recommended)

```yaml
# In docker-compose.yml, add to volumes:
volumes:
  - ./yolo26n-seg.pt:/app/yolo26n-seg.pt:ro
  - ./yolo26n.pt:/app/yolo26n.pt:ro
```

Then in `.env`:
```env
MODEL=/app/yolo26n-seg.pt
```

### Option 2: Auto-download at Runtime

The ultralytics library will auto-download models on first run. Ensure the container has internet access and set:
```env
MODEL=yolo26n-seg.pt  # Will download to /root/.cache/ultralytics
```

## Data Persistence

The following directories are persisted via Docker volumes:

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./gallery_segmented` | `/app/gallery_segmented` | Person identities & embeddings |
| `./data/models` | `/root/.cache/torch` | PyTorch model cache |
| `./data/ultralytics` | `/root/.cache/ultralytics` | Ultralytics model cache |

## Accessing the Dashboard

Once running with `DASHBOARD=true`:

- **Local:** http://localhost:5000
- **Network:** http://<host-ip>:5000

## Using a Video File Instead of Webcam

```bash
# Mount your video directory
# In docker-compose.yml:
volumes:
  - ./videos:/app/videos:ro

# In .env:
SOURCE=/app/videos/your_video.mp4
```

## Using an RTSP Stream

```env
SOURCE=rtsp://user:pass@192.168.1.100:554/stream
```

## GPU Memory Management

If you encounter OOM errors:

```env
# Reduce batch processing
FRAME_GAP=30
IMGSZ=480
MAX_SAMPLES=15
```

Or limit GPU memory in docker-compose:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
          device_ids: ['0']
```

## Troubleshooting

### Container exits immediately
```bash
docker compose logs person-detector
# Check for missing model files, invalid config, or import errors
```

### No video feed / "Waiting for frames"
- Verify `SOURCE` is correct (camera index, file path, or stream URL)
- For webcam: ensure `/dev/video0` is accessible (add `devices: - /dev/video0:/dev/video0` to compose)
- For files: ensure file is mounted and path is correct inside container

### Email not sending
- Verify `EMAIL_PASSWORD` is a 16-char App Password (not regular password for Gmail)
- Check SMTP settings match your provider
- Enable `VERBOSE=true` for debug output

### Dashboard not accessible
- Ensure `HOST=0.0.0.0` (not `127.0.0.1`)
- Check firewall allows port 5000
- Verify `DASHBOARD=true` in .env

## Building Custom Images

### CPU-only production image (smaller)
```bash
docker build -t person-detector:latest .
```

### GPU image
```bash
docker build -f Dockerfile.gpu -t person-detector:gpu .
```

### Multi-arch build
```bash
docker buildx build --platform linux/amd64,linux/arm64 -t person-detector:latest .
```

## Health Checks

Add to docker-compose.yml for production:
```yaml
services:
  person-detector:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/api/state"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

## Updating

```bash
# Pull latest code changes
git pull

# Rebuild and restart
docker compose up --build -d

# Or for GPU
docker compose --profile gpu up --build -d
```