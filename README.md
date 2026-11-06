# Person Detector

YOLO person segmentation and MobileNet re-identification, with a live dashboard, a stored identity gallery, and optional email alerts. All application dependencies run inside one Docker image. No host Python installation or webcam helper is required.

## Start on Windows or Linux

On Windows use Docker Desktop with Linux containers and the WSL 2 backend. On Linux use Docker Engine with Compose or Docker Desktop.

1. Copy `.env.example` to `.env` (`Copy-Item .env.example .env` in PowerShell, or `cp .env.example .env` on Linux). For an existing `.env`, preserve your credentials and set `SOURCE=browser` and `DEVICE=auto`.
2. Keep `yolo26n-seg.pt` in the project directory; Compose mounts it into the container. Leave the optional Linux `devices` and NVIDIA `deploy` blocks commented out initially.
3. Build and start:

   ```bash
   docker compose up --build -d
   ```

4. Open **http://localhost:5000**. In **Camera 1**, select the first webcam, click **Start camera**, and allow browser access. In **Camera 2**, select a different webcam and click **Start camera**. Both cameras capture and stream to the same container. You can also start just one camera. Device names may only appear after permission is granted for the first camera.

Each camera uploads JPEG frames at up to four frames per second. Each has its own annotated feed, controls, received/analysed frame counts, and processing rate. Stopping or disconnecting one camera does not stop the other. A device already active in one panel cannot be selected in the other.

The detector consumes the latest available frame from each camera in a batch, sharing one YOLO model and embedder. Only one pending frame per camera is retained, so slow inference does not build a backlog or let one camera monopolize the queue. Tracking is separate for each view; the identity gallery is shared, allowing appearance matching across cameras. Recent events and email alerts identify the camera. `FRAME_GAP` applies to direct camera/video/stream sources; browser uploads are already rate limited.

Two cameras increase processing load. On a slow CPU, try `IMGSZ=320`; this can reduce detection accuracy. Each panel distinguishes frames received from frames analysed and shows **Processing** during inference, even when a frame takes several seconds.

**Browser mode requires the camera tab to stay open.** Closing it or navigating away stops both captures; reopening the dashboard and starting the cameras resumes them. Each camera can also be stopped independently. Browsers may throttle or suspend background tabs. The container, dashboard, and stored gallery remain available. Each camera slot has its own browser session owner; abandoned sessions expire after 15 seconds.

Camera permission requires `localhost` or HTTPS. The dashboard is published on host loopback by default; a remote browser camera would belong to that browser's machine, and remote access requires HTTPS and an explicitly configured host port binding.

## Keep detecting after closing the browser

On native Linux Docker Engine, detection can read the webcam directly inside the container. In `.env` set:

```dotenv
SOURCE=0
VIDEO_DEVICE=/dev/video2
DEVICE=auto
```

Choose `VIDEO_DEVICE` from your host's `/dev/video*` devices. Uncomment the two lines in the Linux `devices` block in `docker-compose.yml`. The selected host device is mapped to `/dev/video0` inside the container, so keep `SOURCE=0` even when selecting host `/dev/video2`.

Apply the configuration:

```bash
docker compose up -d --force-recreate
```

This direct mode reads one camera automatically and continues with the dashboard closed. For two USB cameras on either Windows or Linux, use browser mode: set `SOURCE=browser`, comment out `devices`, and recreate the container.

On Windows Docker Desktop, browser capture stops when the camera tab closes. Unattended capture needs a network camera/stream or separately configured, compatible USB passthrough; a Linux container cannot directly open a Windows webcam API. Moving capture code into the image does not expose that hardware automatically.

A network camera works unattended on either platform: set `SOURCE=rtsp://camera-address/stream`, leave `devices` commented out, and recreate the container. For a video file, place it in `data` and use `SOURCE=/app/data/test.mp4`.

## Optional NVIDIA GPU

Uncomment the entire `deploy` block in `docker-compose.yml` and run `docker compose up -d --force-recreate`. Keep `DEVICE=auto` to use CUDA when available, or set `DEVICE=cuda:0` to require it.

Windows requires Docker Desktop's WSL 2 backend and a compatible NVIDIA Windows driver. Native Linux Engine requires the NVIDIA Container Toolkit. See [Docker GPU prerequisites](https://docs.docker.com/desktop/features/gpu/). CPU operation uses the same CUDA-capable image, with the `deploy` block commented out.

## Configuration and storage

[.env.example](.env.example) documents detection, tracking, gallery, email, and dashboard options. Email alerts require `EMAIL_SENDER`, `EMAIL_PASSWORD`, and `EMAIL_RECEIVER`; detection works without them. Use an app password where required by your mail provider.

Gallery files persist in `gallery_segmented/`; `data/` is mounted at `/app/data`. The first run needs internet access to download MobileNet weights. `DASHBOARD_PORT` changes the dashboard port, defaulting to 5000.

```bash
docker compose logs -f person-detector
docker compose down
docker compose build --no-cache
```

## Troubleshooting

- **Camera permission denied:** allow camera access in the browser and operating system, then click Start camera again. Use a localhost URL or HTTPS.
- **Camera unavailable:** close other camera applications, try another camera in the selector, and check operating system permissions. If disconnected, reconnect it and click Start camera again.
- **Another tab is sending video:** stop that camera slot in the other tab, close it, or wait 15 seconds for its session to expire. The other slot can continue independently.
- **Second camera cannot start:** select a different physical device, close other apps using it, and try separate USB ports if a shared hub cannot deliver both streams.
- **No capture after closing/minimizing the browser:** browser capture depends on the tab remaining active. Use Linux direct device mode or a network stream for unattended operation.
- **Old bridge connection error:** replace the old `http://host.docker.internal:8080/video.mjpg` source in `.env` with `SOURCE=browser`, remove any old `extra_hosts` configuration, and rebuild/recreate the container.
- **Linux device missing:** verify the selected `/dev/video*` exists and that the container can access it. The browser mode needs no device mapping.
- **CUDA unavailable:** check GPU prerequisites and the commented `deploy` block, or use `DEVICE=auto` without that block for CPU operation.
- **Model path error:** ensure `yolo26n-seg.pt` is a file in the project directory. The app requires segmentation weights (`*-seg.pt`).

Main files: [detect_and_email.py](detect_and_email.py), [docker-compose.yml](docker-compose.yml), [Dockerfile](Dockerfile), and [requirements.txt](requirements.txt).
