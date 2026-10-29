FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System deps for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/gallery_segmented /app/data /app/data/models /app/data/ultralytics

EXPOSE 5000

# The dashboard captures browser webcam frames; Compose can select other sources.
CMD ["python", "detect_and_email.py", "--model", "/app/yolo26n-seg.pt", "--dashboard", "--host", "0.0.0.0", "--dashboard-port", "5000"]
