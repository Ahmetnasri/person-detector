"""Optimized person segmentation + re-identification + email alerts, with a live dashboard.

Rewrite of detect_and_email.py. Same idea (YOLO finds people, a MobileNetV3 embedding
decides whether each person was seen before, new people trigger an email), but:

  * YOLO only runs on every Nth frame (vid_stride) and only for the "person" class.
  * Person masks remove the background before crops are embedded in one batch per frame.
  * The identity gallery lives in memory as a single matrix; matching is one mat-mul.
  * Disk writes and SMTP run on background threads, so the detection loop never blocks.
  * The dashboard is a single page fed by a JSON API + an MJPEG live feed (no page reloads).

Usage:
    python detect_and_email.py --source browser --dashboard
"""

import argparse
import json
import logging
import os
import queue
import re
import secrets
import smtplib
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, Response, abort, jsonify, request, send_from_directory
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from ultralytics import YOLO


def load_env_file(env_path):
    """Load KEY=VALUE pairs from a .env file without overriding existing env vars."""
    env_file = Path(env_path)
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file(Path(__file__).resolve().with_name(".env"))


# --------------------------------------------------------------------------- helpers

NEW_BGR = (52, 104, 235)     # orange: first sighting
KNOWN_BGR = (229, 135, 57)   # blue: re-identified


def encode_jpeg(img, quality=85):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def resize_max_width(img, max_width):
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / w
    return cv2.resize(img, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def draw_box(img, box, text, color, thickness=2):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    ty = max(y1, th + 8)
    cv2.rectangle(img, (x1, ty - th - 8), (x1 + tw + 8, ty), color, -1)
    cv2.putText(img, text, (x1 + 4, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- embedding

class Embedder:
    """Batched MobileNetV3-Small feature extractor working on masked BGR person crops."""

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, device, input_hw=(256, 128)):
        weights = MobileNet_V3_Small_Weights.DEFAULT
        self.device = torch.device(device)
        self.net = mobilenet_v3_small(weights=weights).features.eval().to(self.device)
        self.half = self.device.type == "cuda"
        if self.half:
            self.net.half()
        # Person-shaped input (tall) keeps the whole body instead of center-cropping it away.
        self.h, self.w = input_hw
        self.mean = torch.tensor(self.MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.STD, device=self.device).view(1, 3, 1, 1)
        self.dim = self(np.zeros((1, self.h, self.w, 3), np.uint8)).shape[1]  # warm-up + dim

    @torch.inference_mode()
    def __call__(self, crops):
        if len(crops) == 0:
            return np.empty((0, getattr(self, "dim", 0)), np.float32)
        batch = np.stack([cv2.resize(c, (self.w, self.h), interpolation=cv2.INTER_LINEAR) for c in crops])
        batch = np.ascontiguousarray(batch[..., ::-1])  # BGR -> RGB
        x = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).float().div_(255.0)
        x = (x - self.mean) / self.std
        if self.half:
            x = x.half()
        feats = self.net(x).mean(dim=(2, 3)).float()
        return torch.nn.functional.normalize(feats, dim=1).cpu().numpy()


# --------------------------------------------------------------------------- gallery

@dataclass
class Sample:
    file: str
    emb: np.ndarray
    ts: float


@dataclass
class Identity:
    name: str
    created: float
    last_seen: float
    sightings: int
    samples: deque
    last_sample_ts: float = 0.0
    last_score: float | None = None

    def meta(self):
        return {"created": self.created, "last_seen": self.last_seen, "sightings": self.sightings}


def _write_sample(folder, stem, crop, emb, frame, meta):
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    np.save(folder / f"{stem}.npy", emb)
    if frame is not None:
        tmp = folder / "latest_frame.tmp.jpg"
        cv2.imwrite(str(tmp), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        os.replace(tmp, folder / "latest_frame.jpg")
    _write_meta(folder, meta)


def _write_meta(folder, meta):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


class Gallery:
    """In-memory identity store backed by <root>/<person>/crop_*.jpg|.npy files."""

    def __init__(self, root, dim, max_samples, io):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.max_samples = max_samples
        self.io = io
        self.lock = threading.RLock()
        self.people: dict[str, Identity] = {}
        self._matrix = None
        self._owners = []
        self._next_id = 1
        self._load()

    def _load(self):
        for folder in sorted(p for p in self.root.iterdir() if p.is_dir()):
            files = sorted(folder.glob("crop_*.npy"), key=lambda p: p.stat().st_mtime)[-self.max_samples:]
            samples = deque(maxlen=self.max_samples)
            for path in files:
                try:
                    emb = np.load(path).astype(np.float32).ravel()
                except Exception:
                    continue
                if emb.shape[0] != self.dim:
                    continue
                emb /= np.linalg.norm(emb) or 1.0
                samples.append(Sample(path.with_suffix(".jpg").name, emb, path.stat().st_mtime))
            if not samples:
                continue
            try:
                meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
            except Exception:
                meta = {}
            self.people[folder.name] = Identity(
                name=folder.name,
                created=meta.get("created", samples[0].ts),
                last_seen=meta.get("last_seen", samples[-1].ts),
                sightings=meta.get("sightings", len(samples)),
                samples=samples,
                last_sample_ts=samples[-1].ts,
            )
            m = re.fullmatch(r"person_(\d+)", folder.name)
            if m:
                self._next_id = max(self._next_id, int(m.group(1)) + 1)

    def _build_matrix(self):
        embs, owners = [], []
        for ident in self.people.values():
            for s in ident.samples:
                embs.append(s.emb)
                owners.append(ident.name)
        self._matrix = np.stack(embs) if embs else None
        self._owners = owners

    def match(self, emb, exclude=()):
        """Return (best identity name, cosine similarity) or (None, 0.0) if the gallery is empty."""
        with self.lock:
            if self._matrix is None:
                self._build_matrix()
            if self._matrix is None:
                return None, 0.0
            sims = self._matrix @ emb
            # An identity can belong to only one person in the current frame.
            if exclude:
                sims = np.where([name not in exclude for name in self._owners], sims, -np.inf)
            i = int(np.argmax(sims))
            if not np.isfinite(sims[i]):
                return None, 0.0
            return self._owners[i], float(sims[i])

    def create(self, now):
        with self.lock:
            name = f"person_{self._next_id:04d}"
            self._next_id += 1
            ident = Identity(name, now, now, 0, deque(maxlen=self.max_samples))
            self.people[name] = ident
            return ident

    def observe(self, ident, now, score):
        with self.lock:
            ident.last_seen = now
            ident.sightings += 1
            ident.last_score = score

    def add_sample(self, ident, emb, crop, frame, now):
        stem = f"crop_{int(now * 1000)}"
        with self.lock:
            ident.samples.append(Sample(f"{stem}.jpg", emb, now))
            ident.last_sample_ts = now
            self._matrix = None
            meta = ident.meta()
        self.io.submit(_write_sample, self.root / ident.name, stem, crop, emb, frame, meta)

    def flush_meta(self):
        with self.lock:
            metas = [(ident.name, ident.meta()) for ident in self.people.values()]
        for name, meta in metas:
            self.io.submit(_write_meta, self.root / name, meta)

    def summary(self):
        with self.lock:
            out = []
            for ident in self.people.values():
                if not ident.samples:
                    continue
                out.append({
                    "name": ident.name,
                    "created": ident.created,
                    "last_seen": ident.last_seen,
                    "sightings": ident.sightings,
                    "samples": len(ident.samples),
                    "last_score": ident.last_score,
                    "thumb": f"/gallery/{ident.name}/{ident.samples[-1].file}",
                })
        out.sort(key=lambda d: d["last_seen"], reverse=True)
        return out

    def detail(self, name):
        with self.lock:
            ident = self.people.get(name)
            if ident is None or not ident.samples:
                return None
            samples = list(ident.samples)
            info = {"name": ident.name, **ident.meta(), "last_score": ident.last_score}
        embs = np.stack([s.emb for s in samples])
        sims = embs @ embs[-1]
        info["frame"] = f"/gallery/{name}/latest_frame.jpg"
        info["samples"] = [
            {"url": f"/gallery/{name}/{s.file}", "ts": s.ts, "similarity": round(float(sim), 4)}
            for s, sim in zip(reversed(samples), reversed(sims))
        ]
        return info


# --------------------------------------------------------------------------- email

class Emailer(threading.Thread):
    """Sends alert emails from a background queue so SMTP latency never stalls detection."""

    def __init__(self, sender, password, receiver, server, port, use_ssl, on_result):
        super().__init__(daemon=True, name="emailer")
        self.sender, self.password, self.receiver = sender, password, receiver
        self.server, self.port, self.use_ssl = server, port, use_ssl
        self.on_result = on_result
        self.jobs = queue.Queue(maxsize=20)

    def submit(self, **job):
        try:
            self.jobs.put_nowait(job)
        except queue.Full:
            print("Email queue full; dropping alert.")

    def stop(self):
        self.jobs.put(None)

    def run(self):
        while (job := self.jobs.get()) is not None:
            self.on_result(self._send(**job))

    def _send(self, identity, source, ts, crop_jpeg, frame_jpeg):
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        msg = EmailMessage()
        msg["Subject"] = f"New person detected: {identity}"
        msg["From"] = self.sender
        msg["To"] = self.receiver
        msg.set_content(
            f"A new person ({identity}) was detected on source '{source}' at {when}.\n"
            "The person crop and the full annotated frame are attached."
        )
        msg.add_attachment(crop_jpeg, maintype="image", subtype="jpeg", filename=f"{identity}_crop.jpg")
        if frame_jpeg:
            msg.add_attachment(frame_jpeg, maintype="image", subtype="jpeg", filename=f"{identity}_frame.jpg")
        try:
            if self.use_ssl:
                with smtplib.SMTP_SSL(self.server, self.port, timeout=20) as smtp:
                    smtp.login(self.sender, self.password)
                    smtp.send_message(msg)
            else:
                with smtplib.SMTP(self.server, self.port, timeout=20) as smtp:
                    smtp.starttls()
                    smtp.login(self.sender, self.password)
                    smtp.send_message(msg)
            print(f"Email sent to {self.receiver} for {identity}")
            return True
        except smtplib.SMTPAuthenticationError as e:
            print(f"SMTP authentication failed (for Gmail use a 16-character App Password): {e}")
        except Exception as e:
            print(f"Email error: {e}")
        return False


# --------------------------------------------------------------------------- live state

class BrowserCamera:
    """One browser camera owner, with a single pending frame to avoid latency buildup."""

    LEASE_SECONDS = 15

    def __init__(self):
        self.condition = threading.Condition()
        self.token = None
        self.last_upload = 0
        self.pending = None

    def _active(self):
        return self.token is not None and time.monotonic() - self.last_upload < self.LEASE_SECONDS

    def start(self):
        with self.condition:
            if self._active():
                return None
            self.token = secrets.token_urlsafe(24)
            self.last_upload = time.monotonic()
            self.pending = None
            return self.token

    def submit(self, token, frame):
        with self.condition:
            if not self._active() or token != self.token:
                return False
            self.pending = frame
            self.last_upload = time.monotonic()
            self.condition.notify_all()
            return True

    def stop(self, token):
        with self.condition:
            if not token or token != self.token:
                return False
            self.token = None
            self.pending = None
            self.condition.notify_all()
            return True

    def read(self):
        with self.condition:
            self.condition.wait_for(lambda: self.pending is not None, timeout=1)
            frame, self.pending = self.pending, None
            return frame if self._active() else None


class LiveState:
    """Everything the dashboard shows that isn't in the gallery."""

    TIMELINE_MINUTES = 30

    def __init__(self, source, recent_limit, email_enabled, device="auto"):
        self.lock = threading.Lock()
        self.frame_cond = threading.Condition()
        self.frame_jpeg = None
        self.frame_seq = 0
        self.source = source
        self.cuda_available = torch.cuda.is_available()
        self.device = ("cuda:0" if self.cuda_available else "cpu") if device == "auto" else device
        self.email_enabled = email_enabled
        self.started = time.time()
        self.running = True
        self.frames = 0
        self.detections = 0
        self.new_people = 0
        self.fps = 0.0
        self.last_frame_ts = None
        self.emails_sent = 0
        self.emails_failed = 0
        self.last_email_ts = None
        self.recent = deque(maxlen=recent_limit)
        self._event_id = 0
        self.minutes = {}  # minute epoch -> detection count

    def tick_frame(self, now):
        with self.lock:
            if self.last_frame_ts is not None:
                dt = now - self.last_frame_ts
                if dt > 0:
                    self.fps = (1.0 / dt) if self.fps == 0 else 0.9 * self.fps + 0.1 / dt
            self.last_frame_ts = now
            self.frames += 1

    def push_frame(self, jpeg):
        with self.frame_cond:
            self.frame_jpeg = jpeg
            self.frame_seq += 1
            self.frame_cond.notify_all()

    def add_event(self, identity, score, is_new, crop_jpeg, now):
        with self.lock:
            self._event_id += 1
            self.recent.appendleft({
                "id": self._event_id, "identity": identity, "score": score,
                "is_new": is_new, "ts": now, "jpeg": crop_jpeg,
            })
            self.detections += 1
            self.new_people += int(is_new)
            minute = int(now // 60) * 60
            self.minutes[minute] = self.minutes.get(minute, 0) + 1
            cutoff = minute - 60 * self.TIMELINE_MINUTES
            for k in [k for k in self.minutes if k <= cutoff]:
                del self.minutes[k]

    def email_result(self, ok):
        with self.lock:
            if ok:
                self.emails_sent += 1
                self.last_email_ts = time.time()
            else:
                self.emails_failed += 1

    def recent_jpeg(self, event_id):
        with self.lock:
            for e in self.recent:
                if e["id"] == event_id:
                    return e["jpeg"]
        return None

    def snapshot(self):
        now = time.time()
        with self.lock:
            current = int(now // 60) * 60
            timeline = [
                {"t": m, "count": self.minutes.get(m, 0)}
                for m in range(current - 60 * (self.TIMELINE_MINUTES - 1), current + 1, 60)
            ]
            return {
                "now": now,
                "source": self.source,
                "compute": {
                    "device": self.device,
                    "cuda_available": self.cuda_available,
                    "using_gpu": self.cuda_available and self.device.startswith("cuda"),
                },
                "running": self.running and (self.source != "browser" or (
                    self.last_frame_ts is not None and now - self.last_frame_ts < 5)),
                "started": self.started,
                "frames": self.frames,
                "detections": self.detections,
                "new_people": self.new_people,
                "fps": round(self.fps, 2),
                "last_frame_ts": self.last_frame_ts,
                "email": {
                    "enabled": self.email_enabled, "sent": self.emails_sent,
                    "failed": self.emails_failed, "last": self.last_email_ts,
                },
                "recent": [
                    {k: v for k, v in e.items() if k != "jpeg"} | {"img": f"/api/recent/{e['id']}.jpg"}
                    for e in self.recent
                ],
                "timeline": timeline,
            }


# --------------------------------------------------------------------------- pipeline

class Pipeline:
    def __init__(self, args, embedder, gallery, state, emailer):
        self.args = args
        self.embedder = embedder
        self.gallery = gallery
        self.state = state
        self.emailer = emailer
        self.last_email = float("-inf")
        self.tracks = {}
        self.frame_index = 0

    def _associate_tracks(self, boxes, embs, now):
        """Reuse recent identities when both position and appearance agree."""
        self.tracks = {
            name: track for name, track in self.tracks.items()
            if now - track["ts"] <= self.args.track_timeout
            and self.frame_index - track["frame"] <= 3
        }
        candidates = []
        for i, (box, emb) in enumerate(zip(boxes, embs)):
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            for name, track in self.tracks.items():
                tx1, ty1, tx2, ty2 = track["box"]
                intersection = max(0, min(x2, tx2) - max(x1, tx1)) * max(0, min(y2, ty2) - max(y1, ty1))
                union = area + (tx2 - tx1) * (ty2 - ty1) - intersection
                iou = intersection / union if union else 0.0
                score = float(emb @ track["emb"])
                if iou >= self.args.track_iou and score >= self.args.track_threshold:
                    candidates.append((iou + score, i, name, score))

        matches, used = {}, set()
        for _, i, name, score in sorted(candidates, reverse=True):
            if i not in matches and name not in used:
                matches[i] = (name, score)
                used.add(name)
        return matches

    def process(self, result):
        now = time.time()
        self.frame_index += 1
        self.state.tick_frame(now)
        frame = result.orig_img  # BGR
        h, w = frame.shape[:2]

        boxes, crops = [], []
        if result.boxes is not None and len(result.boxes):
            if result.masks is None:
                raise ValueError("Person detections have no masks; use a YOLO segmentation model (--model ...-seg.pt).")
            masks = result.masks.data.cpu().numpy()
            if len(masks) != len(result.boxes) or masks.shape[1:] != (h, w):
                raise ValueError("Expected one full-resolution mask per detection; enable retina_masks=True.")
            for coords, mask in zip(result.boxes.xyxy.cpu().numpy().astype(int), masks):
                x1, y1, x2, y2 = coords
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                if (y2 - y1) >= self.args.min_box and (x2 - x1) >= 8:
                    person_mask = mask[y1:y2, x1:x2] > 0.5
                    if not person_mask.any():
                        continue
                    crop = frame[y1:y2, x1:x2].copy()
                    crop[~person_mask] = 0
                    boxes.append((x1, y1, x2, y2))
                    crops.append(crop)

        embs = self.embedder(crops)
        tracked = self._associate_tracks(boxes, embs, now)
        # Reserve track matches first so another detection cannot take their identity.
        used = {name for name, _ in tracked.values()}
        labels = []
        for i, (box, crop, emb) in enumerate(zip(boxes, crops, embs)):
            label = self._identify(box, crop, emb, frame, now, tracked.get(i), used)
            used.add(label[0])
            labels.append(label)

        if self.args.dashboard:
            vis = frame.copy()
            for box, (name, score, is_new) in zip(boxes, labels):
                text = f"{name} NEW" if is_new else f"{name} {score:.2f}"
                draw_box(vis, box, text, NEW_BGR if is_new else KNOWN_BGR)
            self.state.push_frame(encode_jpeg(resize_max_width(vis, 960), 75))

    def _identify(self, box, crop, emb, frame, now, tracked=None, exclude=()):
        name, score = tracked if tracked is not None else self.gallery.match(emb, exclude)
        is_new = name is None or (tracked is None and score < self.args.threshold)
        ident = self.gallery.create(now) if is_new else self.gallery.people[name]
        self.gallery.observe(ident, now, None if is_new else score)

        # Smooth confident observations. Weak continuity matches keep the stable
        # reference so marginal matches cannot gradually replace its appearance.
        previous = self.tracks.get(ident.name)
        reference = emb.copy() if previous is None else previous["emb"]
        if previous is not None and score >= self.args.threshold:
            reference = 0.8 * reference + 0.2 * emb
            reference /= np.linalg.norm(reference) or 1.0
        self.tracks[ident.name] = {"box": box, "emb": reference, "ts": now, "frame": self.frame_index}

        crop = crop.copy()
        framed = None
        if is_new or (score >= self.args.threshold and now - ident.last_sample_ts >= self.args.sample_interval):
            framed = frame.copy()
            draw_box(framed, box, ident.name, NEW_BGR if is_new else KNOWN_BGR, 3)
            self.gallery.add_sample(ident, emb, crop, framed, now)

        crop_jpeg = encode_jpeg(crop)
        self.state.add_event(ident.name, None if is_new else round(score, 4), is_new, crop_jpeg, now)

        if is_new:
            closest = f" (closest {name} {score:.3f})" if name else ""
            print(f"New person: {ident.name}{closest}")
            self._maybe_email(ident.name, crop_jpeg, framed, now)
        elif self.args.verbose:
            print(f"Matched {ident.name} ({score:.3f})")
        return ident.name, score, is_new

    def _maybe_email(self, identity, crop_jpeg, framed, now):
        if self.emailer is None:
            return
        if now - self.last_email < self.args.cooldown:
            print(f"  email cooldown active, skipping alert for {identity}")
            return
        self.last_email = now
        self.emailer.submit(
            identity=identity, source=str(self.args.source), ts=now,
            crop_jpeg=crop_jpeg, frame_jpeg=encode_jpeg(resize_max_width(framed, 1280)),
        )


# --------------------------------------------------------------------------- dashboard

DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Person Monitor</title>
<style>
  :root {
    color-scheme: light;
    --bg: #f4f4f2; --surface: #fcfcfb; --surface-2: #f0efec; --border: #e2e1dc;
    --text: #0b0b0b; --text-2: #52514e; --muted: #77766f;
    --accent: #2a78d6; --new: #eb6834; --track: #cde2fb;
    --good: #0ca30c; --warning: #fab219; --critical: #d03b3b;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      color-scheme: dark;
      --bg: #121211; --surface: #1a1a19; --surface-2: #232321; --border: #2f2f2c;
      --text: #f0efec; --text-2: #c3c2b7; --muted: #8d8c85;
      --accent: #3987e5; --new: #d95926; --track: #184f95;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  header { position: sticky; top: 0; z-index: 5; display: flex; align-items: center; gap: 16px;
           padding: 14px 28px; background: var(--bg); border-bottom: 1px solid var(--border); }
  header h1 { font-size: 18px; margin: 0; font-weight: 600; }
  header .src { color: var(--text-2); font-size: 13px; }
  header .spacer { flex: 1; }
  .status { display: inline-flex; align-items: center; gap: 8px; font-size: 13px; color: var(--text-2);
            padding: 4px 12px; border: 1px solid var(--border); border-radius: 999px; background: var(--surface); }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
  .dot.live { background: var(--good); box-shadow: 0 0 0 0 var(--good); animation: pulse 2s infinite; }
  .dot.down { background: var(--critical); }
  @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(12,163,12,.5); } 70% { box-shadow: 0 0 0 8px rgba(12,163,12,0); } 100% { box-shadow: 0 0 0 0 rgba(12,163,12,0); } }
  main { max-width: 1500px; margin: 0 auto; padding: 22px 28px 40px; display: grid; gap: 20px; }
  .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 16px; }
  .tile, .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; }
  .tile { padding: 16px 18px; }
  .tile .label { color: var(--text-2); font-size: 13px; }
  .tile .value { font-size: 30px; font-weight: 600; margin-top: 2px; }
  .tile .sub { color: var(--muted); font-size: 12px; margin-top: 2px; }
  .row { display: grid; grid-template-columns: minmax(0, 2fr) minmax(300px, 1fr); gap: 20px; }
  @media (max-width: 1000px) { .row { grid-template-columns: 1fr; } }
  .panel { padding: 16px 18px; min-width: 0; }
  .panel-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 12px; }
  .panel-head h2 { font-size: 15px; font-weight: 600; margin: 0; }
  .panel-head .hint { color: var(--muted); font-size: 12px; }
  .feed { position: relative; aspect-ratio: 16 / 9; background: #000; border-radius: 8px; overflow: hidden;
          display: grid; place-items: center; }
  .feed img { position: relative; width: 100%; height: 100%; object-fit: contain; }
  .feed .ph { position: absolute; color: #8d8c85; font-size: 13px; }
  .legend { display: flex; gap: 14px; font-size: 12px; color: var(--text-2); }
  .legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 2px;
                         margin-right: 6px; vertical-align: -1px; background: var(--c); }
  .activity { display: flex; flex-direction: column; gap: 8px; max-height: 560px; overflow-y: auto; }
  .event { display: flex; gap: 12px; align-items: center; padding: 8px; border-radius: 8px;
           background: var(--surface-2); cursor: pointer; border: 1px solid transparent; }
  .event:hover { border-color: var(--border); }
  .event img { width: 42px; height: 60px; object-fit: cover; border-radius: 6px; background: #000; flex: none; }
  .event .who { font-weight: 600; }
  .event .meta { color: var(--muted); font-size: 12px; }
  .event .right { margin-left: auto; text-align: right; }
  .tag { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text); font-weight: 500; }
  .tag::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--c); }
  .empty { color: var(--muted); font-size: 13px; padding: 24px 0; text-align: center; }
  .chart { position: relative; }
  .chart svg { display: block; width: 100%; overflow: visible; }
  .chart .grid line { stroke: var(--border); }
  .chart text { fill: var(--muted); font-size: 11px; }
  .chart .bar { fill: var(--accent); }
  .chart .hit { fill: transparent; cursor: default; }
  .chart .hit:hover + .bar, .chart .bar.hl { fill-opacity: .75; }
  .tooltip { position: absolute; pointer-events: none; background: var(--surface); color: var(--text);
             border: 1px solid var(--border); border-radius: 8px; padding: 6px 10px; font-size: 12px;
             box-shadow: 0 4px 16px rgba(0,0,0,.18); white-space: nowrap; transform: translate(-50%, -110%); display: none; }
  .tooltip b { font-weight: 600; }
  .controls { display: flex; gap: 8px; }
  input[type=search], select { background: var(--surface-2); color: var(--text); border: 1px solid var(--border);
            border-radius: 8px; padding: 6px 10px; font: inherit; font-size: 13px; }
  input[type=search] { width: 200px; }
  .ids { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 14px; }
  .card { background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px; overflow: hidden;
          cursor: pointer; transition: transform .12s ease, border-color .12s ease; }
  .card:hover { transform: translateY(-2px); border-color: var(--accent); }
  .card img { display: block; width: 100%; aspect-ratio: 3 / 4; object-fit: cover; background: #000; }
  .card .body { padding: 8px 10px 10px; }
  .card .name { font-weight: 600; font-size: 13px; }
  .card .meta { color: var(--muted); font-size: 12px; }
  .modal { position: fixed; inset: 0; z-index: 20; background: rgba(0,0,0,.6); display: none;
           place-items: center; padding: 20px; }
  .modal.open { display: grid; }
  .dialog { background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
            width: min(1080px, 100%); max-height: 90vh; overflow: auto; padding: 20px 22px; }
  .dialog-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }
  .dialog-head h3 { margin: 0; font-size: 18px; }
  .close { background: none; border: 1px solid var(--border); color: var(--text); border-radius: 8px;
           padding: 4px 10px; cursor: pointer; font: inherit; }
  .detail { display: grid; grid-template-columns: minmax(0, 3fr) minmax(220px, 1fr); gap: 18px; }
  @media (max-width: 800px) { .detail { grid-template-columns: 1fr; } }
  .detail .frame { width: 100%; border-radius: 8px; background: #000; display: block; }
  .facts { display: grid; gap: 10px; align-content: start; }
  .fact .k { color: var(--text-2); font-size: 12px; }
  .fact .v { font-size: 16px; font-weight: 600; }
  .samples { display: grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); gap: 12px; margin-top: 18px; }
  .sample img { width: 100%; aspect-ratio: 3 / 4; object-fit: cover; border-radius: 6px; background: #000; display: block; }
  .meter { height: 6px; border-radius: 3px; background: var(--track); margin-top: 6px; overflow: hidden; }
  .meter > div { height: 100%; background: var(--accent); border-radius: 3px; }
  .sample .meta { color: var(--text-2); font-size: 12px; margin-top: 4px; display: flex; justify-content: space-between; }
</style>
</head>
<body>
<header>
  <h1>Person Monitor</h1>
  <span class="src" id="source"></span>
  <span class="spacer"></span>
  <span class="status"><span class="dot" id="dot"></span><span id="status">Connecting…</span></span>
</header>

<main>
  <section class="panel" id="camera-controls" hidden>
    <div class="panel-head">
      <h2>Camera</h2>
      <div class="controls">
        <select id="camera-device" aria-label="Camera"><option value="">Default camera</option></select>
        <button class="close" id="camera-start">Start camera</button>
        <button class="close" id="camera-stop" disabled>Stop camera</button>
      </div>
    </div>
    <p id="camera-message" role="status">Start the camera and allow access. Keep this tab open while detecting.</p>
    <video id="camera-preview" autoplay muted playsinline style="width:160px;border-radius:8px" hidden></video>
    <canvas id="camera-canvas" hidden></canvas>
  </section>
  <section class="kpis" id="kpis"></section>

  <section class="row">
    <div class="panel">
      <div class="panel-head">
        <h2>Live feed</h2>
        <div class="legend"><span style="--c:var(--new)">New person</span><span style="--c:var(--accent)">Known person</span></div>
      </div>
      <div class="feed"><span class="ph">Waiting for frames…</span><img src="/stream.mjpg" alt="Live annotated camera feed"></div>
    </div>
    <div class="panel">
      <div class="panel-head"><h2>Recent detections</h2><span class="hint" id="recent-count"></span></div>
      <div class="activity" id="activity"></div>
    </div>
  </section>

  <section class="panel">
    <div class="panel-head"><h2>Detections per minute</h2><span class="hint">Last 30 minutes</span></div>
    <div class="chart" id="chart"><div class="tooltip" id="tip"></div></div>
  </section>

  <section class="panel">
    <div class="panel-head">
      <h2>Identities <span class="hint" id="id-count"></span></h2>
      <div class="controls">
        <input type="search" id="q" placeholder="Search identities" aria-label="Search identities">
        <select id="sort" aria-label="Sort identities">
          <option value="last_seen">Last seen</option>
          <option value="sightings">Most sightings</option>
          <option value="created">Newest first</option>
          <option value="name">Name</option>
        </select>
      </div>
    </div>
    <div class="ids" id="ids"></div>
  </section>
</main>

<div class="modal" id="modal" role="dialog" aria-modal="true">
  <div class="dialog" id="dialog"></div>
</div>

<script>
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const compact = n => n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e4 ? (n / 1e3).toFixed(1) + "K" : n.toLocaleString();
const clock = ts => new Date(ts * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});
const stamp = ts => new Date(ts * 1000).toLocaleString();
function ago(ts) {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 5) return "just now";
  if (s < 60) return Math.floor(s) + "s ago";
  if (s < 3600) return Math.floor(s / 60) + "m ago";
  if (s < 86400) return Math.floor(s / 3600) + "h ago";
  return Math.floor(s / 86400) + "d ago";
}
function duration(s) {
  const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60);
  return h ? `${h}h ${m}m` : m ? `${m}m ${Math.floor(s % 60)}s` : `${Math.floor(s)}s`;
}
function setHTML(el, html) { if (el._html !== html) { el.innerHTML = html; el._html = html; } }

let state = null, identities = [];
let cameraMedia = null, cameraToken = null, cameraStarting = false, uploadAbort = null;
let cameraRelease = Promise.resolve();

async function cameraDevices(selected = "") {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === "videoinput");
  const select = $("#camera-device");
  select.innerHTML = '<option value="">Default camera</option>';
  devices.forEach((device, i) => select.add(new Option(device.label || `Camera ${i + 1}`, device.deviceId)));
  select.value = selected;
}

function stopCamera(message = "Camera stopped. Click Start camera to resume.") {
  const token = cameraToken;
  cameraToken = null;
  uploadAbort?.abort();
  cameraMedia?.getTracks().forEach(track => track.stop());
  cameraMedia = null;
  $("#camera-preview").srcObject = null;
  $("#camera-preview").hidden = true;
  $("#camera-start").disabled = cameraStarting;
  $("#camera-stop").disabled = true;
  $("#camera-device").disabled = false;
  $("#camera-message").textContent = message;
  if (token) cameraRelease = fetch("/api/camera/stop", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({token}), keepalive: true,
  }).catch(() => {});
}

async function uploadCamera(token) {
  const video = $("#camera-preview"), canvas = $("#camera-canvas");
  const context = canvas.getContext("2d");
  while (cameraToken === token && cameraMedia) {
    try {
      if (!video.videoWidth || video.readyState < 2) {
        await new Promise(resolve => setTimeout(resolve, 100));
        continue;
      }
      canvas.width = Math.min(640, video.videoWidth);
      canvas.height = Math.round(video.videoHeight * canvas.width / video.videoWidth);
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.8));
      if (cameraToken !== token) return;
      if (!blob) throw new Error("Could not capture a camera frame.");
      const controller = new AbortController();
      uploadAbort = controller;
      const timer = setTimeout(() => controller.abort(), 10000);
      let response;
      try {
        response = await fetch("/api/camera/frame", {
          method: "POST", headers: {"Content-Type": "image/jpeg", "X-Camera-Token": token},
          body: blob, signal: controller.signal,
        });
      } finally {
        clearTimeout(timer);
      }
      if (!response.ok) throw new Error(response.status === 409
        ? "Camera session expired. Start the camera again." : "Frame upload failed. Check the container logs.");
      if (cameraToken === token) $("#camera-message").textContent = "Camera active. Keep this tab open; closing it stops capture.";
      await new Promise(resolve => setTimeout(resolve, 250));
    } catch (error) {
      if (cameraToken === token) stopCamera(error.name === "AbortError"
        ? "Connection timed out. Check Docker and start the camera again." : error.message);
      return;
    }
  }
}

$("#camera-start").addEventListener("click", async () => {
  if (cameraStarting || cameraMedia) return;
  cameraStarting = true;
  $("#camera-start").disabled = true;
  $("#camera-message").textContent = "Waiting for camera permission…";
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera access needs localhost or HTTPS. Open this dashboard using localhost.");
    await cameraRelease;
    const deviceId = $("#camera-device").value;
    cameraMedia = await navigator.mediaDevices.getUserMedia({
      audio: false, video: {width: {ideal: 640}, height: {ideal: 480},
        ...(deviceId ? {deviceId: {exact: deviceId}} : {})},
    });
    const preview = $("#camera-preview");
    preview.srcObject = cameraMedia;
    preview.hidden = false;
    await preview.play();
    await cameraDevices(cameraMedia.getVideoTracks()[0].getSettings().deviceId);
    const response = await fetch("/api/camera/start", {method: "POST"});
    if (!response.ok) throw new Error(response.status === 409
      ? "Another tab is sending camera video. Stop it or wait 15 seconds after closing it." : "Could not start camera capture.");
    cameraToken = (await response.json()).token;
    cameraMedia.getVideoTracks()[0].addEventListener("ended", () => stopCamera("Camera disconnected. Reconnect it and start again."));
    $("#camera-stop").disabled = false;
    $("#camera-device").disabled = true;
    uploadCamera(cameraToken);
  } catch (error) {
    stopCamera(error.name === "NotAllowedError" ? "Camera permission denied. Allow camera access in your browser and try again." : error.message);
  } finally {
    cameraStarting = false;
    $("#camera-start").disabled = !!cameraMedia;
  }
});
$("#camera-stop").addEventListener("click", () => stopCamera());
window.addEventListener("pagehide", () => stopCamera());
cameraDevices().catch(() => {});

async function poll() {
  try {
    const r = await fetch("/api/state", {cache: "no-store"});
    const data = await r.json();
    state = data.live; identities = data.identities;
    render();
  } catch (e) {
    $("#dot").className = "dot down"; $("#status").textContent = "Dashboard disconnected";
  } finally {
    setTimeout(poll, 1500);
  }
}

function render() {
  const s = state, now = Date.now() / 1000;
  $("#source").textContent = "Source: " + s.source;
  $("#camera-controls").hidden = s.source !== "browser";
  const live = s.running && s.last_frame_ts && now - s.last_frame_ts < 10;
  $("#dot").className = "dot" + (live ? " live" : "");
  $("#status").textContent = live ? "Live" : s.source === "browser" ? "Waiting for camera" : s.running ? "Waiting for frames" : "Source finished";

  const e = s.email;
  const emailSub = !e.enabled ? "Email alerts disabled"
    : e.failed ? `${e.failed} failed` + (e.last ? ` · last ${ago(e.last)}` : "")
    : e.last ? `Last sent ${ago(e.last)}` : "None sent yet";
  const tiles = [
    ["Inference device", s.compute.using_gpu ? "GPU" : s.compute.device.toUpperCase(),
      `${s.compute.cuda_available ? "CUDA available" : "CUDA unavailable"} · ${s.compute.device}`],
    ["Identities", compact(identities.length), `${s.new_people} new this session`],
    ["Detections", compact(s.detections), `${compact(s.frames)} frames analysed`],
    ["Processing rate", s.fps.toFixed(1) + " fps", "Analysed frames per second"],
    ["Email alerts", compact(e.sent), emailSub],
    ["Uptime", duration(now - s.started), "Since " + clock(s.started)],
  ];
  setHTML($("#kpis"), tiles.map(([l, v, sub]) =>
    `<div class="tile"><div class="label">${l}</div><div class="value">${v}</div><div class="sub">${esc(sub)}</div></div>`).join(""));

  $("#recent-count").textContent = s.recent.length ? `Latest ${s.recent.length}` : "";
  setHTML($("#activity"), s.recent.length ? s.recent.map(ev => `
    <div class="event" data-name="${esc(ev.identity)}">
      <img src="${ev.img}" alt="">
      <div><div class="who">${esc(ev.identity)}</div><div class="meta">${ago(ev.ts)}</div></div>
      <div class="right">${ev.is_new
        ? `<span class="tag" style="--c:var(--new)">New person</span>`
        : `<span class="tag" style="--c:var(--accent)">Match ${ev.score.toFixed(2)}</span>`}</div>
    </div>`).join("") : `<div class="empty">No detections yet.</div>`);

  renderChart(s.timeline);
  renderIdentities();
}

function renderChart(tl) {
  const box = $("#chart"), W = Math.max(320, box.clientWidth), H = 170, pl = 30, pb = 22, pt = 8;
  const raw = Math.max(...tl.map(d => d.count), 1);
  const step = raw <= 4 ? 1 : Math.pow(10, Math.floor(Math.log10(raw / 2))) * (raw / 2 / Math.pow(10, Math.floor(Math.log10(raw / 2))) > 5 ? 5 : raw / 2 / Math.pow(10, Math.floor(Math.log10(raw / 2))) > 2 ? 2 : 1);
  const max = Math.ceil(raw / step) * step;
  const ih = H - pt - pb, iw = W - pl, bw = iw / tl.length;
  const y = v => pt + ih - v / max * ih;
  let grid = "";
  for (let v = 0; v <= max; v += step) {
    if (max / step > 5 && v % (step * 2) && v !== max) continue;
    grid += `<line x1="${pl}" x2="${W}" y1="${y(v)}" y2="${y(v)}"/><text x="${pl - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`;
  }
  const rr = (x, top, w, h, r) => { r = Math.min(r, w / 2, h);
    return `M${x},${top + h}V${top + r}Q${x},${top} ${x + r},${top}H${x + w - r}Q${x + w},${top} ${x + w},${top + r}V${top + h}Z`; };
  let bars = "";
  tl.forEach((d, i) => {
    const x = pl + i * bw + 1, w = Math.max(1, bw - 2), h = d.count / max * ih;
    bars += `<rect class="hit" data-i="${i}" x="${pl + i * bw}" y="${pt}" width="${bw}" height="${ih}"/>`;
    bars += h > 0 ? `<path class="bar" d="${rr(x, y(d.count), w, h, 4)}"/>` : `<path class="bar"/>`;
  });
  const labels = [0, Math.floor(tl.length / 2), tl.length - 1].map(i =>
    `<text x="${pl + i * bw + bw / 2}" y="${H - 6}" text-anchor="middle">${i === tl.length - 1 ? "now" : clock(tl[i].t)}</text>`).join("");
  const total = tl.reduce((a, d) => a + d.count, 0);
  const svg = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="Detections per minute over the last 30 minutes, ${total} total">
    <g class="grid">${grid}</g>${bars}${labels}</svg>`;
  const old = box.querySelector("svg");
  if (!old || old._key !== svg) {
    if (old) old.remove();
    box.insertAdjacentHTML("afterbegin", svg);
    const el = box.querySelector("svg"); el._key = svg;
    const tip = $("#tip");
    el.querySelectorAll(".hit").forEach(r => {
      r.addEventListener("mouseenter", () => {
        const d = tl[+r.dataset.i];
        tip.innerHTML = `<b>${d.count}</b> detection${d.count === 1 ? "" : "s"} · ${clock(d.t)}`;
        tip.style.left = (+r.getAttribute("x") + bw / 2) / W * box.clientWidth + "px";
        tip.style.top = y(d.count) + "px";
        tip.style.display = "block";
      });
      r.addEventListener("mouseleave", () => tip.style.display = "none");
    });
  }
}

function renderIdentities() {
  const q = $("#q").value.trim().toLowerCase(), key = $("#sort").value;
  let list = identities.filter(p => p.name.toLowerCase().includes(q));
  list.sort((a, b) => key === "name" ? a.name.localeCompare(b.name) : b[key] - a[key]);
  $("#id-count").textContent = `· ${identities.length}`;
  setHTML($("#ids"), list.length ? list.map(p => `
    <div class="card" data-name="${esc(p.name)}">
      <img src="${esc(p.thumb)}" alt="${esc(p.name)}" loading="lazy">
      <div class="body">
        <div class="name">${esc(p.name)}</div>
        <div class="meta">${p.sightings} sighting${p.sightings === 1 ? "" : "s"} · ${ago(p.last_seen)}</div>
      </div>
    </div>`).join("") : `<div class="empty">${identities.length ? "No identities match your search." : "No people identified yet."}</div>`);
}

async function openIdentity(name) {
  const r = await fetch("/api/identity/" + encodeURIComponent(name), {cache: "no-store"});
  if (!r.ok) return;
  const d = await r.json();
  $("#dialog").innerHTML = `
    <div class="dialog-head"><h3>${esc(d.name)}</h3><button class="close" data-close>Close</button></div>
    <div class="detail">
      <img class="frame" src="${esc(d.frame)}?t=${Date.now()}" alt="Latest annotated frame for ${esc(d.name)}">
      <div class="facts">
        <div class="fact"><div class="k">Sightings</div><div class="v">${d.sightings}</div></div>
        <div class="fact"><div class="k">First seen</div><div class="v">${stamp(d.created)}</div></div>
        <div class="fact"><div class="k">Last seen</div><div class="v">${stamp(d.last_seen)} <span class="hint">(${ago(d.last_seen)})</span></div></div>
        <div class="fact"><div class="k">Last match score</div><div class="v">${d.last_score == null ? "—" : d.last_score.toFixed(3)}</div></div>
        <div class="fact"><div class="k">Stored samples</div><div class="v">${d.samples.length}</div></div>
      </div>
    </div>
    <div class="panel-head" style="margin-top:18px"><h2>Samples</h2><span class="hint">Similarity to the latest sample</span></div>
    <div class="samples">${d.samples.map((s, i) => `
      <div class="sample">
        <img src="${esc(s.url)}" alt="" loading="lazy">
        <div class="meter"><div style="width:${Math.max(0, s.similarity) * 100}%"></div></div>
        <div class="meta"><span>${i === 0 ? "latest" : s.similarity.toFixed(3)}</span><span>${clock(s.ts)}</span></div>
      </div>`).join("")}</div>`;
  $("#modal").classList.add("open");
}

document.addEventListener("click", ev => {
  const item = ev.target.closest("[data-name]");
  if (item) return openIdentity(item.dataset.name);
  if (ev.target.matches("[data-close]") || ev.target.id === "modal") $("#modal").classList.remove("open");
});
document.addEventListener("keydown", ev => { if (ev.key === "Escape") $("#modal").classList.remove("open"); });
$("#q").addEventListener("input", () => identities.length && renderIdentities());
$("#sort").addEventListener("change", () => identities.length && renderIdentities());
window.addEventListener("resize", () => state && renderChart(state.timeline));
poll();
</script>
</body>
</html>
"""


def build_dashboard(state, gallery, browser_camera=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    @app.before_request
    def check_camera_origin():
        # Camera control and uploads are only accepted from this dashboard.
        if request.method == "POST" and request.path.startswith("/api/camera/"):
            origin = request.headers.get("Origin")
            if origin and origin != request.host_url.rstrip("/"):
                abort(403)
            if browser_camera is None:
                abort(404)

    @app.post("/api/camera/start")
    def camera_start():
        token = browser_camera.start()
        if token is None:
            return jsonify(error="Another browser camera is active"), 409
        return jsonify(token=token)

    @app.post("/api/camera/frame")
    def camera_frame():
        if request.mimetype != "image/jpeg":
            abort(415)
        data = request.get_data()
        if not data:
            abort(400, "Empty frame")
        try:
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        except cv2.error:
            frame = None
        if frame is None or max(frame.shape[:2]) > 1920 or frame.shape[0] * frame.shape[1] > 1920 * 1080:
            abort(400, "Invalid frame or frame exceeds 1920x1080 pixels")
        if not browser_camera.submit(request.headers.get("X-Camera-Token"), frame):
            return jsonify(error="Camera session expired"), 409
        state.running = True
        return "", 204

    @app.post("/api/camera/stop")
    def camera_stop():
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict) or not browser_camera.stop(data.get("token")):
            return jsonify(error="Camera session expired"), 409
        state.running = False
        return "", 204

    @app.route("/")
    def index():
        return Response(DASHBOARD_HTML, mimetype="text/html")

    @app.route("/api/state")
    def api_state():
        return jsonify(live=state.snapshot(), identities=gallery.summary())

    @app.route("/api/identity/<name>")
    def api_identity(name):
        info = gallery.detail(name)
        if info is None:
            abort(404)
        return jsonify(info)

    @app.route("/api/recent/<int:event_id>.jpg")
    def api_recent(event_id):
        jpeg = state.recent_jpeg(event_id)
        if jpeg is None:
            abort(404)
        return Response(jpeg, mimetype="image/jpeg", headers={"Cache-Control": "max-age=3600"})

    @app.route("/gallery/<path:filename>")
    def gallery_file(filename):
        return send_from_directory(gallery.root.resolve(), filename)

    @app.route("/stream.mjpg")
    def stream():
        def frames():
            last = -1
            while True:
                with state.frame_cond:
                    state.frame_cond.wait_for(lambda: state.frame_seq != last, timeout=5)
                    seq, jpeg = state.frame_seq, state.frame_jpeg
                if jpeg is None:
                    continue
                last = seq  # on timeout we resend the last frame, which also detects closed clients
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")

        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


# --------------------------------------------------------------------------- main

def parse_args():
    p = argparse.ArgumentParser(description="YOLO person segmentation with masked embedding re-identification, email alerts for new people, and a live dashboard.")
    p.add_argument("--source", default="browser", help="'browser' for dashboard webcam capture, a path, stream URL, or camera index")
    p.add_argument("--model", default="yolo26n-seg.pt", help="YOLO segmentation weights file (e.g. yolo26n-seg.pt)")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda:0', ...")
    p.add_argument("--frame-gap", type=int, default=15, help="Run detection on every Nth frame only")
    p.add_argument("--threshold", type=float, default=0.70, help="Cosine similarity needed to count as the same person")
    p.add_argument("--track-threshold", type=float, default=0.60, help="Minimum appearance similarity to keep a recent overlapping person's identity")
    p.add_argument("--track-iou", type=float, default=0.3, help="Minimum box overlap (IoU) for short-term identity continuity")
    p.add_argument("--track-timeout", type=float, default=3.0, help="Seconds before a short-term identity track expires (also expires after 3 analyzed frames)")
    p.add_argument("--min-box", type=int, default=48, help="Ignore person boxes shorter than this many pixels")
    p.add_argument("--sample-interval", type=float, default=3.0, help="Seconds between stored samples of the same identity")
    p.add_argument("--max-samples", type=int, default=30, help="Samples per identity kept in memory for matching")
    p.add_argument("--gallery", default="gallery_segmented", help="Folder for masked-person identities; use a separate gallery from unmasked embeddings")
    p.add_argument("--cooldown", type=float, default=60.0, help="Minimum seconds between alert emails")
    p.add_argument("--sender", default=os.getenv("EMAIL_SENDER", ""), help="Sender email (or EMAIL_SENDER)")
    p.add_argument("--password", default=os.getenv("EMAIL_PASSWORD", ""), help="Email app password (or EMAIL_PASSWORD)")
    p.add_argument("--receiver", default=os.getenv("EMAIL_RECEIVER", ""), help="Receiver email (or EMAIL_RECEIVER)")
    p.add_argument("--smtp-server", default=os.getenv("SMTP_SERVER", "smtp.gmail.com"))
    p.add_argument("--smtp-port", type=int, default=int(os.getenv("SMTP_PORT", "465")))
    p.add_argument("--no-ssl", action="store_true", help="Use STARTTLS instead of SMTP_SSL")
    p.add_argument("--no-email", action="store_true", help="Disable email alerts")
    p.add_argument("--dashboard", action="store_true", help="Start the live web dashboard")
    p.add_argument("--host", default="127.0.0.1", help="Dashboard bind address (0.0.0.0 to expose on the network)")
    p.add_argument("--dashboard-port", type=int, default=5000)
    p.add_argument("--latest-count", type=int, default=12, help="Recent detections shown on the dashboard")
    p.add_argument("--verbose", action="store_true", help="Log every re-identification, not only new people")
    return p.parse_args()


def main():
    args = parse_args()
    if args.source == "browser" and not args.dashboard:
        raise ValueError("--source browser requires --dashboard to capture camera frames.")
    device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"Using device: {device}")

    detector = YOLO(args.model)
    if detector.task != "segment":
        raise ValueError("Use YOLO segmentation weights, such as --model yolo26n-seg.pt.")

    email_ready = bool(args.sender and args.receiver and args.password) and args.password != "your_app_password"
    if not args.no_email and not email_ready:
        print("Email alerts disabled: set EMAIL_SENDER, EMAIL_PASSWORD and EMAIL_RECEIVER (in .env or via flags).")
    email_on = email_ready and not args.no_email

    io = ThreadPoolExecutor(max_workers=1, thread_name_prefix="disk")  # one worker keeps writes ordered
    embedder = Embedder(device)
    gallery = Gallery(args.gallery, embedder.dim, max(1, args.max_samples), io)
    print(f"Loaded {len(gallery.people)} known identities from '{args.gallery}/'")
    state = LiveState(str(args.source), max(1, args.latest_count), email_on, device=device)
    browser_camera = BrowserCamera() if args.source == "browser" else None

    emailer = None
    if email_on:
        emailer = Emailer(args.sender, args.password, args.receiver, args.smtp_server,
                          args.smtp_port, not args.no_ssl, state.email_result)
        emailer.start()

    if args.dashboard:
        app = build_dashboard(state, gallery, browser_camera)
        threading.Thread(
            target=app.run, daemon=True, name="dashboard",
            kwargs={"host": args.host, "port": args.dashboard_port, "threaded": True, "use_reloader": False},
        ).start()
        print(f"Dashboard: http://localhost:{args.dashboard_port}")

    pipeline = Pipeline(args, embedder, gallery, state, emailer)
    predict_options = dict(conf=args.conf, imgsz=args.imgsz, classes=[0], device=device,
                           half=device.startswith("cuda"), retina_masks=True, verbose=False)

    try:
        if browser_camera is not None:
            print("Open the dashboard and click Start camera. Waiting for browser frames...")
            while True:
                frame = browser_camera.read()
                if frame is not None:
                    # Browser uploads are already rate limited; frame-gap applies to video streams only.
                    for result in detector(source=frame, **predict_options):
                        pipeline.process(result)
        else:
            stream = detector(source=args.source, stream=True,
                              vid_stride=max(1, args.frame_gap), **predict_options)
            for result in stream:
                pipeline.process(result)
        state.running = False
        if args.dashboard:
            print("Source finished. Dashboard is still running - press Ctrl+C to exit.")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        state.running = False
        gallery.flush_meta()
        io.shutdown(wait=True)
        if emailer:
            emailer.stop()
            emailer.join(timeout=30)


if __name__ == "__main__":
    main()
