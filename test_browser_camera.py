"""Run with: python -m unittest test_browser_camera -v"""

import time
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from detect_and_email import (BrowserCamera, CameraLiveState, Gallery, LiveState, Pipeline,
                              build_dashboard, process_browser_batch)


class BrowserCameraTests(unittest.TestCase):
    def setUp(self):
        self.camera = BrowserCamera()
        self.state = LiveState("browser", 12, False)
        self.gallery = Mock()
        self.gallery.summary.return_value = []
        self.client = build_dashboard(self.state, self.gallery, self.camera).test_client()
        self.jpeg = cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))[1].tobytes()

    def start(self):
        response = self.client.post("/api/camera/start")
        self.assertEqual(response.status_code, 200)
        return response.json["token"]

    def upload(self, token, jpeg=None):
        return self.client.post("/api/camera/frame", data=self.jpeg if jpeg is None else jpeg,
                                content_type="image/jpeg", headers={"X-Camera-Token": token})

    def test_upload_decodes_and_latest_frame_replaces_pending_frame(self):
        token = self.start()
        self.assertEqual(self.upload(token).status_code, 204)
        white = cv2.imencode(".jpg", np.full((48, 64, 3), 255, np.uint8))[1].tobytes()
        self.assertEqual(self.upload(token, white).status_code, 204)
        frame = self.camera.read()
        self.assertEqual(frame.shape, (48, 64, 3))
        self.assertGreater(frame.mean(), 250)
        self.assertIsNone(self.camera.pending)

    def test_one_camera_owner_and_stop_releases_it(self):
        token = self.start()
        self.assertEqual(self.client.post("/api/camera/start").status_code, 409)
        self.assertEqual(self.upload("another-tab").status_code, 409)
        self.assertEqual(self.client.post("/api/camera/stop", json={"token": "wrong"}).status_code, 409)
        self.assertEqual(self.upload(token).status_code, 204)
        self.assertEqual(self.client.post("/api/camera/stop", json={"token": token}).status_code, 204)
        self.assertIsNone(self.camera.pending)
        self.assertEqual(self.upload(token).status_code, 409)
        self.assertNotEqual(self.start(), token)

    def test_closed_tab_expires_and_can_be_replaced(self):
        with patch("detect_and_email.time.monotonic", return_value=100):
            token = self.start()
            self.assertEqual(self.upload(token).status_code, 204)
        with patch("detect_and_email.time.monotonic", return_value=116):
            self.assertEqual(self.upload(token).status_code, 409)
            self.assertIsNone(self.camera.read())
            replacement = self.start()
            self.assertNotEqual(replacement, token)
            self.assertEqual(self.upload(replacement).status_code, 204)

    def test_invalid_and_oversize_frames_rejected_without_refreshing_session(self):
        token = self.start()
        original_time = self.camera.last_upload
        for data in (b"", b"invalid jpeg"):
            self.assertEqual(self.upload(token, data).status_code, 400)
        self.assertEqual(self.upload(token, b"x" * (2 * 1024 * 1024 + 1)).status_code, 413)
        big = cv2.imencode(".jpg", np.zeros((1081, 1920, 3), np.uint8))[1].tobytes()
        self.assertEqual(self.upload(token, big).status_code, 400)
        self.assertEqual(self.client.post("/api/camera/frame", data=self.jpeg).status_code, 415)
        self.assertEqual(self.camera.last_upload, original_time)
        self.assertEqual(self.upload(token).status_code, 204)

    def test_cross_origin_control_rejected(self):
        response = self.client.post("/api/camera/start", headers={"Origin": "https://another.example"})
        self.assertEqual(response.status_code, 403)
        self.assertIsNone(self.camera.token)
        self.assertEqual(self.client.post("/api/camera/start", headers={"Origin": "http://localhost"}).status_code, 200)

    def test_status_idle_live_stale_and_stopped(self):
        self.assertFalse(self.client.get("/api/state").json["live"]["running"])
        token = self.start()
        self.upload(token)
        camera_state = self.state.cameras[0]
        camera_state.tick_frame(time.time())
        self.assertTrue(self.client.get("/api/state").json["live"]["running"])
        camera_state.last_received_ts -= 16
        self.assertFalse(self.client.get("/api/state").json["live"]["running"])
        camera_state.processing = True
        self.assertTrue(self.client.get("/api/state").json["live"]["running"])
        camera_state.processing = False
        self.upload(token)
        self.client.post("/api/camera/stop", json={"token": token})
        self.assertFalse(self.client.get("/api/state").json["live"]["running"])

    def test_direct_camera_has_no_upload_api_and_survives_dashboard_requests(self):
        state = LiveState("0", 12, False)
        client = build_dashboard(state, self.gallery).test_client()
        self.assertEqual(client.post("/api/camera/start").status_code, 404)
        self.assertEqual(client.post("/api/camera/stop", json={}).status_code, 404)
        self.assertEqual(client.get("/").status_code, 200)
        self.assertTrue(client.get("/api/state").json["live"]["running"])


class TwoCameraTests(unittest.TestCase):
    def setUp(self):
        self.cameras = [BrowserCamera(), BrowserCamera()]
        self.state = LiveState("browser", 12, False, device="cpu")
        self.gallery = Mock()
        self.gallery.summary.return_value = []
        self.client = build_dashboard(self.state, self.gallery, self.cameras).test_client()
        self.tokens = [self.client.post(f"/api/camera/{i}/start").json["token"] for i in range(2)]

    def upload(self, camera_id, value=100, token=None):
        jpeg = cv2.imencode(".jpg", np.full((48, 64, 3), value, np.uint8))[1].tobytes()
        return self.client.post(f"/api/camera/{camera_id}/frame", data=jpeg, content_type="image/jpeg",
                                headers={"X-Camera-Token": token or self.tokens[camera_id]})

    def pipelines(self):
        result = []
        for state in self.state.cameras:
            pipeline = Mock(state=state, tracks={})
            pipeline.camera_session = None
            pipeline.process.side_effect = lambda result, state=state: state.tick_frame(time.time())
            result.append(pipeline)
        return result

    def test_independent_upload_stop_and_ownership(self):
        self.assertEqual(self.upload(0).status_code, 204)
        self.assertEqual(self.upload(1, 200).status_code, 204)
        self.assertEqual(self.upload(1, token=self.tokens[0]).status_code, 409)
        self.assertEqual(self.client.post("/api/camera/0/start").status_code, 409)
        self.assertEqual(self.client.post("/api/camera/0/stop", json={"token": self.tokens[0]}).status_code, 204)
        self.assertEqual(self.upload(1, 210).status_code, 204)
        self.assertIsNone(self.cameras[0].pending)
        self.assertGreater(self.cameras[1].pending.mean(), 200)
        live = self.client.get("/api/state").json["live"]
        self.assertTrue(live["running"])
        self.assertFalse(live["cameras"][0]["running"])
        self.assertEqual([c["received"] for c in live["cameras"]], [1, 2])

    def test_ready_cameras_are_batched_and_inactive_camera_does_not_block(self):
        pipelines = self.pipelines()
        detector = Mock(side_effect=lambda **kw: [object() for _ in kw["source"]])
        self.upload(1, 200)
        self.assertTrue(process_browser_batch(detector, self.cameras, pipelines, {}))
        self.assertEqual(pipelines[0].process.call_count, 0)
        self.assertEqual(pipelines[1].process.call_count, 1)
        self.upload(0, 100)
        self.upload(1, 200)
        process_browser_batch(detector, self.cameras, pipelines, {})
        frames = detector.call_args.kwargs["source"]
        self.assertEqual(len(frames), 2)
        self.assertLess(frames[0].mean(), frames[1].mean())
        self.assertEqual([s.frames for s in self.state.cameras], [1, 2])
        self.assertEqual(self.state.frames, 3)
        self.assertFalse(process_browser_batch(detector, self.cameras, pipelines, {}))

    def test_feed_and_events_identify_the_correct_camera(self):
        for i, camera_state in enumerate(self.state.cameras):
            jpeg = cv2.imencode(".jpg", np.full((48, 64, 3), i * 200, np.uint8))[1].tobytes()
            camera_state.push_frame(jpeg)
            response = self.client.get(f"/stream/{i}.mjpg", buffered=False)
            self.assertIn(jpeg, next(response.response))
            response.close()
            camera_state.add_event("person_0001", 0.9, False, jpeg, time.time())
        events = self.client.get("/api/state").json["live"]["recent"]
        self.assertEqual([e["camera_id"] for e in events], [1, 0])
        html = self.client.get("/").text
        self.assertIn('data-camera="0"', html)
        self.assertIn('data-camera="1"', html)
        self.assertNotIn("__CAMERA_PANELS__", html)
        self.assertEqual(self.client.post("/api/camera/2/start").status_code, 404)
        self.assertEqual(self.client.get("/stream/2.mjpg").status_code, 404)

    def test_new_session_resets_only_its_own_tracks(self):
        pipelines = self.pipelines()
        detector = Mock(side_effect=lambda **kw: [object() for _ in kw["source"]])
        for i in range(2): self.upload(i)
        process_browser_batch(detector, self.cameras, pipelines, {})
        for pipeline in pipelines: pipeline.tracks["old"] = object()
        self.client.post("/api/camera/0/stop", json={"token": self.tokens[0]})
        self.tokens[0] = self.client.post("/api/camera/0/start").json["token"]
        for i in range(2): self.upload(i)
        process_browser_batch(detector, self.cameras, pipelines, {})
        self.assertEqual(pipelines[0].tracks, {})
        self.assertIn("old", pipelines[1].tracks)

    def test_inference_failure_clears_processing_status(self):
        for i in range(2): self.upload(i)
        with self.assertRaises(RuntimeError):
            process_browser_batch(Mock(side_effect=RuntimeError("inference failed")), self.cameras, self.pipelines(), {})
        self.assertFalse(any(s.processing for s in self.state.cameras))

    def test_batch_arrival_does_not_inflate_total_processing_rate(self):
        for timestamp in (100, 100.001, 101, 101.001):
            self.state.tick_frame(timestamp)
        self.assertGreater(self.state.fps, 2)
        self.assertLess(self.state.fps, 4)

    def test_overlapping_boxes_do_not_transfer_tracks_between_cameras(self):
        import torch
        from ultralytics.engine.results import Results
        with tempfile.TemporaryDirectory() as folder:
            gallery = Gallery(folder, 2, 30, Mock())
            def embed(crops):
                return np.array([[1, 0] if crop.mean() < 150 else [0.65, np.sqrt(1 - 0.65**2)] for crop in crops], dtype=np.float32).reshape(-1, 2)
            args = SimpleNamespace(min_box=4, dashboard=True, threshold=0.7, track_threshold=0.6,
                                   track_iou=0.3, track_timeout=3, sample_interval=3, verbose=False)
            pipelines = [Pipeline(args, embed, gallery, state, None) for state in self.state.cameras]
            def detect(**kwargs):
                results = []
                for image in kwargs["source"]:
                    masks = torch.zeros((1, 48, 64))
                    masks[0, 4:40, 4:20] = 1
                    results.append(Results(image, path="camera", names={0: "person"},
                                           boxes=torch.tensor([[4, 4, 20, 40, 0.9, 0]]), masks=masks))
                return results
            self.upload(0, 100)
            self.upload(1, 200)
            process_browser_batch(detect, self.cameras, pipelines, {})
            self.assertEqual(len(gallery.people), 2)
            self.assertEqual(list(pipelines[0].tracks), ["person_0001"])
            self.assertEqual(list(pipelines[1].tracks), ["person_0002"])


if __name__ == "__main__":
    unittest.main()
