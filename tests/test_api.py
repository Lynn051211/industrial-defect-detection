"""API 集成测试"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestHealthCheck:
    """健康检查"""

    def test_health(self):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_model_info(self):
        resp = client.get("/api/v1/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_classes"] == 6
        assert data["input_size"] == 160


class TestDetectAPI:
    """检测 API"""

    def test_detect_no_file(self):
        """缺少文件参数返回 422"""
        resp = client.post("/api/v1/detect")
        assert resp.status_code == 422

    def test_detect_invalid_type(self):
        """无效文件类型返回 400"""
        resp = client.post(
            "/api/v1/detect",
            files={"file": ("test.txt", b"hello world", "text/plain")},
        )
        assert resp.status_code == 400

    def test_detect_valid_image(self):
        """上传有效图像返回检测结果"""
        import cv2
        import numpy as np

        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)

        resp = client.post(
            "/api/v1/detect",
            files={"file": ("test.jpg", buf.tobytes(), "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "num_detections" in data
        assert "inference_time_ms" in data
        assert "detections" in data
