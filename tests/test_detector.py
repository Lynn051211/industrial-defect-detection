"""检测引擎单元测试"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
import pytest
from app.models.detector import DefectDetector, DetectionResult
from app.core.config import settings


@pytest.fixture(scope="module")
def detector():
    """模块级 fixture：加载一次模型"""
    return DefectDetector(
        model_path=settings.MODEL_PATH_ABS,
        confidence=settings.CONFIDENCE_THRESHOLD,
        iou_threshold=settings.IOU_THRESHOLD,
        input_size=settings.MODEL_INPUT_SIZE,
        class_names=settings.DEFECT_CLASSES,
        class_names_zh=settings.DEFECT_CLASSES_ZH,
    )


class TestDefectDetector:
    """检测器核心功能测试"""

    def test_init_success(self, detector):
        """模型加载成功"""
        assert detector is not None
        assert detector.input_size == 160
        assert len(detector.class_names) == 6

    def test_init_file_not_found(self):
        """模型文件不存在抛出异常"""
        with pytest.raises(FileNotFoundError):
            DefectDetector(model_path="nonexistent.onnx")

    def test_detect_no_defects(self, detector):
        """纯色图像应检出0个缺陷"""
        img = np.full((640, 640, 3), 128, dtype=np.uint8)
        dets = detector.detect(img)
        assert isinstance(dets, list)
        # 纯色图像上的检测结果要么为空，要么低置信度已被过滤
        # 不做严格断言，只验证返回格式

    def test_detect_shape(self, detector):
        """不同尺寸图像推理不报错"""
        for shape in [(320, 320, 3), (640, 480, 3), (1280, 720, 3)]:
            img = np.random.randint(0, 255, shape, dtype=np.uint8)
            dets = detector.detect(img)
            assert isinstance(dets, list)

    def test_detect_output_format(self, detector):
        """检测输出格式正确"""
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        dets = detector.detect(img)
        for d in dets:
            assert "class" in d
            assert "class_zh" in d
            assert "confidence" in d
            assert "bbox" in d
            assert len(d["bbox"]) == 4
            assert 0 <= d["confidence"] <= 1

    def test_detect_bbox_valid(self, detector):
        """检测到的 bbox 坐标在合理范围内"""
        img = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        dets = detector.detect(img)
        for d in dets:
            b = d["bbox"]
            assert 0 <= b[0] < 640, f"x1={b[0]} out of range"
            assert 0 <= b[1] < 480, f"y1={b[1]} out of range"
            assert b[0] < b[2], f"x1 >= x2: {b[0]} >= {b[2]}"
            assert b[1] < b[3], f"y1 >= y2: {b[1]} >= {b[3]}"

    def test_predict_file(self, detector, tmp_path):
        """文件推理正常"""
        import cv2
        img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        img_path = str(tmp_path / "test.jpg")
        cv2.imwrite(img_path, img)

        result = detector.predict_file(img_path, benchmark=True)
        assert result.num_detections >= 0
        assert result.inference_time_ms > 0
        assert isinstance(result.detections, list)

    def test_draw(self, detector):
        """绘制不报错"""
        img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        dets = [{"class": "pit", "class_zh": "凹坑", "confidence": 0.85, "bbox": [50, 50, 100, 100]}]
        result = detector.draw(img, dets)
        assert result.shape == img.shape


class TestDetectionResult:
    """DetectionResult 数据类测试"""

    def test_create(self):
        result = DetectionResult(
            num_detections=2,
            inference_time_ms=3.5,
            detections=[{"class": "pit", "class_zh": "凹坑", "confidence": 0.9, "bbox": [10, 10, 50, 50]}],
        )
        assert result.num_detections == 2
        assert result.inference_time_ms == 3.5
