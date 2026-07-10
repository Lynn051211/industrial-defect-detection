"""缺陷检测引擎 — 企业级封装（ultralytics YOLO 驱动，支持 ONNX/PT）"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO
from loguru import logger


@dataclass
class DetectionResult:
    """单次推理结果"""
    num_detections: int
    inference_time_ms: float
    detections: list[dict]
    image_path: Optional[str] = None


class DefectDetector:
    """工业缺陷检测器 — ultralytics YOLO 引擎（自动处理预处理+后处理）"""

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.35,
        iou_threshold: float = 0.45,
        input_size: int = 160,
        class_names: Optional[list[str]] = None,
        class_names_zh: Optional[dict[str, str]] = None,
    ):
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.class_names = class_names or []
        self.class_names_zh = class_names_zh or {}

        _path = Path(model_path)
        if not _path.exists():
            raise FileNotFoundError(f"模型不存在: {model_path}")

        logger.info(f"加载模型: {model_path} ({_path.stat().st_size / 1024 / 1024:.1f} MB)")

        # ultralytics 原生支持 ONNX/PT
        self._model = YOLO(str(_path))

        logger.info(f"输入尺寸: {input_size}, 类别: {len(self.class_names)}, "
                    f"置信度: {confidence}, IoU: {iou_threshold}")

        # 预热
        self._warmup()

    def _warmup(self, runs: int = 3):
        """模型预热"""
        logger.info(f"预热 ({runs} 次)...")
        dummy = np.random.randint(0, 255, (self.input_size, self.input_size, 3), dtype=np.uint8)
        for _ in range(runs):
            self._model(dummy, imgsz=self.input_size, conf=self.confidence,
                        iou=self.iou_threshold, verbose=False)
        logger.info("预热完成")

    def detect(self, image: np.ndarray) -> list[dict]:
        """推理 BGR 图像，返回检测列表"""
        results = self._model(
            image,
            imgsz=self.input_size,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections = []
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            for cls_id, conf, xyxy in zip(
                r.boxes.cls.cpu().numpy().astype(int),
                r.boxes.conf.cpu().numpy(),
                r.boxes.xyxy.cpu().numpy().astype(int),
            ):
                cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else "unknown"
                detections.append({
                    "class": cls_name,
                    "class_zh": self.class_names_zh.get(cls_name, cls_name),
                    "confidence": round(float(conf), 4),
                    "bbox": xyxy.tolist(),
                })

        return detections

    def predict_file(self, image_path: str, benchmark: bool = False) -> DetectionResult:
        """推理图像文件"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        if benchmark:
            self._warmup(1)
            times = []
            for _ in range(10):
                t0 = time.perf_counter()
                _ = self.detect(img)
                times.append((time.perf_counter() - t0) * 1000)
            avg_ms = float(np.mean(times))
        else:
            t0 = time.perf_counter()
            detections = self.detect(img)
            avg_ms = (time.perf_counter() - t0) * 1000

        detections = self.detect(img)

        return DetectionResult(
            num_detections=len(detections),
            inference_time_ms=round(avg_ms, 2),
            detections=detections,
            image_path=image_path,
        )

    def draw(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        """在图像上绘制检测框"""
        img = image.copy()
        colors = [
            (0, 0, 255), (0, 255, 0), (255, 0, 0),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
        ]

        for det in detections:
            b = det["bbox"]
            cls_idx = self.class_names.index(det["class"]) if det["class"] in self.class_names else 0
            color = colors[cls_idx % len(colors)]

            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), color, 2)
            label = f"{det['class_zh']} {det['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (b[0], b[1] - th - 6), (b[0] + tw + 4, b[1]), color, -1)
            cv2.putText(img, label, (b[0] + 2, b[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img
