"""检测器服务层 — 业务逻辑 + 单例管理"""

from typing import Optional
from loguru import logger

from app.core.config import settings
from app.models.detector import DefectDetector, DetectionResult

_detector: Optional[DefectDetector] = None


def get_detector() -> DefectDetector:
    """获取检测器单例（延迟初始化）"""
    global _detector
    if _detector is None:
        logger.info("初始化缺陷检测器...")
        _detector = DefectDetector(
            model_path=settings.MODEL_PATH_ABS,
            confidence=settings.CONFIDENCE_THRESHOLD,
            iou_threshold=settings.IOU_THRESHOLD,
            input_size=settings.MODEL_INPUT_SIZE,
            class_names=settings.DEFECT_CLASSES,
            class_names_zh=settings.DEFECT_CLASSES_ZH,
        )
        logger.info("检测器初始化完成")
    return _detector


def predict(image_path: str, benchmark: bool = False) -> DetectionResult:
    """业务入口：对图像文件执行缺陷检测"""
    detector = get_detector()
    return detector.predict_file(image_path, benchmark=benchmark)
