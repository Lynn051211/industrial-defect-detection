"""FastAPI 路由 — 缺陷检测 API"""

import os
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

from app.services.detector_service import predict, get_detector
from app.core.config import settings

router = APIRouter(prefix="/api/v1", tags=["defect-detection"])

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "defect-detector"}


@router.post("/detect")
async def detect_defects(file: UploadFile = File(...), benchmark: bool = False):
    """
    上传图像进行缺陷检测

    - **file**: 图像文件 (jpg/png/bmp)
    - **benchmark**: 是否执行性能基准测试（10次取平均）
    """
    # 校验文件类型
    if file.content_type not in ("image/jpeg", "image/png", "image/bmp"):
        raise HTTPException(400, f"不支持的格式: {file.content_type}，仅支持 jpg/png/bmp")

    # 保存临时文件
    ext = os.path.splitext(file.filename or "image.jpg")[1] or ".jpg"
    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"

    try:
        contents = await file.read()
        save_path.write_bytes(contents)
        logger.info(f"收到检测请求: {file.filename} → {save_path.name} ({len(contents)} bytes)")
    except Exception as e:
        logger.error(f"文件保存失败: {e}")
        raise HTTPException(500, f"文件保存失败: {e}")

    # 执行检测
    try:
        result = predict(str(save_path), benchmark=benchmark)
    except Exception as e:
        logger.error(f"推理失败: {e}")
        raise HTTPException(500, f"推理失败: {e}")

    # 生成结果图
    try:
        img = cv2.imread(str(save_path))
        from app.services.detector_service import get_detector
        detector = get_detector()
        result_img = detector.draw(img, result.detections)
        result_path = UPLOAD_DIR / f"result_{save_path.name}"
        cv2.imwrite(str(result_path), result_img)
    except Exception as e:
        logger.warning(f"结果图生成失败: {e}")
        result_path = None

    return JSONResponse({
        "success": True,
        "num_detections": result.num_detections,
        "inference_time_ms": result.inference_time_ms,
        "detections": result.detections,
        "result_image": str(result_path) if result_path else None,
    })


@router.get("/model/info")
async def model_info():
    """模型信息"""
    detector = get_detector()
    return {
        "model_path": str(settings.MODEL_PATH_ABS),
        "input_size": detector.input_size,
        "num_classes": len(detector.class_names),
        "classes": detector.class_names,
        "confidence_threshold": detector.confidence,
        "iou_threshold": detector.iou_threshold,
    }
