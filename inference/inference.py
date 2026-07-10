"""
ONNX Runtime / PyTorch 推理脚本
支持直接加载 ONNX 或 PT 模型进行推理。
使用 ultralytics 原生推理（自动处理预处理+后处理），结果准确。
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DEFECT_CLASSES, ONNX_MODEL_PATH, PT_MODEL_PATH


def infer_image(image_path: str, model_path: str = None, conf: float = 0.35):
    """
    推理单张图片

    Args:
        image_path: 图片路径
        model_path: 模型路径（.pt 或 .onnx），默认使用 ONNX
        conf: 置信度阈值

    Returns:
        dict: {time_ms, detections, image_path}
    """
    from ultralytics import YOLO

    model_path = model_path or ONNX_MODEL_PATH

    if not os.path.exists(model_path):
        print(f"错误：模型不存在 {model_path}")
        return None

    if not os.path.exists(image_path):
        print(f"错误：图片不存在 {image_path}")
        return None

    # ---- 加载模型 ----
    model = YOLO(model_path)

    # ---- 推理（测速） ----
    # 预热
    _ = model(image_path, imgsz=160, conf=conf, verbose=False)

    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        results = model(image_path, imgsz=160, conf=conf, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = np.mean(times)
    result = results[0]

    # ---- 解析结果 ----
    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        for cls_id, cls_conf, xyxy in zip(
            result.boxes.cls.cpu().numpy().astype(int),
            result.boxes.conf.cpu().numpy(),
            result.boxes.xyxy.cpu().numpy().astype(int),
        ):
            cls_name = DEFECT_CLASSES[cls_id] if cls_id < len(DEFECT_CLASSES) else "unknown"
            detections.append({
                "class": cls_name,
                "class_zh": cls_name,  # 保持兼容
                "confidence": round(float(cls_conf), 4),
                "bbox": xyxy.tolist(),
            })

    # ---- 打印结果 ----
    print(f"\n{'=' * 60}")
    print(f"  推理耗时: {avg_ms:.2f} ms (平均 10 次)")
    print(f"  检测到 {len(detections)} 个缺陷:")
    print(f"{'─' * 60}")

    for d in detections:
        b = d["bbox"]
        print(f"  [{d['class']:12s}]  置信度: {d['confidence']:.3f}  |  "
              f"bbox: ({b[0]}, {b[1]}) → ({b[2]}, {b[3]})")

    if not detections:
        print("  (无缺陷检出)")

    print(f"{'=' * 60}\n")

    # ---- 可视化 ----
    img_bgr = cv2.imread(image_path)
    colors = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
    ]

    import hashlib
    hash_str = hashlib.md5(image_path.encode()).hexdigest()[:8]

    for d in detections:
        b = d["bbox"]
        cls_idx = DEFECT_CLASSES.index(d["class"]) if d["class"] in DEFECT_CLASSES else 0
        color = colors[cls_idx % len(colors)]

        cv2.rectangle(img_bgr, (b[0], b[1]), (b[2], b[3]), color, 2)
        label = f"{d['class']} {d['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_bgr, (b[0], b[1] - th - 6), (b[0] + tw + 4, b[1]), color, -1)
        cv2.putText(img_bgr, label, (b[0] + 2, b[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(img_bgr, f"Time: {avg_ms:.1f}ms",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    out_path = f"output/inference_{hash_str}.jpg"
    os.makedirs("output", exist_ok=True)
    cv2.imwrite(out_path, img_bgr)
    print(f"结果图像保存至: {out_path}")

    return {"time_ms": round(avg_ms, 2), "detections": len(detections), "image": out_path}


def infer_onnx(image_path: str, onnx_path: str = None):
    """兼容旧接口"""
    return infer_image(image_path, onnx_path or ONNX_MODEL_PATH)


if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else ""
    if not img_path:
        # 默认用验证集第一张
        val_dir = "data/raw/val/images"
        imgs = [f for f in os.listdir(val_dir) if f.endswith(('.jpg', '.png'))]
        if imgs:
            img_path = os.path.join(val_dir, imgs[0])
    infer_image(img_path)
