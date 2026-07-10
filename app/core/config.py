"""企业级配置管理 — Pydantic Settings + .env"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ---- 应用 ----
    APP_NAME: str = "工业表面缺陷检测系统"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False

    # ---- 模型 ----
    MODEL_PATH: str = "models/defect_detector.onnx"
    CONFIDENCE_THRESHOLD: float = 0.35
    IOU_THRESHOLD: float = 0.45
    MODEL_INPUT_SIZE: int = 160

    # ---- 缺陷类别 ----
    DEFECT_CLASSES: list[str] = [
        "crack", "scratch", "pit",
        "inclusion", "oxidation", "stain",
    ]

    DEFECT_CLASSES_ZH: dict[str, str] = {
        "crack": "裂纹", "scratch": "划痕", "pit": "凹坑",
        "inclusion": "夹杂", "oxidation": "氧化皮", "stain": "斑痕",
    }

    # ---- 服务 ----
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    # ---- 推理 ----
    INFERENCE_WARMUP_RUNS: int = 3
    INFERENCE_BENCHMARK_RUNS: int = 10

    # ---- 路径 ----
    @property
    def BASE_DIR(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent

    @property
    def MODEL_PATH_ABS(self) -> Path:
        return self.BASE_DIR / self.MODEL_PATH


settings = Settings()
