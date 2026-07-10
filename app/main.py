"""FastAPI 应用入口"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时预热模型，关闭时清理"""
    setup_logging("DEBUG" if settings.DEBUG else "INFO")
    logger.info(f"启动 {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"端口: {settings.PORT}, 调试: {settings.DEBUG}")

    # 预热检测器
    from app.services.detector_service import get_detector
    get_detector()
    logger.info("模型预热完成，服务就绪")

    yield

    logger.info("服务关闭")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由
app.include_router(router)

# 静态文件（Web 上传界面）
from pathlib import Path
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
