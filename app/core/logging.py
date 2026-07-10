"""Loguru 结构化日志配置"""

import sys
from loguru import logger


def setup_logging(level: str = "INFO"):
    """配置全局日志：控制台彩色 + 文件轮转"""
    logger.remove()

    # 控制台：彩色格式
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level:<7}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "{message}"
        ),
        level=level,
        colorize=True,
    )

    # 文件：JSON 结构化，便于日志采集
    logger.add(
        "logs/app_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
    )

    return logger


# 模块级 logger 实例
log = logger
