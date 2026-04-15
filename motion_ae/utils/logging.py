"""简单日志工具。"""
from __future__ import annotations

import logging
import sys


def get_logger(name: str = "motion_ae", level: int = logging.INFO) -> logging.Logger:
    """获取统一格式的 logger。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
