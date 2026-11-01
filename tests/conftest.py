"""pytest 共用配置：默认用离线假模型，让单元测试不花钱、不联网。"""

import os
import sys
from pathlib import Path

# 必须在导入任何章节模块之前设置
os.environ.setdefault("LC_MODEL", "fake")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: 需要真实模型和 API key 的测试")
