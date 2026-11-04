"""pytest 共用配置：默认用离线假模型，让单元测试不花钱、不联网。

代码来源标注（每个类、函数和主要常量前面都有一行）：
  【文档原代码】   和文档的代码完全一致（只比较语法，不比较注释和排版）
  【补写】         文档没有给出这段代码，是我写的
这个文件是为学习项目写的，不是文档原有的内容；"参考"表示只借用了文档的写法。
"""

import os
import sys
from pathlib import Path

# 必须在导入任何章节模块之前设置
os.environ.setdefault("LC_MODEL", "fake")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# 【补写】
def pytest_configure(config):
    config.addinivalue_line("markers", "integration: 需要真实模型和 API key 的测试")
