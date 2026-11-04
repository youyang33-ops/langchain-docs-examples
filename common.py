"""所有章节共用的配置与小工具。

- 模型名称统一从 .env 读取，换供应商只需改一处。
- LC_MODEL=fake 时使用离线假模型：不花钱、不联网，用来检查代码能否跑通。
- run_demos() 让每个章节文件都可以这样运行：
      python ch02_models.py            # 列出本章所有 demo
      python ch02_models.py stream     # 只运行名为 demo_stream 的函数
      python ch02_models.py all        # 依次运行全部 demo

代码来源标注（每个类、函数和主要常量前面都有一行）：
  【文档原代码】          和文档的代码完全一致（只比较语法，不比较注释和排版）
  【文档原代码·有改动】   来自文档，做了改动（标注里写了改了什么）
  【补写】                文档没有给出这段代码，是我写的
来源写成"页面 · 小节"，页面相对于 https://docs.langchain.com/oss/python/langchain/
注意：overview 页只有 get_weather 这一个例子，其余代码来自它链接的其他页面。
"""

from __future__ import annotations

import inspect
import json
import os
import sys
from collections.abc import Callable
from itertools import cycle
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk

load_dotenv()

# 主模型：负责 agent 的推理与工具调用
# 【补写】主模型，从 .env 读取
MODEL = os.getenv("LC_MODEL", "openai:gpt-5.5")
# 小模型：用于摘要、安全审查、工具筛选等辅助任务
# 【补写】小模型，从 .env 读取
SMALL_MODEL = os.getenv("LC_SMALL_MODEL", "openai:gpt-5.4-mini")
# 备用模型：ModelFallbackMiddleware 演示用
# 【补写】备用模型，从 .env 读取
FALLBACK_MODEL = os.getenv("LC_FALLBACK_MODEL", SMALL_MODEL)

FAKE = MODEL == "fake"


# 【文档原代码·有改动】test/unit-testing · Test a tool-calling loop：多了 _stream 方法（支持流式）
class FakeToolCallingModel(GenericFakeChatModel):
    """离线假模型（来自 Unit testing 文档）：bind_tools 直接返回自身。"""

    # 【文档原代码】test/unit-testing · Test a tool-calling loop
    def bind_tools(self, tools, **kwargs):
        return self

    # 【补写】文档的假模型流式时会丢掉 tool_calls
    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        # 父类的流式实现会丢掉 tool_calls；这里把整条消息作为一个 chunk 输出
        message = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs).generations[0].message
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=message.content,
                id=message.id,
                tool_call_chunks=[
                    {"name": tc["name"], "args": json.dumps(tc["args"]), "id": tc["id"], "index": i}
                    for i, tc in enumerate(message.tool_calls)
                ],
                chunk_position="last",
            )
        )


# 【补写】构造循环回复的假模型
def fake_model(replies: list[Any] | None = None) -> FakeToolCallingModel:
    """不传 replies 时，永远回复同一句话，足够让大部分 demo 跑完一轮。"""
    replies = replies or [AIMessage("（假模型）这是一条离线回复。")]
    # 每次都复制一份：同一个消息对象（同一个 id）重复出现时，LangGraph 会把它当成"更新"而不是"追加"
    return FakeToolCallingModel(
        messages=(r.model_copy() if isinstance(r, BaseMessage) else r for r in cycle(replies))
    )


# 【补写】统一的模型入口
def get_model(name: str | None = None, **kwargs: Any) -> BaseChatModel:
    """统一的模型入口。name 省略时用主模型。"""
    if FAKE:
        return fake_model()
    return init_chat_model(name or MODEL, **kwargs)


# 【补写】统一的小模型入口
def get_small_model(**kwargs: Any) -> BaseChatModel:
    return get_model(SMALL_MODEL, **kwargs)


# 【补写】打印标题
def title(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


# 【补写】取最后一条消息的文本
def last_text(result: dict) -> str:
    """取 agent 结果里最后一条消息的文本。"""
    return result["messages"][-1].text


# 【补写】命令行运行 demo
def run_demos(namespace: dict[str, Any]) -> None:
    """按命令行参数运行当前模块里的 demo_* 函数。"""
    demos: dict[str, Callable[[], Any]] = {
        name.removeprefix("demo_"): fn
        for name, fn in namespace.items()
        if name.startswith("demo_") and inspect.isfunction(fn)
    }
    args = sys.argv[1:]
    if not args:
        module_doc = (namespace.get("__doc__") or "").strip().splitlines()
        print(module_doc[0] if module_doc else "")
        print(f"\n当前模型: {MODEL}   小模型: {SMALL_MODEL}\n")
        print("可运行的 demo（python <文件> <名字>，或 all）：")
        for name, fn in demos.items():
            summary = (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""
            print(f"  {name:<28} {summary}")
        return
    selected = list(demos) if args == ["all"] else args
    for name in selected:
        if name not in demos:
            print(f"没有名为 {name!r} 的 demo。")
            continue
        title(f"{name}: {(demos[name].__doc__ or '').strip().splitlines()[0]}")
        result = demos[name]()
        if inspect.iscoroutine(result):
            import asyncio

            asyncio.run(result)
