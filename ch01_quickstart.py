"""第 1 章 · Overview + Quickstart：第一个 agent，以及一个"真实"的 agent。

文档：
- https://docs.langchain.com/oss/python/langchain/overview
- https://docs.langchain.com/oss/python/langchain/quickstart

核心公式：Agent = Model + Harness（提示词 + 工具 + 中间件）。
create_agent 就是 LangChain 提供的那个 harness。
"""

import urllib.error
import urllib.request

from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from common import get_model, last_text, run_demos


# ---------------------------------------------------------------------------
# Overview：最小可运行的 agent
# ---------------------------------------------------------------------------
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


def demo_basic_agent():
    """最小 agent：一个模型 + 一个普通函数工具"""
    agent = create_agent(
        model=get_model(),
        tools=[get_weather],  # 普通函数也能当工具，docstring 就是工具描述
        system_prompt="You are a helpful assistant",
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
    )
    # content_blocks 是跨供应商统一的内容格式（文本、推理、工具调用……）
    print(result["messages"][-1].content_blocks)


# ---------------------------------------------------------------------------
# Quickstart：真实场景的 agent（联网抓取文本 + 记忆）
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a literary data assistant.

## Capabilities

- `fetch_text_from_url`: loads document text from a URL into the conversation.
Do not guess line counts or positions—ground them in tool results from the saved file."""


@tool
def fetch_text_from_url(url: str) -> str:
    """Fetch the document from a URL."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.URLError as e:
        return f"Fetch failed: {e}"
    return raw.decode("utf-8", errors="replace")


def demo_real_world_agent():
    """真实 agent：调参后的模型 + 联网工具 + checkpointer 记忆"""
    # 文档里各供应商的参数都类似：temperature / timeout / max_tokens
    model = get_model(temperature=0.5, timeout=300, max_tokens=25000)
    checkpointer = InMemorySaver()  # 短期记忆，第 6 章详细讲

    agent = create_agent(
        model=model,
        tools=[fetch_text_from_url],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

    content = """Project Gutenberg hosts a full plain-text copy of F. Scott Fitzgerald's The Great Gatsby.
URL: https://www.gutenberg.org/files/64317/64317-0.txt

Answer as much as you can:

1) How many lines in the complete Gutenberg file contain the substring `Gatsby`.
2) The 1-based line number of the first line in the file that contains `Daisy`.
3) A two-sentence neutral synopsis.

If you cannot **verify** an exact answer with your tools, use `null` and explain why."""

    print("Running create_agent...", flush=True)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": content}]},
        config={"configurable": {"thread_id": "great-gatsby-lc"}},
    )
    print(last_text(result))
    # 文档还对比了 deepagents.create_deep_agent（自带文件系统、子 agent、上下文压缩），
    # 见 ch16_multi_agent.py 的 demo_deep_agent。


if __name__ == "__main__":
    run_demos(globals())
