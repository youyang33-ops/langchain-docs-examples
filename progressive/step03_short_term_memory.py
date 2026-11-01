"""第 3 步 · 短期记忆：agent 记得这次对话说过什么

目标：加 checkpointer，同一个 thread_id 的多次调用共享消息历史。
新增知识点：InMemorySaver、thread_id、/new 开新对话            （对应 ch06）
运行：python progressive/step03_short_term_memory.py
试试：说"我叫小明" → 问"我叫什么？"（记得）→ /new → 再问（忘了）
"""

import sys
import uuid  # ★ 新增
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.agents import create_agent  # noqa: E402
from langchain.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402  ★ 新增

from common import get_model  # noqa: E402

SYSTEM_PROMPT = (
    "You are a friendly study assistant for someone learning LangChain. Reply in the user's language.\n"
    "Use save_note when the user asks you to remember something, and list_notes to show notes."
)

NOTES: list[str] = []


@tool
def save_note(text: str) -> str:
    """Save a study note for the user."""
    NOTES.append(text)
    return f"Saved note #{len(NOTES)}."


@tool
def list_notes() -> str:
    """List all of the user's notes."""
    return "\n".join(f"#{i} {n}" for i, n in enumerate(NOTES, 1)) or "No notes yet."


def build_agent():
    return create_agent(
        model=get_model(),
        tools=[save_note, list_notes],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),  # ★ 新增：每一步自动保存 state
    )


def new_config() -> dict:  # ★ 新增：一个 thread_id 就是一段独立的对话
    return {"configurable": {"thread_id": uuid.uuid4().hex}}


def chat(agent, text: str, config: dict) -> None:  # ★ 修改：多了 config
    # 只需要传"新的"这条消息，之前的历史由 checkpointer 按 thread_id 取回
    result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config)
    for message in result["messages"][-4:]:
        for tc in getattr(message, "tool_calls", None) or []:
            print(f"  🔧 {tc['name']}({tc['args']})")
    print("助手:", result["messages"][-1].text)
    print(f"  （这段对话现在共有 {len(result['messages'])} 条消息）")


def main() -> None:
    agent = build_agent()
    config = new_config()
    print("学习助手 v3（/new 开新对话，/quit 退出）")
    while True:
        try:
            text = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text == "/quit":
            break
        if text == "/new":  # ★ 新增
            config = new_config()
            print("（已开启新对话）")
            continue
        if text:
            chat(agent, text, config)


if __name__ == "__main__":
    main()
