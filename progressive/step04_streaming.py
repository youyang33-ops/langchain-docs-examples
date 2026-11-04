"""第 4 步 · 流式输出：边生成边显示

目标：不再等整轮结束才打印，而是实时显示 token 和工具调用。
新增知识点：agent.stream_events(version="v3")、stream.interleave     （对应 ch07）
运行：python progressive/step04_streaming.py
试试：让它记一条长一点的笔记，观察工具调用和回复是怎么逐步出现的。

代码来源标注（每个类、函数和主要常量前面都有一行）：
  【文档原代码】   和文档的代码完全一致（只比较语法，不比较注释和排版）
  【补写】         文档没有给出这段代码，是我写的
这个文件是为学习项目写的，不是文档原有的内容；"参考"表示只借用了文档的写法。
"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.agents import create_agent  # noqa: E402
from langchain.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from common import get_model  # noqa: E402

# 【补写】学习助手的提示词
SYSTEM_PROMPT = (
    "You are a friendly study assistant for someone learning LangChain. Reply in the user's language.\n"
    "Use save_note when the user asks you to remember something, and list_notes to show notes."
)

NOTES: list[str] = []


# 【补写】学习助手的工具（ToolRuntime 写法参考 tools 页）
@tool
def save_note(text: str) -> str:
    """Save a study note for the user."""
    NOTES.append(text)
    return f"Saved note #{len(NOTES)}."


# 【补写】学习助手的工具
@tool
def list_notes() -> str:
    """List all of the user's notes."""
    return "\n".join(f"#{i} {n}" for i, n in enumerate(NOTES, 1)) or "No notes yet."


# 【补写】组装学习助手
def build_agent():
    return create_agent(
        model=get_model(),
        tools=[save_note, list_notes],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )


# 【补写】生成新的 thread_id
def new_config() -> dict:
    return {"configurable": {"thread_id": uuid.uuid4().hex}}


# 【补写】聊天一轮
def chat(agent, text: str, config: dict) -> None:  # ★ 修改：invoke → stream_events
    stream = agent.stream_events({"messages": [{"role": "user", "content": text}]}, config, version="v3")
    print("助手: ", end="", flush=True)
    # interleave 按发生顺序交错给出"模型输出的消息"和"工具调用"
    for kind, item in stream.interleave("messages", "tool_calls"):
        if kind == "messages":
            for token in item.text:  # 逐 token 打印
                print(token, end="", flush=True)
        elif kind == "tool_calls":
            for _ in item.output_deltas:  # ★ 新增：读完 output_deltas（工具执行完）后 item.output 才有值
                pass
            print(f"\n  🔧 {item.tool_name}({item.input}) → {item.output}")
    print()


# 【补写】命令行主循环
def main() -> None:
    agent = build_agent()
    config = new_config()
    print("学习助手 v4（/new 开新对话，/quit 退出）")
    while True:
        try:
            text = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text == "/quit":
            break
        if text == "/new":
            config = new_config()
            print("（已开启新对话）")
            continue
        if text:
            chat(agent, text, config)


# 【补写】命令行入口
if __name__ == "__main__":
    main()
