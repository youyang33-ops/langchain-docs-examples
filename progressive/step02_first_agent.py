"""第 2 步 · 变成 agent：让模型能调用工具

目标：用 create_agent 代替手写的模型调用，加两个记笔记的工具。
新增知识点：create_agent、@tool、agent 循环（模型 → 工具 → 模型）   （对应 ch01、ch04、ch05）
运行：python progressive/step02_first_agent.py
试试：
    > 帮我记一条笔记：agent = 模型 + 工具 + 提示词
    > 我有哪些笔记？
注意：这一步 agent 没有记忆——每次 invoke 都是全新的对话（第 3 步解决）。

代码来源标注（每个类、函数和主要常量前面都有一行）：
  【文档原代码】   和文档的代码完全一致（只比较语法，不比较注释和排版）
  【补写】         文档没有给出这段代码，是我写的
这个文件是为学习项目写的，不是文档原有的内容；"参考"表示只借用了文档的写法。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.agents import create_agent  # noqa: E402  ★ 新增
from langchain.tools import tool  # noqa: E402  ★ 新增

from common import get_model  # noqa: E402

# 【补写】学习助手的提示词
SYSTEM_PROMPT = (
    "You are a friendly study assistant for someone learning LangChain. Reply in the user's language.\n"
    "Use save_note when the user asks you to remember something, and list_notes to show notes."  # ★ 新增
)

NOTES: list[str] = []  # ★ 新增：先用一个 Python 列表存笔记（第 6 步换成 Store）


# ★ 新增：两个工具。docstring 会作为工具描述发给模型，决定模型何时调用它
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
def build_agent():  # ★ 新增
    return create_agent(model=get_model(), tools=[save_note, list_notes], system_prompt=SYSTEM_PROMPT)


# 【补写】聊天一轮
def chat(agent, text: str) -> None:  # ★ 修改：交给 agent 处理
    result = agent.invoke({"messages": [{"role": "user", "content": text}]})
    # 打印这一轮里发生的工具调用，看看 agent 循环做了什么
    for message in result["messages"]:
        for tc in getattr(message, "tool_calls", None) or []:
            print(f"  🔧 {tc['name']}({tc['args']})")
    print("助手:", result["messages"][-1].text)


# 【补写】命令行主循环
def main() -> None:
    agent = build_agent()
    print("学习助手 v2（输入 /quit 退出）")
    while True:
        try:
            text = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text == "/quit":
            break
        if text:
            chat(agent, text)


# 【补写】命令行入口
if __name__ == "__main__":
    main()
