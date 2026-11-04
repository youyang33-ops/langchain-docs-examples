"""第 5 步 · 运行时 context：区分不同用户

目标：每次调用传入"当前是谁"，工具按用户分别存取笔记。
新增知识点：context_schema、ToolRuntime[Context]、invoke 时传 context=     （对应 ch04、ch09）
运行：python progressive/step05_runtime_context.py
试试：记一条笔记 → /user bob → 问"我有哪些笔记？"（bob 没有）→ /user local-user（又有了）
关键点：user_id 不是模型告诉工具的，而是程序通过 context 注入的——模型无法冒充别人。

代码来源标注（每个类、函数和主要常量前面都有一行）：
  【文档原代码】   和文档的代码完全一致（只比较语法，不比较注释和排版）
  【补写】         文档没有给出这段代码，是我写的
这个文件是为学习项目写的，不是文档原有的内容；"参考"表示只借用了文档的写法。
"""

import sys
import uuid
from dataclasses import dataclass  # ★ 新增
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.agents import create_agent  # noqa: E402
from langchain.tools import ToolRuntime, tool  # noqa: E402  ★ 修改：多了 ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from common import get_model  # noqa: E402

# 【补写】学习助手的提示词
SYSTEM_PROMPT = (
    "You are a friendly study assistant for someone learning LangChain. Reply in the user's language.\n"
    "Use save_note when the user asks you to remember something, and list_notes to show notes."
)


# 【文档原代码】tools · Context
@dataclass
class UserContext:  # ★ 新增：每次调用都会传入的只读信息
    user_id: str


NOTES: dict[str, list[str]] = {}  # ★ 修改：按用户分开存


# 【补写】学习助手的工具（ToolRuntime 写法参考 tools 页）
@tool
def save_note(text: str, runtime: ToolRuntime[UserContext]) -> str:  # ★ 修改
    """Save a study note for the user."""
    # runtime 参数对模型不可见，LangChain 会自动注入
    notes = NOTES.setdefault(runtime.context.user_id, [])
    notes.append(text)
    return f"Saved note #{len(notes)}."


# 【补写】学习助手的工具
@tool
def list_notes(runtime: ToolRuntime[UserContext]) -> str:  # ★ 修改
    """List all of the user's notes."""
    notes = NOTES.get(runtime.context.user_id, [])
    return "\n".join(f"#{i} {n}" for i, n in enumerate(notes, 1)) or "No notes yet."


# 【补写】组装学习助手
def build_agent():
    return create_agent(
        model=get_model(),
        tools=[save_note, list_notes],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
        context_schema=UserContext,  # ★ 新增
    )


# 【补写】生成新的 thread_id
def new_config() -> dict:
    return {"configurable": {"thread_id": uuid.uuid4().hex}}


# 【补写】聊天一轮
def chat(agent, text: str, config: dict, context: UserContext) -> None:  # ★ 修改：多了 context
    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": text}]}, config, context=context, version="v3"
    )
    print("助手: ", end="", flush=True)
    for kind, item in stream.interleave("messages", "tool_calls"):
        if kind == "messages":
            for token in item.text:
                print(token, end="", flush=True)
        elif kind == "tool_calls":
            for _ in item.output_deltas:  # 读完 output_deltas（工具执行完）后 item.output 才有值
                pass
            print(f"\n  🔧 {item.tool_name}({item.input}) → {item.output}")
    print()


# 【补写】命令行主循环
def main() -> None:
    agent = build_agent()
    config = new_config()
    context = UserContext(user_id="local-user")  # ★ 新增
    print("学习助手 v5（/user <名字> 切换用户，/new 开新对话，/quit 退出）")
    while True:
        try:
            text = input(f"\n{context.user_id}: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text == "/quit":
            break
        if text == "/new":
            config = new_config()
            print("（已开启新对话）")
            continue
        if text.startswith("/user "):  # ★ 新增：换用户时也换一段新对话
            context = UserContext(user_id=text.removeprefix("/user ").strip())
            config = new_config()
            print(f"（已切换到用户 {context.user_id}）")
            continue
        if text:
            chat(agent, text, config, context)


# 【补写】命令行入口
if __name__ == "__main__":
    main()
