"""第 6 步 · 长期记忆：跨对话记住笔记和用户资料

目标：把笔记从 Python 字典搬进 LangGraph Store，并新增"用户资料"（名字、回答风格）。
新增知识点：InMemoryStore、namespace、runtime.store.put / get / search      （对应 ch14）
运行：python progressive/step06_long_term_memory.py
试试：
    > 我叫小明，我喜欢简洁的回答        （update_profile 写入 store）
    > /new                            （新对话：短期记忆清空）
    > /memory                         （长期记忆还在）
区别：checkpointer 按 thread_id 存"这段对话"；store 按 namespace 存"关于这个用户的事"。

代码来源标注（每个类、函数和主要常量前面都有一行）：
  【文档原代码】   和文档的代码完全一致（只比较语法，不比较注释和排版）
  【补写】         文档没有给出这段代码，是我写的
这个文件是为学习项目写的，不是文档原有的内容；"参考"表示只借用了文档的写法。
"""

import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.agents import create_agent  # noqa: E402
from langchain.tools import ToolRuntime, tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402  ★ 新增

from common import get_model  # noqa: E402

# 【补写】学习助手的提示词
SYSTEM_PROMPT = (
    "You are a friendly study assistant for someone learning LangChain. Reply in the user's language.\n"
    "Use save_note when the user asks you to remember something, and list_notes to show notes.\n"
    "When the user tells you their name or how they like answers, call update_profile."  # ★ 新增
)


# 【文档原代码】tools · Context
@dataclass
class UserContext:
    user_id: str


# ★ 新增：namespace 像文件夹路径，按 (用户, 类别) 隔离数据
# 【补写】笔记的 namespace
def notes_ns(runtime) -> tuple[str, str]:
    return (runtime.context.user_id, "notes")


# 【补写】用户资料的 namespace
def profile_ns(runtime) -> tuple[str, str]:
    return (runtime.context.user_id, "profile")


# 【补写】学习助手的工具（ToolRuntime 写法参考 tools 页）
@tool
def save_note(text: str, runtime: ToolRuntime[UserContext]) -> str:  # ★ 修改：写入 store
    """Save a study note for the user."""
    key = uuid.uuid4().hex[:8]
    runtime.store.put(notes_ns(runtime), key, {"text": text})
    return f"Saved note {key}."


# 【补写】学习助手的工具
@tool
def list_notes(runtime: ToolRuntime[UserContext]) -> str:  # ★ 修改：从 store 读取
    """List all of the user's notes with their IDs."""
    items = runtime.store.search(notes_ns(runtime), limit=50)
    return "\n".join(f"[{i.key}] {i.value['text']}" for i in items) or "No notes yet."


# 【补写】学习助手的工具（读写 store 写法参考 context-engineering · Writes）
@tool
def update_profile(  # ★ 新增
    name: str | None = None, answer_style: str | None = None, *, runtime: ToolRuntime[UserContext]
) -> str:
    """Remember the user's name and/or preferred answer style (e.g. 'concise', 'detailed')."""
    existing = runtime.store.get(profile_ns(runtime), "me")
    profile = dict(existing.value) if existing else {}
    if name:
        profile["name"] = name
    if answer_style:
        profile["answer_style"] = answer_style
    runtime.store.put(profile_ns(runtime), "me", profile)
    return f"Profile updated: {profile}"


# 【补写】组装学习助手
def build_agent(store: InMemoryStore):  # ★ 修改：store 从外面传入，main 里还要用它
    return create_agent(
        model=get_model(),
        tools=[save_note, list_notes, update_profile],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
        context_schema=UserContext,
        store=store,  # ★ 新增
    )


# 【补写】生成新的 thread_id
def new_config() -> dict:
    return {"configurable": {"thread_id": uuid.uuid4().hex}}


# 【补写】聊天一轮
def chat(agent, text: str, config: dict, context: UserContext) -> None:
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


# 【补写】打印长期记忆
def show_memory(store, user_id: str) -> None:  # ★ 新增
    for category in ("profile", "notes"):
        for item in store.search((user_id, category), limit=50):
            print(f"  {category:<8} {item.key}: {item.value}")


# 【补写】命令行主循环
def main() -> None:
    store = InMemoryStore()  # ★ 新增
    agent = build_agent(store)
    config = new_config()
    context = UserContext(user_id="local-user")
    print("学习助手 v6（/memory 查看长期记忆，/user <名字>，/new，/quit）")
    while True:
        try:
            text = input(f"\n{context.user_id}: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text == "/quit":
            break
        if text == "/new":
            config = new_config()
            print("（已开启新对话：短期记忆清空，长期记忆保留）")
            continue
        if text == "/memory":  # ★ 新增
            show_memory(store, context.user_id)
            continue
        if text.startswith("/user "):
            context = UserContext(user_id=text.removeprefix("/user ").strip())
            config = new_config()
            print(f"（已切换到用户 {context.user_id}）")
            continue
        if text:
            chat(agent, text, config, context)


# 【补写】命令行入口
if __name__ == "__main__":
    main()
