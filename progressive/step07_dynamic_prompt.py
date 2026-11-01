"""第 7 步 · 动态提示词：根据长期记忆个性化回答

目标：每次调用模型前，从 store 读出用户资料，拼进系统提示词。
新增知识点：@dynamic_prompt、ModelRequest、request.runtime.store       （对应 ch09、ch11）
运行：python progressive/step07_dynamic_prompt.py
试试：
    > 我叫小明，我喜欢非常简洁的回答
    > /new
    > 介绍一下 LangChain        （新对话里它依然叫你小明，并且回答很简短）
第 6 步里资料虽然存下来了，但模型看不到；这一步让它"用上"长期记忆。
"""

import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.agents import create_agent  # noqa: E402
from langchain.agents.middleware import ModelRequest, dynamic_prompt  # noqa: E402  ★ 新增
from langchain.tools import ToolRuntime, tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402

from common import get_model  # noqa: E402

BASE_PROMPT = (  # ★ 修改：改名为 BASE_PROMPT，由 personalized_prompt 拼接使用
    "You are a friendly study assistant for someone learning LangChain. Reply in the user's language.\n"
    "Use save_note when the user asks you to remember something, and list_notes to show notes.\n"
    "When the user tells you their name or how they like answers, call update_profile."
)


@dataclass
class UserContext:
    user_id: str


def notes_ns(runtime) -> tuple[str, str]:
    return (runtime.context.user_id, "notes")


def profile_ns(runtime) -> tuple[str, str]:
    return (runtime.context.user_id, "profile")


@tool
def save_note(text: str, runtime: ToolRuntime[UserContext]) -> str:
    """Save a study note for the user."""
    key = uuid.uuid4().hex[:8]
    runtime.store.put(notes_ns(runtime), key, {"text": text})
    return f"Saved note {key}."


@tool
def list_notes(runtime: ToolRuntime[UserContext]) -> str:
    """List all of the user's notes with their IDs."""
    items = runtime.store.search(notes_ns(runtime), limit=50)
    return "\n".join(f"[{i.key}] {i.value['text']}" for i in items) or "No notes yet."


@tool
def update_profile(
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


# ★ 新增：中间件。每次调用模型前运行，返回值就是这次用的系统提示词
@dynamic_prompt
def personalized_prompt(request: ModelRequest) -> str:
    profile = request.runtime.store.get(profile_ns(request.runtime), "me")
    p = profile.value if profile else {}
    prompt = BASE_PROMPT
    if p.get("name"):
        prompt += f"\nThe user's name is {p['name']}."
    if p.get("answer_style"):
        prompt += f"\nThe user prefers {p['answer_style']} answers."
    return prompt


def build_agent(store: InMemoryStore):
    return create_agent(
        model=get_model(),
        tools=[save_note, list_notes, update_profile],
        checkpointer=InMemorySaver(),
        context_schema=UserContext,
        store=store,
        middleware=[personalized_prompt],  # ★ 修改：system_prompt=... 换成了中间件
    )


def new_config() -> dict:
    return {"configurable": {"thread_id": uuid.uuid4().hex}}


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


def show_memory(store, user_id: str) -> None:
    for category in ("profile", "notes"):
        for item in store.search((user_id, category), limit=50):
            print(f"  {category:<8} {item.key}: {item.value}")


def main() -> None:
    store = InMemoryStore()
    agent = build_agent(store)
    config = new_config()
    context = UserContext(user_id="local-user")
    print("学习助手 v7（/memory 查看长期记忆，/user <名字>，/new，/quit）")
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
        if text == "/memory":
            show_memory(store, context.user_id)
            continue
        if text.startswith("/user "):
            context = UserContext(user_id=text.removeprefix("/user ").strip())
            config = new_config()
            print(f"（已切换到用户 {context.user_id}）")
            continue
        if text:
            chat(agent, text, config, context)


if __name__ == "__main__":
    main()
