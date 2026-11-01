"""第 8 步 · 语义检索：按意思而不是按关键词找笔记（RAG）

目标：给 store 配上向量索引，新增 search_notes 工具；agent 回答前先查自己的笔记。
新增知识点：Embeddings、InMemoryStore(index=...)、store.search(query=...)、Agentic RAG
                                                                         （对应 ch14、ch17）
运行：python progressive/step08_semantic_search.py
试试：
    > 记笔记：checkpointer 负责短期记忆，按 thread_id 保存
    > 记笔记：store 负责长期记忆，按 namespace 保存
    > 怎么让 agent 记住上一轮对话？     （关键词不同，但能检索到第一条）
向量模型由 .env 的 LC_EMBEDDINGS 决定；LC_MODEL=fake 时用假向量（能跑，但检索结果随机）。
"""

import os  # ★ 新增
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.agents import create_agent  # noqa: E402
from langchain.agents.middleware import ModelRequest, dynamic_prompt  # noqa: E402
from langchain.embeddings import init_embeddings  # noqa: E402  ★ 新增
from langchain.tools import ToolRuntime, tool  # noqa: E402
from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402  ★ 新增
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402

from common import FAKE, get_model  # noqa: E402  ★ 修改：多了 FAKE

BASE_PROMPT = (
    "You are a friendly study assistant for someone learning LangChain. Reply in the user's language.\n"
    "Use save_note when the user asks you to remember something, and list_notes to show notes.\n"
    "Before answering a question about LangChain, call search_notes to check the user's own notes and cite them.\n"  # ★ 新增
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
    runtime.store.put(notes_ns(runtime), key, {"text": text})  # 写入时自动计算 text 字段的向量
    return f"Saved note {key}."


@tool
def list_notes(runtime: ToolRuntime[UserContext]) -> str:
    """List all of the user's notes with their IDs."""
    items = runtime.store.search(notes_ns(runtime), limit=50)
    return "\n".join(f"[{i.key}] {i.value['text']}" for i in items) or "No notes yet."


@tool
def search_notes(query: str, runtime: ToolRuntime[UserContext]) -> str:  # ★ 新增
    """Semantic search over the user's saved notes. Returns the most relevant notes."""
    hits = runtime.store.search(notes_ns(runtime), query=query, limit=3)
    if not hits:
        return "No notes yet."
    return "\n".join(f"[{h.key}] (score {h.score:.2f}) {h.value['text']}" for h in hits)


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


def get_embeddings():  # ★ 新增
    if FAKE:
        return DeterministicFakeEmbedding(size=256)
    return init_embeddings(os.getenv("LC_EMBEDDINGS", "openai:text-embedding-3-small"))


def make_store() -> InMemoryStore:  # ★ 新增：带向量索引的 store
    embeddings = get_embeddings()
    dims = len(embeddings.embed_query("dimension probe"))
    # fields=["text"]：只对 value["text"] 建向量索引（profile 没有 text 字段，不会被索引）
    return InMemoryStore(index={"embed": embeddings, "dims": dims, "fields": ["text"]})


def build_agent(store: InMemoryStore):
    return create_agent(
        model=get_model(),
        tools=[save_note, list_notes, search_notes, update_profile],  # ★ 修改：加 search_notes
        checkpointer=InMemorySaver(),
        context_schema=UserContext,
        store=store,
        middleware=[personalized_prompt],
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
    store = make_store()  # ★ 修改
    agent = build_agent(store)
    config = new_config()
    context = UserContext(user_id="local-user")
    print("学习助手 v8（/memory 查看长期记忆，/user <名字>，/new，/quit）")
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
