"""第 9 步 · 结构化输出：生成格式固定的练习题

目标：新增 make_quiz 工具。它让小模型按 Pydantic schema 输出题目，程序再排版成文本。
新增知识点：Pydantic schema、model.with_structured_output、工具内部再调用模型   （对应 ch02、ch08）
运行：python progressive/step09_structured_output.py
试试：
    > 出 3 道关于 checkpointer 的选择题
为什么要结构化：拿到的是 Quiz 对象而不是一段文字，程序可以可靠地取出题目、选项和答案
（比如以后做成网页上的答题卡）。
"""

import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal  # ★ 新增

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.agents import create_agent  # noqa: E402
from langchain.agents.middleware import ModelRequest, dynamic_prompt  # noqa: E402
from langchain.embeddings import init_embeddings  # noqa: E402
from langchain.tools import ToolRuntime, tool  # noqa: E402
from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402  ★ 新增

from common import FAKE, get_model, get_small_model  # noqa: E402  ★ 修改

BASE_PROMPT = (
    "You are a friendly study assistant for someone learning LangChain. Reply in the user's language.\n"
    "Use save_note when the user asks you to remember something, and list_notes to show notes.\n"
    "Before answering a question about LangChain, call search_notes to check the user's own notes and cite them.\n"
    "Use make_quiz when the user asks for a quiz or practice questions.\n"  # ★ 新增
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
def search_notes(query: str, runtime: ToolRuntime[UserContext]) -> str:
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


# ★ 新增：用 Pydantic 描述想要的输出结构；Field 的 description 会告诉模型每个字段的含义
class QuizQuestion(BaseModel):
    question: str
    choices: list[str] = Field(description="Exactly 4 answer choices, without letter prefixes")
    answer: Literal["A", "B", "C", "D"]
    explanation: str = Field(description="One sentence explaining the correct answer")


class Quiz(BaseModel):
    topic: str
    questions: list[QuizQuestion]


@tool
def make_quiz(topic: str, n: int = 3) -> str:  # ★ 新增
    """Create a multiple-choice quiz about a topic."""
    if FAKE:  # 假模型无法产生结构化结果
        return f"(fake) a {n}-question quiz about {topic}"
    quiz: Quiz = get_small_model().with_structured_output(Quiz).invoke(
        f"Write {n} multiple-choice questions about {topic} for a LangChain learner."
    )
    lines = [f"Quiz: {quiz.topic}"]
    for i, q in enumerate(quiz.questions, 1):
        lines.append(f"{i}. {q.question}")
        lines += [f"   {letter}. {choice}" for letter, choice in zip("ABCD", q.choices)]
    lines.append("Answers: " + "; ".join(f"{i}-{q.answer} ({q.explanation})" for i, q in enumerate(quiz.questions, 1)))
    return "\n".join(lines)


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


def get_embeddings():
    if FAKE:
        return DeterministicFakeEmbedding(size=256)
    return init_embeddings(os.getenv("LC_EMBEDDINGS", "openai:text-embedding-3-small"))


def make_store() -> InMemoryStore:
    embeddings = get_embeddings()
    dims = len(embeddings.embed_query("dimension probe"))
    # fields=["text"]：只对 value["text"] 建向量索引（profile 没有 text 字段，不会被索引）
    return InMemoryStore(index={"embed": embeddings, "dims": dims, "fields": ["text"]})


def build_agent(store: InMemoryStore):
    return create_agent(
        model=get_model(),
        tools=[save_note, list_notes, search_notes, update_profile, make_quiz],  # ★ 修改
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
    store = make_store()
    agent = build_agent(store)
    config = new_config()
    context = UserContext(user_id="local-user")
    print("学习助手 v9（/memory 查看长期记忆，/user <名字>，/new，/quit）")
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
