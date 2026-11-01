"""第 14 步 · 持久化：关掉程序再打开，记忆还在

目标：把 InMemorySaver / InMemoryStore 换成 SQLite 版本，数据写进 progressive/data/ 下的文件。
新增知识点：SqliteSaver（短期记忆持久化）、SqliteStore（长期记忆 + 向量索引持久化）
                                                                         （对应 ch06、ch14 的"生产环境"部分）
安装：pip install langgraph-checkpoint-sqlite
运行：python progressive/step14_persistence.py
试试：记一条笔记 → /quit → 重新运行 → 问"我刚才说了什么？"和"我有哪些笔记？"
换成 PostgresSaver / PostgresStore 就是真正的生产部署；agent 的代码一行都不用改。
"""

import os
import sys
import time
import uuid
from collections.abc import Callable
from typing import Any
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.agents import create_agent  # noqa: E402
from langchain.agents.middleware import (  # noqa: E402
    HumanInTheLoopMiddleware,
    ModelRequest,
    ModelRetryMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    before_agent,
    dynamic_prompt,
    wrap_tool_call,
)
from langchain.messages import ToolMessage  # noqa: E402
from langchain.embeddings import init_embeddings  # noqa: E402
from langchain.tools import ToolRuntime, tool  # noqa: E402
from langchain.tools.tool_node import ToolCallRequest  # noqa: E402
from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402  ★ 修改：InMemorySaver → SqliteSaver
from langgraph.store.sqlite import SqliteStore  # noqa: E402  ★ 修改：InMemoryStore → SqliteStore
from langgraph.runtime import Runtime  # noqa: E402
from langgraph.types import Command  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from common import FAKE, get_model, get_small_model  # noqa: E402

DATA_DIR = Path(__file__).with_name("data")  # ★ 新增
LAST_THREAD_FILE = DATA_DIR / "last_thread.txt"  # ★ 新增

BASE_PROMPT = (
    "You are a friendly study assistant for someone learning LangChain. Reply in the user's language.\n"
    "Use save_note when the user asks you to remember something, and list_notes to show notes.\n"
    "Before answering a question about LangChain, call search_notes to check the user's own notes and cite them.\n"
    "Use delete_note (with the ID from list_notes) when the user wants to remove a note.\n"
    "Use research to get an explanation of a concept you are unsure about.\n"
    "Use make_quiz when the user asks for a quiz or practice questions.\n"
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
def delete_note(note_id: str, runtime: ToolRuntime[UserContext]) -> str:
    """Delete a note by its ID. Requires human approval."""
    if runtime.store.get(notes_ns(runtime), note_id) is None:
        return f"No note with ID {note_id}."
    runtime.store.delete(notes_ns(runtime), note_id)
    return f"Deleted note {note_id}."


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


class QuizQuestion(BaseModel):
    question: str
    choices: list[str] = Field(description="Exactly 4 answer choices, without letter prefixes")
    answer: Literal["A", "B", "C", "D"]
    explanation: str = Field(description="One sentence explaining the correct answer")


class Quiz(BaseModel):
    topic: str
    questions: list[QuizQuestion]


@tool
def make_quiz(topic: str, n: int = 3) -> str:
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


BLOCKED_KEYWORDS = ["malware", "exploit", "ddos"]


@before_agent(can_jump_to=["end"])
def block_harmful_requests(state, runtime: Runtime) -> dict[str, Any] | None:
    text = state["messages"][-1].text.lower()
    if any(word in text for word in BLOCKED_KEYWORDS):
        return {
            "messages": [{"role": "assistant", "content": "这个请求我不能帮忙。我们继续聊 LangChain 吧！"}],
            "jump_to": "end",
        }
    return None


@wrap_tool_call
def time_tool_calls(request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]):
    start = time.perf_counter()
    result = handler(request)
    print(f"\n  ⏱  {request.tool_call['name']} 用时 {(time.perf_counter() - start) * 1000:.0f} ms", end="")
    return result


def get_embeddings():
    if FAKE:
        return DeterministicFakeEmbedding(size=256)
    return init_embeddings(os.getenv("LC_EMBEDDINGS", "openai:text-embedding-3-small"))


def index_config() -> dict:  # ★ 修改：原来的 make_store 只保留索引配置，store 在 main 里打开
    embeddings = get_embeddings()
    dims = len(embeddings.embed_query("dimension probe"))
    return {"embed": embeddings, "dims": dims, "fields": ["text"]}


def make_research_tool():
    researcher = create_agent(
        get_small_model(),
        tools=[],
        system_prompt=(
            "You are a concise research assistant for LangChain learners. "
            "Explain the topic in 3-5 bullet points with one tiny code example if useful."
        ),
        name="researcher",  # 流式输出时可以通过名字区分是哪个 agent 在说话
    )

    @tool("research", description="Delegate a question to the researcher subagent and get its explanation.")
    def research(question: str) -> str:
        result = researcher.invoke({"messages": [{"role": "user", "content": question}]})
        return result["messages"][-1].text  # 只把最终结论交回主 agent

    return research


def build_agent(store: SqliteStore, checkpointer: SqliteSaver):  # ★ 修改
    return create_agent(
        model=get_model(),
        tools=[save_note, list_notes, search_notes, delete_note, update_profile, make_quiz, make_research_tool()],
        checkpointer=checkpointer,  # ★ 修改
        context_schema=UserContext,
        store=store,
        name="study_assistant",
        middleware=[
            block_harmful_requests,
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            personalized_prompt,
            SummarizationMiddleware(model=get_small_model(), trigger=("tokens", 6000), keep=("messages", 20)),
            ModelRetryMiddleware(max_retries=2),
            ToolCallLimitMiddleware(run_limit=8),
            HumanInTheLoopMiddleware(interrupt_on={"delete_note": {"allowed_decisions": ["approve", "reject"]}}),
            time_tool_calls,
        ],
    )


def new_config() -> dict:  # ★ 修改：记下最新的 thread_id，下次启动接着用
    thread_id = uuid.uuid4().hex
    LAST_THREAD_FILE.write_text(thread_id)
    return {"configurable": {"thread_id": thread_id}}


def resume_config() -> dict:  # ★ 新增
    if LAST_THREAD_FILE.exists():
        return {"configurable": {"thread_id": LAST_THREAD_FILE.read_text().strip()}}
    return new_config()


def chat(agent, text: str, config: dict, context: UserContext) -> None:
    payload = {"messages": [{"role": "user", "content": text}]}
    while True:
        stream = agent.stream_events(payload, config, context=context, version="v3")
        print("助手: ", end="", flush=True)
        printed = False
        for kind, item in stream.interleave("messages", "tool_calls"):
            if kind == "messages":
                for token in item.text:
                    printed = True
                    print(token, end="", flush=True)
            elif kind == "tool_calls":
                for _ in item.output_deltas:  # 读完 output_deltas（工具执行完）后 item.output 才有值
                    pass
                print(f"\n  🔧 {item.tool_name}({item.input}) → {item.output}")
        if not printed:  # 被护栏拦截时回复不是模型生成的，流里没有 token
            print(agent.get_state(config).values["messages"][-1].text, end="")
        print()
        if not stream.interrupted:  # 正常结束
            return
        # 暂停了：一个 interrupt 里可能有多个待审批的调用，按顺序给出决策
        decisions = []
        for request in stream.interrupts[0].value["action_requests"]:
            answer = input(f"⚠️  {request['description']}\n   批准吗？[y/n] ").strip().lower()
            decisions.append({"type": "approve"} if answer == "y" else {"type": "reject", "message": "The user declined."})
        payload = Command(resume={"decisions": decisions})  # 用 Command 继续，而不是发新消息


def show_memory(store, user_id: str) -> None:
    for category in ("profile", "notes"):
        for item in store.search((user_id, category), limit=50):
            print(f"  {category:<8} {item.key}: {item.value}")


def main() -> None:  # ★ 修改：在 with 里打开两个数据库文件，退出时自动关闭
    DATA_DIR.mkdir(exist_ok=True)
    with (
        SqliteSaver.from_conn_string(str(DATA_DIR / "checkpoints.db")) as checkpointer,
        SqliteStore.from_conn_string(str(DATA_DIR / "store.db"), index=index_config()) as store,
    ):
        checkpointer.setup()  # 第一次运行时建表
        store.setup()
        agent = build_agent(store, checkpointer)
        config = resume_config()  # 接着上次的对话
        context = UserContext(user_id="local-user")
        print("学习助手 v14（/memory 查看长期记忆，/user <名字>，/new，/quit）")
        print(f"（数据保存在 {DATA_DIR}；继续对话 {config['configurable']['thread_id'][:8]}…）")
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
