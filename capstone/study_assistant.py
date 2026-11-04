"""综合项目 · 学习助手：把前 19 章的知识点串成一个能用的命令行应用。

运行：python capstone/study_assistant.py
可以试试：
    > 我叫小明，我喜欢简洁的回答                    （长期记忆：保存偏好）
    > 帮我记一条笔记：LangChain 的中间件分节点式和包裹式   （工具写 store）
    > 我的笔记里关于中间件说了什么？                  （RAG：语义检索笔记）
    > 出 3 道关于 LangChain agent 的选择题             （结构化输出子任务）
    > 研究一下 LangGraph 是什么                       （子 agent）
    > 删除第一条笔记                                  （人工审批）
命令：/new 开新对话   /memory 查看长期记忆   /quit 退出

用到的知识点（括号里是章节）：
    create_agent(1,5)  @tool + ToolRuntime(4)  context_schema(9)  checkpointer(6)
    store + 语义检索(14,17)  dynamic_prompt(9)  SummarizationMiddleware(6,10)
    ModelRetry / ToolCallLimit / PII(10)  自定义 wrap_tool_call(11)  护栏(12)
    HumanInTheLoop(13)  结构化输出(8)  子 agent(16)  stream_events v3(7)

代码来源标注（每个类、函数和主要常量前面都有一行）：
  【文档原代码】   和文档的代码完全一致（只比较语法，不比较注释和排版）
  【补写】         文档没有给出这段代码，是我写的
这个文件是为学习项目写的，不是文档原有的内容；"参考"表示只借用了文档的写法。
"""

import sys
import uuid
from collections.abc import Callable
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
from langchain.tools import ToolRuntime, tool  # noqa: E402
from langchain.tools.tool_node import ToolCallRequest  # noqa: E402
from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402
from langgraph.types import Command  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from common import FAKE, get_model, get_small_model  # noqa: E402


# ---------------------------------------------------------------------------
# Context 与 Store
# ---------------------------------------------------------------------------
# 【文档原代码】tools · Context
@dataclass
class UserContext:
    user_id: str


# 【补写】创建带向量索引的 store
def make_store() -> InMemoryStore:
    """笔记会被向量化，支持语义检索（第 14、17 章）。"""
    if FAKE:
        embeddings = DeterministicFakeEmbedding(size=256)
    else:
        from ch17_knowledge_base_rag import get_embeddings

        embeddings = get_embeddings()
    return InMemoryStore(index={"embed": embeddings, "dims": len(embeddings.embed_query("x")), "fields": ["text"]})


# 【补写】笔记的 namespace
def notes_ns(runtime) -> tuple[str, str]:
    return (runtime.context.user_id, "notes")


# 【补写】用户资料的 namespace
def profile_ns(runtime) -> tuple[str, str]:
    return (runtime.context.user_id, "profile")


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
# 【补写】学习助手的工具（ToolRuntime 写法参考 tools 页）
@tool
def save_note(text: str, runtime: ToolRuntime[UserContext]) -> str:
    """Save a study note for the user. Use this when the user asks you to remember or note something."""
    key = uuid.uuid4().hex[:8]
    runtime.store.put(notes_ns(runtime), key, {"text": text})
    return f"Saved note {key}."


# 【补写】学习助手的工具（store.search 写法参考 long-term-memory 页）
@tool
def search_notes(query: str, runtime: ToolRuntime[UserContext]) -> str:
    """Semantic search over the user's saved notes."""
    hits = runtime.store.search(notes_ns(runtime), query=query, limit=3)
    if not hits:
        return "No notes yet."
    return "\n".join(f"[{h.key}] {h.value['text']}" for h in hits)


# 【补写】学习助手的工具
@tool
def list_notes(runtime: ToolRuntime[UserContext]) -> str:
    """List all of the user's notes with their IDs."""
    items = runtime.store.search(notes_ns(runtime), limit=50)
    return "\n".join(f"[{i.key}] {i.value['text']}" for i in items) or "No notes yet."


# 【补写】学习助手的工具
@tool
def delete_note(note_id: str, runtime: ToolRuntime[UserContext]) -> str:
    """Delete a note by its ID. Requires human approval."""
    runtime.store.delete(notes_ns(runtime), note_id)
    return f"Deleted note {note_id}."


# 【补写】学习助手的工具（读写 store 写法参考 context-engineering · Writes）
@tool
def update_profile(name: str | None = None, answer_style: str | None = None, *, runtime: ToolRuntime[UserContext]) -> str:
    """Remember the user's name and/or preferred answer style (e.g. 'concise', 'detailed')."""
    existing = runtime.store.get(profile_ns(runtime), "me")
    profile = dict(existing.value) if existing else {}
    if name:
        profile["name"] = name
    if answer_style:
        profile["answer_style"] = answer_style
    runtime.store.put(profile_ns(runtime), "me", profile)
    return f"Profile updated: {profile}"


# 【补写】出题的 schema（写法参考 models · Structured output）
class QuizQuestion(BaseModel):
    question: str
    choices: list[str] = Field(description="Exactly 4 choices")
    answer: Literal["A", "B", "C", "D"]


# 【补写】出题的 schema
class Quiz(BaseModel):
    topic: str
    questions: list[QuizQuestion]


# 【补写】出题工具
@tool
def make_quiz(topic: str, n: int = 3) -> str:
    """Create a multiple-choice quiz about a topic."""
    if FAKE:
        return f"(fake) quiz about {topic}"
    quiz = get_small_model().with_structured_output(Quiz).invoke(f"Write {n} multiple-choice questions about {topic}.")
    lines = [f"Quiz: {quiz.topic}"]
    for i, q in enumerate(quiz.questions, 1):
        lines.append(f"{i}. {q.question}")
        lines += [f"   {'ABCD'[j]}. {c}" for j, c in enumerate(q.choices)]
    lines.append("Answers: " + ", ".join(f"{i}-{q.answer}" for i, q in enumerate(quiz.questions, 1)))
    return "\n".join(lines)


# 【补写】写法参考 multi-agent/subagents · Basic implementation
def make_research_tool():
    """子 agent（第 16 章）：专门负责"研究"，主 agent 只看到它的结论。"""
    researcher = create_agent(
        get_small_model(),
        tools=[],
        system_prompt="You are a concise research assistant. Explain the topic in 3-5 bullet points.",
        name="researcher",
    )

    # 【补写】子 agent 包装成工具
    @tool("research", description="Delegate a research question to the researcher subagent.")
    def research(question: str) -> str:
        result = researcher.invoke({"messages": [{"role": "user", "content": question}]})
        return result["messages"][-1].text

    return research


# ---------------------------------------------------------------------------
# 中间件
# ---------------------------------------------------------------------------
# 【补写】写法参考 context-engineering · System Prompt
@dynamic_prompt
def personalized_prompt(request: ModelRequest) -> str:
    """根据长期记忆里的用户资料生成系统提示词（第 9 章）。"""
    profile = request.runtime.store.get(profile_ns(request.runtime), "me")
    p = profile.value if profile else {}
    lines = [
        "You are a friendly study assistant for someone learning LangChain.",
        "Use tools to save, search, list and delete notes; use research for explanations; use make_quiz for quizzes.",
        "Reply in the same language the user writes in.",
    ]
    if p.get("name"):
        lines.append(f"The user's name is {p['name']}.")
    if p.get("answer_style"):
        lines.append(f"The user prefers {p['answer_style']} answers.")
    return "\n".join(lines)


# 【补写】写法参考 guardrails · Before agent guardrails
@before_agent(can_jump_to=["end"])
def block_off_topic(state, runtime):
    """简单的确定性护栏（第 12 章）。"""
    text = state["messages"][-1].text.lower()
    if any(word in text for word in ("exploit", "malware")):
        return {"messages": [{"role": "assistant", "content": "这个问题我帮不上忙，我们聊聊学习吧。"}], "jump_to": "end"}
    return None


# 【补写】写法参考 middleware/custom · Tool call monitoring
@wrap_tool_call
def log_tool_calls(request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]):
    """自定义包裹式中间件（第 11 章）：打印每次工具调用。"""
    print(f"\n  🔧 {request.tool_call['name']}({request.tool_call['args']})", flush=True)
    return handler(request)


# 【补写】组装学习助手
def build_agent(store: InMemoryStore):
    return create_agent(
        model=get_model(),
        tools=[save_note, search_notes, list_notes, delete_note, update_profile, make_quiz, make_research_tool()],
        context_schema=UserContext,
        checkpointer=InMemorySaver(),  # 想跨进程保存对话，换成 SqliteSaver / PostgresSaver
        store=store,
        middleware=[
            block_off_topic,
            personalized_prompt,
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            SummarizationMiddleware(model=get_small_model(), trigger=("tokens", 6000), keep=("messages", 20)),
            ModelRetryMiddleware(max_retries=2),
            ToolCallLimitMiddleware(run_limit=8),
            HumanInTheLoopMiddleware(interrupt_on={"delete_note": {"allowed_decisions": ["approve", "reject"]}}),
            log_tool_calls,
        ],
        name="study_assistant",
    )


# ---------------------------------------------------------------------------
# 命令行界面
# ---------------------------------------------------------------------------
# 【补写】流式聊天一轮，处理审批
def stream_turn(agent, payload, config, context) -> None:
    """流式打印一轮回复；遇到人工审批就询问用户，然后继续（第 7、13 章）。"""
    while True:
        stream = agent.stream_events(payload, config=config, context=context, version="v3")
        print("助手: ", end="", flush=True)
        for message in stream.messages:
            for token in message.text:
                print(token, end="", flush=True)
        print()
        if not stream.interrupted:
            return
        decisions = []
        for request in stream.interrupts[0].value["action_requests"]:
            ok = input(f"⚠️  {request['description']}\n   批准吗？[y/n] ").strip().lower() == "y"
            decisions.append({"type": "approve"} if ok else {"type": "reject", "message": "User declined."})
        payload = Command(resume={"decisions": decisions})


# 【补写】命令行主循环
def main() -> None:
    store = make_store()
    agent = build_agent(store)
    context = UserContext(user_id="local-user")
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    print(__doc__.split("用到的知识点")[0])
    while True:
        try:
            text = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/new":
            config = {"configurable": {"thread_id": uuid.uuid4().hex}}
            print("（已开启新对话：短期记忆清空，长期记忆保留）")
            continue
        if text == "/memory":
            for ns in (("local-user", "profile"), ("local-user", "notes")):
                for item in store.search(ns, limit=50):
                    print(ns[1], item.key, item.value)
            continue
        stream_turn(agent, {"messages": [{"role": "user", "content": text}]}, config, context)


# 【补写】命令行入口
if __name__ == "__main__":
    main()
