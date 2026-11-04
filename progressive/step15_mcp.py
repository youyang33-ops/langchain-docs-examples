"""第 15 步 · MCP：接入外部工具服务器（最终版）

目标：通过 MCP 协议接入一个独立运行的"LangChain 术语表"服务器（step15_mcp_server.py），
      它的工具和我们自己写的工具一起交给 agent。
新增知识点：MCPAdapter、stdio 传输、把整个程序改成 async                （对应 ch15）
运行：python progressive/step15_mcp.py
试试：
    > 查一下术语表里 checkpointer 的定义

为什么要改成 async：MCP 工具只有异步实现，所以 agent 必须用 astream_events 运行。随之要改的地方：
    - 在事件循环里访问 store 要用 aget / asearch（动态提示词、/memory）
    - 包裹式中间件（wrap_tool_call）要写成 async def，并 await handler(...)
    - 普通的同步工具不用改：LangChain 会把它们放到线程池里执行
    - input() 用 asyncio.to_thread 包一层，不阻塞事件循环
异步流没有 interleave，所以把显示工具调用的工作交给了 log_tool_calls 中间件。

代码来源标注（每个类、函数和主要常量前面都有一行）：
  【文档原代码】   和文档的代码完全一致（只比较语法，不比较注释和排版）
  【补写】         文档没有给出这段代码，是我写的
这个文件是为学习项目写的，不是文档原有的内容；"参考"表示只借用了文档的写法。
"""

import asyncio  # ★ 新增
import os
import sys
import time
import uuid
from collections.abc import Awaitable, Callable  # ★ 修改
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
from langchain.mcp import MCPAdapter  # noqa: E402  ★ 新增
from langchain.tools import ToolRuntime, tool  # noqa: E402
from langchain.tools.tool_node import ToolCallRequest  # noqa: E402
from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: E402  ★ 修改：换成异步版本
from langgraph.store.sqlite.aio import AsyncSqliteStore  # noqa: E402  ★ 修改：换成异步版本
from langgraph.runtime import Runtime  # noqa: E402
from langgraph.types import Command  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from common import FAKE, get_model, get_small_model  # noqa: E402

DATA_DIR = Path(__file__).with_name("data")
LAST_THREAD_FILE = DATA_DIR / "last_thread.txt"
MCP_SERVER = Path(__file__).with_name("step15_mcp_server.py")  # ★ 新增

# 【补写】学习助手的提示词
BASE_PROMPT = (
    "You are a friendly study assistant for someone learning LangChain. Reply in the user's language.\n"
    "Use save_note when the user asks you to remember something, and list_notes to show notes.\n"
    "Before answering a question about LangChain, call search_notes to check the user's own notes and cite them.\n"
    "Use delete_note (with the ID from list_notes) when the user wants to remove a note.\n"
    "Use research to get an explanation of a concept you are unsure about.\n"
    "Use make_quiz when the user asks for a quiz or practice questions.\n"
    "Use lookup_term to get the official definition of a LangChain term.\n"  # ★ 新增
    "When the user tells you their name or how they like answers, call update_profile."
)


# 【文档原代码】tools · Context
@dataclass
class UserContext:
    user_id: str


# 【补写】笔记的 namespace
def notes_ns(runtime) -> tuple[str, str]:
    return (runtime.context.user_id, "notes")


# 【补写】用户资料的 namespace
def profile_ns(runtime) -> tuple[str, str]:
    return (runtime.context.user_id, "profile")


# 【补写】学习助手的工具（ToolRuntime 写法参考 tools 页）
@tool
def save_note(text: str, runtime: ToolRuntime[UserContext]) -> str:
    """Save a study note for the user."""
    key = uuid.uuid4().hex[:8]
    runtime.store.put(notes_ns(runtime), key, {"text": text})  # 写入时自动计算 text 字段的向量
    return f"Saved note {key}."


# 【补写】学习助手的工具
@tool
def list_notes(runtime: ToolRuntime[UserContext]) -> str:
    """List all of the user's notes with their IDs."""
    items = runtime.store.search(notes_ns(runtime), limit=50)
    return "\n".join(f"[{i.key}] {i.value['text']}" for i in items) or "No notes yet."


# 【补写】学习助手的工具（store.search 写法参考 long-term-memory 页）
@tool
def search_notes(query: str, runtime: ToolRuntime[UserContext]) -> str:
    """Semantic search over the user's saved notes. Returns the most relevant notes."""
    hits = runtime.store.search(notes_ns(runtime), query=query, limit=3)
    if not hits:
        return "No notes yet."
    return "\n".join(f"[{h.key}] (score {h.score:.2f}) {h.value['text']}" for h in hits)


# 【补写】学习助手的工具
@tool
def delete_note(note_id: str, runtime: ToolRuntime[UserContext]) -> str:
    """Delete a note by its ID. Requires human approval."""
    if runtime.store.get(notes_ns(runtime), note_id) is None:
        return f"No note with ID {note_id}."
    runtime.store.delete(notes_ns(runtime), note_id)
    return f"Deleted note {note_id}."


# 【补写】学习助手的工具（读写 store 写法参考 context-engineering · Writes）
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


# 【补写】出题的 schema（写法参考 models · Structured output）
class QuizQuestion(BaseModel):
    question: str
    choices: list[str] = Field(description="Exactly 4 answer choices, without letter prefixes")
    answer: Literal["A", "B", "C", "D"]
    explanation: str = Field(description="One sentence explaining the correct answer")


# 【补写】出题的 schema
class Quiz(BaseModel):
    topic: str
    questions: list[QuizQuestion]


# 【补写】出题工具
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


# 【补写】写法参考 context-engineering · System Prompt
@dynamic_prompt
async def personalized_prompt(request: ModelRequest) -> str:  # ★ 修改：async
    profile = await request.runtime.store.aget(profile_ns(request.runtime), "me")  # ★ 修改：get → aget
    p = profile.value if profile else {}
    prompt = BASE_PROMPT
    if p.get("name"):
        prompt += f"\nThe user's name is {p['name']}."
    if p.get("answer_style"):
        prompt += f"\nThe user prefers {p['answer_style']} answers."
    return prompt


# 【补写】关键词表
BLOCKED_KEYWORDS = ["malware", "exploit", "ddos"]


# 【补写】写法参考 guardrails · Before agent guardrails
@before_agent(can_jump_to=["end"])
def block_harmful_requests(state, runtime: Runtime) -> dict[str, Any] | None:
    text = state["messages"][-1].text.lower()
    if any(word in text for word in BLOCKED_KEYWORDS):
        return {
            "messages": [{"role": "assistant", "content": "这个请求我不能帮忙。我们继续聊 LangChain 吧！"}],
            "jump_to": "end",
        }
    return None


# 【补写】写法参考 middleware/custom · Tool call monitoring
@wrap_tool_call
async def log_tool_calls(  # ★ 修改：改成 async，并且顺便打印调用内容（代替 interleave）
    request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]
):
    start = time.perf_counter()
    result = await handler(request)
    name, args = request.tool_call["name"], request.tool_call["args"]
    print(f"\n  🔧 {name}({args})  ⏱ {(time.perf_counter() - start) * 1000:.0f} ms", end="")
    return result


# 【补写】真实模型 / 假向量二选一
def get_embeddings():
    if FAKE:
        return DeterministicFakeEmbedding(size=256)
    return init_embeddings(os.getenv("LC_EMBEDDINGS", "openai:text-embedding-3-small"))


# 【补写】向量索引配置
def index_config() -> dict:
    embeddings = get_embeddings()
    dims = len(embeddings.embed_query("dimension probe"))
    return {"embed": embeddings, "dims": dims, "fields": ["text"]}


# 【补写】写法参考 multi-agent/subagents · Basic implementation
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

    # 【补写】子 agent 包装成工具
    @tool("research", description="Delegate a question to the researcher subagent and get its explanation.")
    def research(question: str) -> str:
        result = researcher.invoke({"messages": [{"role": "user", "content": question}]})
        return result["messages"][-1].text  # 只把最终结论交回主 agent

    return research


# 【补写】组装学习助手
def build_agent(store: AsyncSqliteStore, checkpointer: AsyncSqliteSaver, mcp_tools: list):  # ★ 修改
    return create_agent(
        model=get_model(),
        tools=[save_note, list_notes, search_notes, delete_note, update_profile, make_quiz, make_research_tool(), *mcp_tools],  # ★ 修改
        checkpointer=checkpointer,
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
            log_tool_calls,  # ★ 修改
        ],
    )


# 【补写】生成新的 thread_id
def new_config() -> dict:
    thread_id = uuid.uuid4().hex
    LAST_THREAD_FILE.write_text(thread_id)
    return {"configurable": {"thread_id": thread_id}}


# 【补写】读取上次的 thread_id
def resume_config() -> dict:
    if LAST_THREAD_FILE.exists():
        return {"configurable": {"thread_id": LAST_THREAD_FILE.read_text().strip()}}
    return new_config()


# 【补写】聊天一轮
async def chat(agent, text: str, config: dict, context: UserContext) -> None:  # ★ 修改：async 版本
    payload = {"messages": [{"role": "user", "content": text}]}
    while True:
        stream = await agent.astream_events(payload, config, context=context, version="v3")
        print("助手: ", end="", flush=True)
        # 异步流没有 interleave，这里只流式打印文字；工具调用由 log_tool_calls 中间件打印
        printed = False
        async for message in stream.messages:
            async for token in message.text:
                printed = True
                print(token, end="", flush=True)
        if not printed:  # 被护栏拦截时回复不是模型生成的，流里没有 token
            print((await agent.aget_state(config)).values["messages"][-1].text, end="")
        print()
        if not await stream.interrupted():  # 异步版本里 interrupted / interrupts 是需要 await 的方法
            return
        decisions = []
        for request in (await stream.interrupts())[0].value["action_requests"]:
            prompt = f"⚠️  {request['description']}\n   批准吗？[y/n] "
            answer = (await asyncio.to_thread(input, prompt)).strip().lower()  # 不阻塞事件循环
            decisions.append({"type": "approve"} if answer == "y" else {"type": "reject", "message": "The user declined."})
        payload = Command(resume={"decisions": decisions})


# 【补写】打印长期记忆
async def show_memory(store, user_id: str) -> None:  # ★ 修改：在事件循环里要用 asearch
    for category in ("profile", "notes"):
        for item in await store.asearch((user_id, category), limit=50):
            print(f"  {category:<8} {item.key}: {item.value}")


# 【补写】命令行主循环
async def main() -> None:  # ★ 修改：async + 打开 MCP 连接
    DATA_DIR.mkdir(exist_ok=True)
    async with (
        AsyncSqliteSaver.from_conn_string(str(DATA_DIR / "checkpoints.db")) as checkpointer,
        AsyncSqliteStore.from_conn_string(str(DATA_DIR / "store.db"), index=index_config()) as store,
        MCPAdapter(MCP_SERVER) as mcp,  # ★ 新增：以子进程方式启动 MCP 服务器（stdio）
    ):
        await checkpointer.setup()
        await store.setup()
        mcp_tools = await mcp.list_tools()  # ★ 新增
        agent = build_agent(store, checkpointer, mcp_tools)
        config = resume_config()
        context = UserContext(user_id="local-user")
        print("学习助手 v15（/memory 查看长期记忆，/user <名字>，/new，/quit）")
        print(f"（数据保存在 {DATA_DIR}；MCP 工具：{', '.join(t.name for t in mcp_tools)}）")
        while True:
            try:
                text = (await asyncio.to_thread(input, f"\n{context.user_id}: ")).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text == "/quit":
                break
            if text == "/new":
                config = new_config()
                print("（已开启新对话：短期记忆清空，长期记忆保留）")
                continue
            if text == "/memory":
                await show_memory(store, context.user_id)
                continue
            if text.startswith("/user "):
                context = UserContext(user_id=text.removeprefix("/user ").strip())
                config = new_config()
                print(f"（已切换到用户 {context.user_id}）")
                continue
            if text:
                await chat(agent, text, config, context)


# 【补写】命令行入口
if __name__ == "__main__":
    asyncio.run(main())  # ★ 修改
