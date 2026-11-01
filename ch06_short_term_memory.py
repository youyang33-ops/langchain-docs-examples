"""第 6 章 · Short-term memory：同一个对话线程（thread）内的记忆。

文档：https://docs.langchain.com/oss/python/langchain/short-term-memory

- checkpointer 每一步都保存 state，同一个 thread_id 的下一次调用会接着上次的历史
- 上下文太长时：裁剪（trim）、删除（RemoveMessage）、摘要（SummarizationMiddleware）
- 生产环境把 InMemorySaver 换成 PostgresSaver 等持久化实现
"""

from typing import Any, TypedDict

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    ModelRequest,
    SummarizationMiddleware,
    after_model,
    before_model,
    dynamic_prompt,
)
from langchain.messages import RemoveMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel
from typing_extensions import NotRequired

from common import get_model, get_small_model, last_text, run_demos


def get_user_info() -> str:
    """Look up information about the current user."""
    return "No user profile on file."


def demo_basic_memory():
    """同一个 thread_id 两次调用，第二次记得第一次说过的名字"""
    agent = create_agent(model=get_model(), tools=[get_user_info], checkpointer=InMemorySaver())
    thread_config = {"configurable": {"thread_id": "1"}}
    print(last_text(agent.invoke({"messages": [{"role": "user", "content": "Hi! My name is Bob."}]}, thread_config)))
    print(last_text(agent.invoke({"messages": [{"role": "user", "content": "What's my name?"}]}, thread_config)))


def demo_postgres_checkpointer():
    """生产环境：PostgresSaver（需要 Postgres 和 langgraph-checkpoint-postgres）"""
    print("""示例代码（需要本地 Postgres）：

from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # 自动建表
    agent = create_agent(model, tools=[get_user_info], checkpointer=checkpointer)
""")


class CustomAgentState(AgentState):
    user_id: NotRequired[str]
    preferences: NotRequired[dict]


def demo_custom_state():
    """自定义 state 字段，也会被 checkpointer 一起保存"""
    agent = create_agent(
        get_model(),
        tools=[get_user_info],
        state_schema=CustomAgentState,
        checkpointer=InMemorySaver(),
    )
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "user_id": "user_123",
            "preferences": {"theme": "dark"},
        },
        {"configurable": {"thread_id": "1"}},
    )
    print(result["user_id"], result["preferences"])


# ---------------------------------------------------------------------------
# 管理上下文长度
# ---------------------------------------------------------------------------
@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]
    if len(messages) <= 3:
        return None
    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    # 先清空，再写回保留的消息
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), first_msg, *recent_messages]}


def _chat(agent, turns: list[str]) -> None:
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}
    for text in turns:
        result = agent.invoke({"messages": text}, config)
    result["messages"][-1].pretty_print()
    print(f"（state 中现在有 {len(result['messages'])} 条消息）")


CONVERSATION = [
    "hi, my name is bob",
    "write a short poem about cats",
    "now do the same but for dogs",
    "what's my name?",
]


def demo_trim_messages():
    """@before_model：调用模型前只保留第一条 + 最近几条"""
    agent = create_agent(get_model(), tools=[], middleware=[trim_messages], checkpointer=InMemorySaver())
    _chat(agent, CONVERSATION)


@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove old messages to keep conversation manageable."""
    messages = state["messages"]
    if len(messages) > 2:
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
    return None


def demo_delete_messages():
    """@after_model + RemoveMessage：删除最早的两条消息"""
    agent = create_agent(
        get_model(),
        tools=[],
        system_prompt="Please be concise and to the point.",
        middleware=[delete_old_messages],
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}
    for text in ("hi! I'm bob", "write a short poem about cats", "what's my name?"):
        stream = agent.stream_events({"messages": [{"role": "user", "content": text}]}, config, version="v3")
        for snapshot in stream.values:
            print([(m.type, m.content[:30]) for m in snapshot["messages"]])
        print()


def demo_summarize_messages():
    """SummarizationMiddleware：超过阈值时用小模型把旧消息压缩成摘要"""
    agent = create_agent(
        model=get_model(),
        tools=[],
        middleware=[
            SummarizationMiddleware(
                model=get_small_model(),
                trigger=("tokens", 4000),  # 超过 4000 token 触发
                keep=("messages", 20),  # 保留最近 20 条原文
            )
        ],
        checkpointer=InMemorySaver(),
    )
    _chat(agent, CONVERSATION)


# ---------------------------------------------------------------------------
# 在工具 / 提示词 / 钩子里访问短期记忆
# ---------------------------------------------------------------------------
class UserState(AgentState):
    user_id: NotRequired[str]
    user_name: NotRequired[str]


class CustomContext(BaseModel):
    user_id: str


@tool
def lookup_user(runtime: ToolRuntime) -> str:
    """Look up user info."""
    return "User is John Smith" if runtime.state.get("user_id") == "user_123" else "Unknown user"


@tool
def update_user_info(runtime: ToolRuntime[CustomContext, UserState]) -> Command:
    """Look up and update user info."""
    name = "John Smith" if runtime.context.user_id == "user_123" else "Unknown user"
    return Command(update={
        "user_name": name,
        "messages": [ToolMessage("Successfully looked up user information", tool_call_id=runtime.tool_call_id)],
    })


@tool
def greet(runtime: ToolRuntime[CustomContext, UserState]) -> str | Command:
    """Use this to greet the user once you found their info."""
    user_name = runtime.state.get("user_name")
    if user_name is None:
        return Command(update={
            "messages": [
                ToolMessage(
                    "Please call the 'update_user_info' tool it will get and update the user's name.",
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        })
    return f"Hello {user_name}!"


def demo_memory_in_tools():
    """工具读 state（lookup_user）、写 state（update_user_info）"""
    agent = create_agent(model=get_model(), tools=[lookup_user], state_schema=UserState)
    print(last_text(agent.invoke({"messages": "look up user information", "user_id": "user_123"})))

    agent = create_agent(
        model=get_model(),
        tools=[update_user_info, greet],
        state_schema=UserState,
        context_schema=CustomContext,
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "greet the user"}]},
        context=CustomContext(user_id="user_123"),
    )
    print(last_text(result), "| state.user_name =", result.get("user_name"))


class NameContext(TypedDict):
    user_name: str


def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is always sunny!"


@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context["user_name"]
    return f"You are a helpful assistant. Address the user as {user_name}."


def demo_memory_in_prompt():
    """@dynamic_prompt：根据 context 动态生成系统提示词"""
    agent = create_agent(
        model=get_model(),
        tools=[get_weather],
        middleware=[dynamic_system_prompt],
        context_schema=NameContext,
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
        context=NameContext(user_name="John Smith"),
    )
    for msg in result["messages"]:
        msg.pretty_print()


@after_model
def validate_response(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove messages containing sensitive words."""
    stop_words = ["password", "secret"]
    last_message = state["messages"][-1]
    if any(word in last_message.text for word in stop_words):
        return {"messages": [RemoveMessage(id=last_message.id)]}
    return None


def demo_after_model_filter():
    """@after_model：模型回复里出现敏感词就删掉这条回复"""
    agent = create_agent(get_model(), tools=[], middleware=[validate_response], checkpointer=InMemorySaver())
    result = agent.invoke(
        {"messages": "Say the word 'secret' once."}, {"configurable": {"thread_id": "1"}}
    )
    print([(m.type, m.text) for m in result["messages"]])


if __name__ == "__main__":
    run_demos(globals())
