"""第 4 章 · Tools：让 agent 能"做事"。

文档：https://docs.langchain.com/oss/python/langchain/tools

要点：
- @tool 把函数变成工具；名字、描述、参数 schema 都能自定义
- ToolRuntime 参数（模型看不到）可以读写 state、context、store，或推送流式进度
- 工具可以返回字符串、字典、多模态内容，或 Command（直接改 agent 的 state）
- 用中间件动态增减工具
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from pydantic import BaseModel, Field

from common import get_model, last_text, run_demos


# ---------------------------------------------------------------------------
# 定义工具
# ---------------------------------------------------------------------------
@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    return f"Found {limit} results for '{query}'"


@tool("web_search")  # 自定义工具名
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"


@tool("calculator", description="Performs arithmetic calculations. Use this for any math problems.")
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    # 学习用：真实项目里不要对模型给出的字符串直接 eval
    return str(eval(expression, {"__builtins__": {}}))


class WeatherInput(BaseModel):
    """Input for weather queries."""

    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius", description="Temperature unit preference"
    )
    include_forecast: bool = Field(default=False, description="Include 5-day forecast")


@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result


def demo_define_tools():
    """查看工具的 name / description / 参数 schema，并直接调用"""
    for t in (search_database, search, calc, get_weather):
        print(f"{t.name}: {t.description}\n  args={t.args}")
    print(get_weather.invoke({"location": "Madison", "units": "fahrenheit"}))
    agent = create_agent(get_model(), tools=[search_database, search, calc, get_weather])
    print(last_text(agent.invoke({"messages": [{"role": "user", "content": "What is 17 * 23?"}]})))


# ---------------------------------------------------------------------------
# ToolRuntime：读写 state
# ---------------------------------------------------------------------------
class CustomState(AgentState):
    user_name: str
    user_preferences: dict


@tool
def get_last_user_message(runtime: ToolRuntime) -> str:
    """Get the most recent message from the user."""
    for message in reversed(runtime.state["messages"]):
        if isinstance(message, HumanMessage):
            return message.content
    return "No user messages found"


@tool
def get_user_preference(pref_name: str, runtime: ToolRuntime) -> str:
    """Get a user preference value."""
    return runtime.state.get("user_preferences", {}).get(pref_name, "Not set")


@tool
def set_user_name(new_name: str, runtime: ToolRuntime[None, CustomState]) -> Command:
    """Set the user's name in the conversation state."""
    # 返回 Command 来更新 state；必须带上 ToolMessage，保证消息历史合法
    return Command(
        update={
            "user_name": new_name,
            "messages": [
                ToolMessage(content=f"User name set to {new_name}.", tool_call_id=runtime.tool_call_id)
            ],
        }
    )


def demo_state_access():
    """工具通过 runtime.state 读状态，通过 Command 写状态"""
    agent = create_agent(
        get_model(),
        tools=[get_last_user_message, get_user_preference, set_user_name],
        state_schema=CustomState,
    )
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "Call me Ada, and tell me my theme preference."}],
            "user_preferences": {"theme": "dark"},
        }
    )
    print(last_text(result))
    print("state 里的 user_name =", result.get("user_name"))


# ---------------------------------------------------------------------------
# ToolRuntime：读取 context（每次调用传入的只读配置，如用户 ID）
# ---------------------------------------------------------------------------
USER_DATABASE = {
    "user123": {"name": "Alice Johnson", "account_type": "Premium", "balance": 5000},
    "user456": {"name": "Bob Smith", "account_type": "Standard", "balance": 1200},
}


@dataclass
class UserContext:
    user_id: str


@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """Get the current user's account information."""
    user = USER_DATABASE.get(runtime.context.user_id)
    if user is None:
        return "User not found"
    return f"Account holder: {user['name']}\nType: {user['account_type']}\nBalance: ${user['balance']}"


def demo_context_access():
    """context_schema + ToolRuntime[Context]：按用户返回不同数据"""
    agent = create_agent(
        get_model(),
        tools=[get_account_info],
        context_schema=UserContext,
        checkpointer=InMemorySaver(),
        system_prompt="You are a financial assistant.",
    )
    config = {"configurable": {"thread_id": str(uuid7())}}
    context = UserContext(user_id="user123")
    for question in ("What's my current balance?", "What was my balance again?"):
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]}, config=config, context=context
        )
        print(last_text(result))


# ---------------------------------------------------------------------------
# ToolRuntime：读写 store（跨会话的长期记忆，第 14 章详细讲）
# ---------------------------------------------------------------------------
@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """Look up user info."""
    user_info = runtime.store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"


@tool
def save_user_info(user_id: str, name: str, age: int, email: str, runtime: ToolRuntime) -> str:
    """Save user info."""
    runtime.store.put(("users",), user_id, {"name": name, "age": age, "email": email})
    return "Successfully saved user info."


def demo_store_access():
    """store：第一次对话写入，第二次对话（无 checkpointer）仍能读到"""
    store = InMemoryStore()
    agent = create_agent(get_model(), tools=[get_user_info, save_user_info], store=store)
    agent.invoke({
        "messages": [{"role": "user", "content": "Save the following user: userid: abc123, name: Foo, age: 25, email: foo@langchain.dev"}]
    })
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Get user info for user with id 'abc123'"}]
    })
    print(last_text(result))
    print("store 里实际存的：", store.get(("users",), "abc123"))


# ---------------------------------------------------------------------------
# ToolRuntime：流式进度、执行信息
# ---------------------------------------------------------------------------
@tool
def get_weather_with_progress(city: str, runtime: ToolRuntime) -> str:
    """Get weather for a given city."""
    writer = runtime.stream_writer  # 推送自定义进度，stream_mode="custom" 可收到
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"


@tool
def log_execution_context(runtime: ToolRuntime) -> str:
    """Log execution identity information."""
    info = runtime.execution_info
    print(f"Thread: {info.thread_id}, Run: {info.run_id}, Attempt: {info.node_attempt}")
    # runtime.server_info 只在 LangGraph Server 上部署时才有值
    return "done"


def demo_stream_writer():
    """runtime.stream_writer 推送工具内部进度"""
    agent = create_agent(get_model(), tools=[get_weather_with_progress, log_execution_context])
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "Log the execution context, then get the weather in SF."}]},
        config={"configurable": {"thread_id": "t-1"}},
        stream_mode="custom",
        version="v2",
    ):
        print("custom:", chunk["data"])


# ---------------------------------------------------------------------------
# 返回值的几种形式
# ---------------------------------------------------------------------------
@tool
def get_weather_data(city: str) -> dict:
    """Get structured weather data for a city."""
    return {"city": city, "temperature_c": 22, "conditions": "sunny"}


@tool
def capture_screenshot() -> list[dict]:
    """Capture a screenshot of the current page."""
    return [
        {"type": "text", "text": "Screenshot of the current page:"},
        {"type": "image", "url": "https://example.com/page.png"},
    ]


@tool(return_direct=True)
def fetch_order_status(order_id: str) -> str:
    """Fetch the current status of a customer order."""
    return f"Order {order_id} is shipped and will arrive in 2 days."


class OrderState(AgentState):
    last_order_status: str


@tool(return_direct=True)
def fetch_and_store_order(order_id: str, runtime: ToolRuntime) -> Command:
    """Fetch order status and store it in state."""
    status = f"Order {order_id} is shipped and will arrive in 2 days."
    return Command(
        update={
            "last_order_status": status,
            "messages": [ToolMessage(content=status, tool_call_id=runtime.tool_call_id)],
        }
    )


def demo_return_values():
    """返回 dict、多模态列表，以及 return_direct 跳过最后一次模型调用"""
    print(get_weather_data.invoke({"city": "Tokyo"}))
    print(capture_screenshot.invoke({}))

    agent = create_agent(get_model(), tools=[fetch_order_status])
    result = agent.invoke({"messages": [{"role": "user", "content": "What is the status of order #12345?"}]})
    print("return_direct ->", last_text(result))

    agent = create_agent(get_model(), tools=[fetch_and_store_order], state_schema=OrderState)
    result = agent.invoke({"messages": [{"role": "user", "content": "Check order 777 and remember it."}]})
    print("state.last_order_status ->", result.get("last_order_status"))


# ---------------------------------------------------------------------------
# 错误处理与动态工具
# ---------------------------------------------------------------------------
@tool
def flaky_tool(x: int) -> str:
    """A tool that fails for negative numbers."""
    if x < 0:
        raise ValueError("x must be non-negative")
    return f"ok: {x}"


@wrap_tool_call
def handle_tool_errors(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage],
) -> ToolMessage:
    """Convert tool exceptions into ToolMessages the model can handle."""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({e})",
            tool_call_id=request.tool_call["id"],
        )


def demo_error_handling():
    """wrap_tool_call 把异常变成模型能读懂的 ToolMessage"""
    agent = create_agent(get_model(), tools=[flaky_tool], middleware=[handle_tool_errors])
    result = agent.invoke({"messages": [{"role": "user", "content": "Call flaky_tool with x=-1, then with x=3."}]})
    for m in result["messages"]:
        if isinstance(m, ToolMessage):
            print("ToolMessage:", m.content)
    print(last_text(result))


@tool
def public_search(query: str) -> str:
    """Public search anyone can use."""
    return f"public results for {query}"


@tool
def private_search(query: str) -> str:
    """Search private records (requires authentication)."""
    return f"private results for {query}"


class AuthState(AgentState):
    authenticated: bool


@wrap_model_call
def state_based_tools(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Filter tools based on conversation State."""
    if not request.state.get("authenticated", False):
        tools = [t for t in request.tools if t.name.startswith("public_")]
        request = request.override(tools=tools)
    return handler(request)


@tool
def calculate_tip(bill_amount: float, tip_percentage: float = 20.0) -> str:
    """Calculate the tip amount for a bill."""
    tip = bill_amount * (tip_percentage / 100)
    return f"Tip: ${tip:.2f}, Total: ${bill_amount + tip:.2f}"


class DynamicToolMiddleware(AgentMiddleware):
    """Middleware that registers and handles dynamic tools."""

    def wrap_model_call(self, request: ModelRequest, handler):
        # 运行时追加工具（可以来自 MCP、数据库……）
        return handler(request.override(tools=[*request.tools, calculate_tip]))

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        if request.tool_call["name"] == "calculate_tip":
            return handler(request.override(tool=calculate_tip))
        return handler(request)


def demo_dynamic_tools():
    """按 state 过滤工具；运行时动态注册新工具"""
    agent = create_agent(
        get_model(),
        tools=[public_search, private_search],
        middleware=[state_based_tools],
        state_schema=AuthState,
    )
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Which search tools can you use right now? Just list them."}],
        "authenticated": False,
    })
    print("未登录：", last_text(result))

    agent = create_agent(get_model(), tools=[get_weather], middleware=[DynamicToolMiddleware()])
    result = agent.invoke({"messages": [{"role": "user", "content": "Calculate a 20% tip on $85"}]})
    print("动态工具：", last_text(result))


if __name__ == "__main__":
    run_demos(globals())
