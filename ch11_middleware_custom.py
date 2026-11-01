"""第 11 章 · Custom middleware：自己写中间件。

文档：https://docs.langchain.com/oss/python/langchain/middleware/custom

两类钩子：
- 节点式（node-style）：before_agent / before_model / after_model / after_agent
    在某个时刻运行一次，返回 dict 更新 state，或 {"jump_to": "end"} 提前结束
- 包裹式（wrap-style）：wrap_model_call / wrap_tool_call
    拿到 handler，自己决定调用几次、改不改请求和结果（重试、换模型、缓存……）
两种写法：装饰器（@before_model）或继承 AgentMiddleware 类。
"""

from collections.abc import Callable
from typing import Annotated, Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    after_model,
    before_model,
    hook_config,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langchain.tools.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing_extensions import NotRequired

from common import get_model, get_small_model, last_text, run_demos


@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


def _ask(agent, text: str = "What's the weather in Paris?", **state):
    return agent.invoke({"messages": [HumanMessage(text)], **state})


# ---------------------------------------------------------------------------
# 节点式钩子：装饰器写法 vs 类写法
# ---------------------------------------------------------------------------
@before_model(can_jump_to=["end"])
def check_message_limit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    if len(state["messages"]) >= 50:
        return {"messages": [AIMessage("Conversation limit reached.")], "jump_to": "end"}
    return None


@after_model
def log_response(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"[after_model] Model returned: {state['messages'][-1].text[:80]!r}")
    return None


class MessageLimitMiddleware(AgentMiddleware):
    def __init__(self, max_messages: int = 50):
        super().__init__()
        self.max_messages = max_messages

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if len(state["messages"]) >= self.max_messages:
            return {"messages": [AIMessage("Conversation limit reached.")], "jump_to": "end"}
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"[MessageLimitMiddleware] {len(state['messages'])} messages so far")
        return None


def demo_node_style_hooks():
    """@before_model / @after_model，以及等价的类写法"""
    agent = create_agent(get_model(), tools=[get_weather], middleware=[check_message_limit, log_response])
    print(last_text(_ask(agent)))

    # max_messages=1：第一次调用模型前就触发 jump_to end
    agent = create_agent(get_model(), tools=[get_weather], middleware=[MessageLimitMiddleware(max_messages=1)])
    print(last_text(_ask(agent)))


# ---------------------------------------------------------------------------
# 包裹式钩子：重试
# ---------------------------------------------------------------------------
@wrap_model_call
def retry_model(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"Retry {attempt + 1}/3 after error: {e}")


class RetryMiddleware(AgentMiddleware):
    def __init__(self, max_retries: int = 3):
        super().__init__()
        self.max_retries = max_retries

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        for attempt in range(self.max_retries):
            try:
                return handler(request)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                print(f"Retry {attempt + 1}/{self.max_retries} after error: {e}")


def demo_wrap_style_hooks():
    """@wrap_model_call 与 AgentMiddleware.wrap_model_call 实现重试"""
    for mw in (retry_model, RetryMiddleware()):
        agent = create_agent(get_model(), tools=[get_weather], middleware=[mw])
        print(last_text(_ask(agent)))


# ---------------------------------------------------------------------------
# 更新 state
# ---------------------------------------------------------------------------
class TrackingState(AgentState):
    model_call_count: NotRequired[int]


@after_model(state_schema=TrackingState)
def increment_after_model(state: TrackingState, runtime: Runtime) -> dict[str, Any] | None:
    return {"model_call_count": state.get("model_call_count", 0) + 1}


class UsageTrackingState(AgentState):
    """Agent state with token usage tracking."""

    last_model_call_tokens: NotRequired[int]


@wrap_model_call(state_schema=UsageTrackingState)
def track_usage(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ExtendedModelResponse:
    response = handler(request)
    usage = getattr(response.result[-1], "usage_metadata", None) or {}
    # 包裹式钩子不能直接返回 dict，用 ExtendedModelResponse + Command 更新 state
    return ExtendedModelResponse(
        model_response=response,
        command=Command(update={"last_model_call_tokens": usage.get("total_tokens", 0)}),
    )


def demo_state_updates():
    """节点式返回 dict；包裹式返回 ExtendedModelResponse(command=...)"""
    agent = create_agent(get_model(), tools=[get_weather], middleware=[increment_after_model, track_usage])
    result = _ask(agent)
    print("model_call_count =", result.get("model_call_count"))
    print("last_model_call_tokens =", result.get("last_model_call_tokens"))


def _last_wins(_a: str, b: str) -> str:
    """Reducer: last writer wins (outer overwrites inner)."""
    return b


class CustomMiddlewareState(AgentState):
    trace_layer: NotRequired[Annotated[str, _last_wins]]


class OuterMiddleware(AgentMiddleware):
    state_schema = CustomMiddlewareState

    def wrap_model_call(self, request, handler) -> ExtendedModelResponse:
        response = handler(request)
        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={"trace_layer": "outer", "messages": [SystemMessage(content="[Outer ran]")]}),
        )


class InnerMiddleware(AgentMiddleware):
    state_schema = CustomMiddlewareState

    def wrap_model_call(self, request, handler) -> ExtendedModelResponse:
        response = handler(request)
        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={"trace_layer": "inner", "messages": [SystemMessage(content="[Inner ran]")]}),
        )


def demo_composition():
    """多个包裹式中间件叠加：列表靠前的在外层；reducer 决定冲突字段谁赢"""
    agent = create_agent(get_model(), tools=[], middleware=[OuterMiddleware(), InnerMiddleware()])
    result = _ask(agent, "hi")
    print("trace_layer =", result.get("trace_layer"))
    print([m.text for m in result["messages"] if isinstance(m, SystemMessage)])


# ---------------------------------------------------------------------------
# 自定义 state schema + 跳转
# ---------------------------------------------------------------------------
class CounterState(AgentState):
    model_call_count: NotRequired[int]
    user_id: NotRequired[str]


class CallCounterMiddleware(AgentMiddleware[CounterState]):
    state_schema = CounterState

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: CounterState, runtime) -> dict[str, Any] | None:
        if state.get("model_call_count", 0) > 10:
            return {"jump_to": "end"}
        return None

    def after_model(self, state: CounterState, runtime) -> dict[str, Any] | None:
        return {"model_call_count": state.get("model_call_count", 0) + 1}


@after_model
@hook_config(can_jump_to=["end"])
def check_for_blocked(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    if "BLOCKED" in state["messages"][-1].text:
        return {"messages": [AIMessage("I cannot respond to that request.")], "jump_to": "end"}
    return None


def demo_custom_state_and_jumps():
    """中间件声明自己的 state 字段；jump_to 提前结束"""
    agent = create_agent(get_model(), tools=[], middleware=[CallCounterMiddleware()])
    result = _ask(agent, "Hello", model_call_count=0, user_id="user-123")
    print("model_call_count =", result["model_call_count"])

    agent = create_agent(get_model(), tools=[], middleware=[check_for_blocked])
    print(last_text(_ask(agent, "Reply with exactly the word BLOCKED.")))


# ---------------------------------------------------------------------------
# 常用示例：动态提示词、动态模型、工具监控、缓存提示
# ---------------------------------------------------------------------------
@wrap_model_call
def add_context(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    # system_message 可能为空；用 content_blocks 追加内容
    blocks = list(request.system_message.content_blocks) if request.system_message else []
    new_system_message = SystemMessage(content=blocks + [{"type": "text", "text": "Always answer in one sentence."}])
    return handler(request.override(system_message=new_system_message))


def make_dynamic_model_middleware():
    # 模型对象在中间件外面只初始化一次，不要每次调用都 init
    complex_model, simple_model = get_model(), get_small_model()

    @wrap_model_call
    def dynamic_model(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        model = complex_model if len(request.messages) > 10 else simple_model
        return handler(request.override(model=model))

    return dynamic_model


@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    print(f"Executing tool: {request.tool_call['name']}  args={request.tool_call['args']}")
    try:
        result = handler(request)
        print("Tool completed successfully")
        return result
    except Exception as e:
        print(f"Tool failed: {e}")
        raise


def demo_examples():
    """动态系统提示词 + 动态模型 + 工具调用监控，叠在一起用"""
    agent = create_agent(
        get_model(),
        tools=[get_weather],
        system_prompt="You are a weather bot.",
        middleware=[add_context, make_dynamic_model_middleware(), monitor_tool],
    )
    print(last_text(_ask(agent)))


@wrap_model_call
def add_cached_context(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """Anthropic 提示缓存：cache_control 之前的内容会被缓存"""
    blocks = list(request.system_message.content_blocks) if request.system_message else []
    new_content = blocks + [
        {
            "type": "text",
            "text": "Here is a large document to analyze:\n\n<document>...</document>",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    return handler(request.override(system_message=SystemMessage(content=new_content)))


def demo_prompt_caching():
    """Anthropic 提示缓存（其他供应商会忽略 cache_control）"""
    agent = create_agent(get_model(), tools=[], system_prompt="You analyze documents.", middleware=[add_cached_context])
    result = _ask(agent, "What is in the document?")
    print(last_text(result))
    print(result["messages"][-1].usage_metadata)  # 第二次调用时留意 cache_read 字段


def demo_class_based_logging():
    """完整的类写法：同步 + 异步钩子"""

    class LoggingMiddleware(AgentMiddleware):
        def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            print(f"About to call model with {len(state['messages'])} messages")
            return None

        def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            print(f"Model returned: {state['messages'][-1].text[:60]!r}")
            return None

        async def abefore_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            print(f"(async) About to call model with {len(state['messages'])} messages")
            return None

        async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            print(f"(async) Model returned: {state['messages'][-1].text[:60]!r}")
            return None

    agent = create_agent(get_model(), tools=[get_weather], middleware=[LoggingMiddleware()])
    _ask(agent)

    import asyncio

    asyncio.run(agent.ainvoke({"messages": [HumanMessage("hi")]}))  # 异步调用走 a* 钩子


if __name__ == "__main__":
    run_demos(globals())
