"""第 9 章 · Runtime + Context engineering：给模型"恰到好处"的上下文。

文档：
- https://docs.langchain.com/oss/python/langchain/runtime
- https://docs.langchain.com/oss/python/langchain/context-engineering

三个数据来源：
    State   —— 本次对话的数据（messages、自定义字段），随对话变化
    Store   —— 跨对话的长期数据（用户偏好等）
    Context —— 每次调用传入的只读配置（用户 ID、角色、环境……）
可以用它们动态决定：系统提示词 / 注入的消息 / 可用工具 / 使用的模型 / 输出格式。
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    after_model,
    before_model,
    dynamic_prompt,
    wrap_model_call,
)
from langchain.tools import ToolRuntime, tool
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from pydantic import BaseModel, Field
from typing_extensions import NotRequired

from common import FAKE, get_model, get_small_model, last_text, run_demos


# ---------------------------------------------------------------------------
# Runtime：在工具与中间件里拿到 context / store / 执行信息
# ---------------------------------------------------------------------------
@dataclass
class NameContext:
    user_name: str


@dynamic_prompt
def name_prompt(request: ModelRequest) -> str:
    return f"You are a helpful assistant. Address the user as {request.runtime.context.user_name}."


@before_model
def log_before_model(state: AgentState, runtime: Runtime[NameContext]) -> dict | None:
    print(f"[before_model] Processing request for user: {runtime.context.user_name}")
    print(f"[before_model] thread: {runtime.execution_info.thread_id}")
    return None


@after_model
def log_after_model(state: AgentState, runtime: Runtime[NameContext]) -> dict | None:
    print(f"[after_model] Completed request for user: {runtime.context.user_name}")
    return None


def demo_runtime_in_middleware():
    """中间件通过 request.runtime / runtime 读取 context"""
    agent = create_agent(
        model=get_model(),
        tools=[],
        middleware=[name_prompt, log_before_model, log_after_model],
        context_schema=NameContext,
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's my name?"}]},
        context=NameContext(user_name="John Smith"),
    )
    print(last_text(result))


@dataclass
class UserIdContext:
    user_id: str


@tool
def fetch_user_email_preferences(runtime: ToolRuntime[UserIdContext]) -> str:
    """Fetch the user's email preferences from the store."""
    preferences = "The user prefers you to write a brief and polite email."
    if runtime.store and (memory := runtime.store.get(("users",), runtime.context.user_id)):
        preferences = memory.value["preferences"]
    return preferences


def demo_runtime_in_tools():
    """工具里同时用 context（谁）和 store（他的偏好）"""
    store = InMemoryStore()
    store.put(("users",), "u1", {"preferences": "Write in a casual tone and sign off with 'Cheers'."})
    agent = create_agent(
        get_model(), tools=[fetch_user_email_preferences], context_schema=UserIdContext, store=store
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Draft a two-line email thanking my TA."}]},
        context=UserIdContext(user_id="u1"),
    )
    print(last_text(result))


# ---------------------------------------------------------------------------
# 系统提示词：按 State / Store / Context 动态生成
# ---------------------------------------------------------------------------
@dataclass
class AppContext:
    user_id: str = "u1"
    user_role: str = "viewer"
    deployment_env: str = "development"
    cost_tier: str = "standard"
    user_jurisdiction: str = "EU"
    industry: str = "finance"
    compliance_frameworks: list[str] = field(default_factory=lambda: ["GDPR"])


@dynamic_prompt
def state_aware_prompt(request: ModelRequest) -> str:
    base = "You are a helpful assistant."
    if len(request.messages) > 10:  # request.messages 是 request.state["messages"] 的简写
        base += "\nThis is a long conversation - be extra concise."
    return base


@dynamic_prompt
def store_aware_prompt(request: ModelRequest) -> str:
    base = "You are a helpful assistant."
    prefs = request.runtime.store.get(("preferences",), request.runtime.context.user_id)
    if prefs:
        base += f"\nUser prefers {prefs.value.get('communication_style', 'balanced')} responses."
    return base


@dynamic_prompt
def context_aware_prompt(request: ModelRequest) -> str:
    ctx = request.runtime.context
    base = "You are a helpful assistant."
    if ctx.user_role == "admin":
        base += "\nYou have admin access. You can perform all operations."
    elif ctx.user_role == "viewer":
        base += "\nYou have read-only access. Guide users to read operations only."
    if ctx.deployment_env == "production":
        base += "\nBe extra careful with any data modifications."
    return base


def demo_dynamic_system_prompt():
    """三个 @dynamic_prompt：分别读 State、Store、Context"""
    store = InMemoryStore()
    store.put(("preferences",), "u1", {"communication_style": "very terse"})
    for prompt_mw in (state_aware_prompt, store_aware_prompt, context_aware_prompt):
        agent = create_agent(
            get_model(), tools=[], middleware=[prompt_mw], context_schema=AppContext, store=store
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Can you delete the sales table?"}]},
            context=AppContext(user_role="viewer", deployment_env="production"),
        )
        print(f"[{prompt_mw.name}] {last_text(result)[:200]}\n")


# ---------------------------------------------------------------------------
# 注入消息：上传的文件、写作风格、合规要求
# ---------------------------------------------------------------------------
class FileState(AgentState):
    uploaded_files: NotRequired[list[dict]]


@wrap_model_call
def inject_file_context(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """Inject context about files user has uploaded this session."""
    uploaded_files = request.state.get("uploaded_files", [])
    if uploaded_files:
        descriptions = "\n".join(f"- {f['name']} ({f['type']}): {f['summary']}" for f in uploaded_files)
        file_context = f"Files you have access to in this conversation:\n{descriptions}\n\nReference these files when answering questions."
        request = request.override(messages=[*request.messages, {"role": "user", "content": file_context}])
    return handler(request)


@wrap_model_call
def inject_writing_style(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """Inject user's email writing style from Store."""
    writing_style = request.runtime.store.get(("writing_style",), request.runtime.context.user_id)
    if writing_style:
        style = writing_style.value
        style_context = (
            f"Your writing style:\n- Tone: {style.get('tone', 'professional')}\n"
            f"- Typical greeting: \"{style.get('greeting', 'Hi')}\"\n"
            f"- Typical sign-off: \"{style.get('sign_off', 'Best')}\""
        )
        # 放在最后：模型更关注末尾的消息
        request = request.override(messages=[*request.messages, {"role": "user", "content": style_context}])
    return handler(request)


@wrap_model_call
def inject_compliance_rules(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """Inject compliance constraints from Runtime Context."""
    ctx = request.runtime.context
    rules = []
    if "GDPR" in ctx.compliance_frameworks:
        rules += ["- Must obtain explicit consent before processing personal data", "- Users have right to data deletion"]
    if "HIPAA" in ctx.compliance_frameworks:
        rules += ["- Cannot share patient health information without authorization"]
    if ctx.industry == "finance":
        rules += ["- Cannot provide financial advice without proper disclaimers"]
    if rules:
        compliance = f"Compliance requirements for {ctx.user_jurisdiction}:\n" + "\n".join(rules)
        request = request.override(messages=[*request.messages, {"role": "user", "content": compliance}])
    return handler(request)


def demo_inject_messages():
    """wrap_model_call + request.override(messages=...) 注入额外上下文"""
    store = InMemoryStore()
    store.put(("writing_style",), "u1", {"tone": "warm", "greeting": "Hey", "sign_off": "Cheers"})
    agent = create_agent(
        get_model(),
        tools=[],
        middleware=[inject_file_context, inject_writing_style, inject_compliance_rules],
        state_schema=FileState,
        context_schema=AppContext,
        store=store,
    )
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "Write a short email to a client about the attached report."}],
            "uploaded_files": [{"name": "q3.pdf", "type": "pdf", "summary": "Q3 revenue up 12%"}],
        },
        context=AppContext(),
    )
    print(last_text(result))


# ---------------------------------------------------------------------------
# 选择工具 / 模型 / 输出格式
# ---------------------------------------------------------------------------
@tool
def read_data(table: str) -> str:
    """Read rows from a table."""
    return f"3 rows from {table}"


@tool
def write_data(table: str, row: str) -> str:
    """Write a row to a table."""
    return f"wrote {row} to {table}"


@tool
def delete_data(table: str) -> str:
    """Delete a table."""
    return f"deleted {table}"


@wrap_model_call
def context_based_tools(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """Filter tools based on Runtime Context permissions."""
    role = request.runtime.context.user_role if request.runtime and request.runtime.context else "viewer"
    if role == "editor":
        request = request.override(tools=[t for t in request.tools if t.name != "delete_data"])
    elif role != "admin":
        request = request.override(tools=[t for t in request.tools if t.name.startswith("read_")])
    return handler(request)


def demo_select_tools():
    """按用户角色过滤工具：viewer 只能 read_*"""
    agent = create_agent(
        get_model(),
        tools=[read_data, write_data, delete_data],
        middleware=[context_based_tools],
        context_schema=AppContext,
    )
    for role in ("viewer", "admin"):
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "List the tool names you can use."}]},
            context=AppContext(user_role=role),
        )
        print(f"{role}: {last_text(result)}")


def demo_select_model():
    """按 context 的付费等级选择模型"""
    premium_model, budget_model = get_model(), get_small_model()

    @wrap_model_call
    def context_based_model(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        """Select model based on Runtime Context."""
        ctx = request.runtime.context
        model = budget_model if ctx.cost_tier == "budget" else premium_model
        print(f"[{ctx.cost_tier}] 使用模型 {getattr(model, 'model_name', None) or getattr(model, 'model', model)}")
        return handler(request.override(model=model))

    agent = create_agent(get_model(), tools=[], middleware=[context_based_model], context_schema=AppContext)
    for tier in ("budget", "premium"):
        agent.invoke({"messages": [{"role": "user", "content": "hi"}]}, context=AppContext(cost_tier=tier))


class SimpleResponse(BaseModel):
    """Simple response for early conversation."""

    answer: str = Field(description="A brief answer")


class DetailedResponse(BaseModel):
    """Detailed response for established conversation."""

    answer: str = Field(description="A detailed answer")
    reasoning: str = Field(description="Explanation of reasoning")
    confidence: float = Field(description="Confidence score 0-1")


@wrap_model_call
def state_based_output(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """Select output format based on State."""
    fmt = SimpleResponse if len(request.messages) < 3 else DetailedResponse
    return handler(request.override(response_format=fmt))


def demo_select_response_format():
    """对话早期用简单格式，后期用详细格式"""
    if FAKE:
        print("假模型不支持结构化输出，跳过。")
        return
    agent = create_agent(get_model(), tools=[], middleware=[state_based_output])
    print(agent.invoke({"messages": [{"role": "user", "content": "Is Python interpreted?"}]})["structured_response"])
    long_history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Is Python interpreted?"},
    ]
    print(agent.invoke({"messages": long_history})["structured_response"])


# ---------------------------------------------------------------------------
# 工具写入 State / Store
# ---------------------------------------------------------------------------
class AuthState(AgentState):
    authenticated: NotRequired[bool]


@tool
def authenticate_user(password: str, runtime: ToolRuntime) -> Command:
    """Authenticate user and update State."""
    from langchain.messages import ToolMessage

    ok = password == "correct"
    return Command(update={
        "authenticated": ok,
        "messages": [ToolMessage(f"authenticated={ok}", tool_call_id=runtime.tool_call_id)],
    })


@tool
def save_preference(preference_key: str, preference_value: str, runtime: ToolRuntime[AppContext]) -> str:
    """Save user preference to Store."""
    user_id = runtime.context.user_id
    existing = runtime.store.get(("preferences",), user_id)
    prefs = existing.value if existing else {}
    prefs[preference_key] = preference_value
    runtime.store.put(("preferences",), user_id, prefs)
    return f"Saved preference: {preference_key} = {preference_value}"


def demo_tool_writes():
    """工具用 Command 写 State，用 runtime.store 写 Store"""
    store = InMemoryStore()
    agent = create_agent(
        get_model(),
        tools=[authenticate_user, save_preference],
        state_schema=AuthState,
        context_schema=AppContext,
        store=store,
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "My password is 'correct'. Also remember I like dark mode."}]},
        context=AppContext(),
    )
    print(last_text(result))
    print("state.authenticated =", result.get("authenticated"))
    print("store =", store.get(("preferences",), "u1"))


if __name__ == "__main__":
    run_demos(globals())
