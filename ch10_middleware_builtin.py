"""第 10 章 · Prebuilt middleware：开箱即用的中间件。

文档：
- https://docs.langchain.com/oss/python/langchain/middleware/overview
- https://docs.langchain.com/oss/python/langchain/middleware/built-in

中间件挂在 agent 循环的各个环节（before_agent → before_model → model →
after_model → tools → ... → after_agent），用来加重试、限流、脱敏、审批、摘要等能力。

代码来源标注（每个类、函数和主要常量前面都有一行）：
  【文档原代码】          和文档的代码完全一致（只比较语法，不比较注释和排版）
  【文档原代码·有改动】   来自文档，做了改动（标注里写了改了什么）
  【补写】                文档没有给出这段代码，是我写的
来源写成"页面 · 小节"，页面相对于 https://docs.langchain.com/oss/python/langchain/
注意：overview 页只有 get_weather 这一个例子，其余代码来自它链接的其他页面。
"""

import re
import tempfile

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    FilesystemFileSearchMiddleware,
    HostExecutionPolicy,
    HumanInTheLoopMiddleware,
    LLMToolEmulator,
    LLMToolSelectorMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    PIIMatch,
    PIIMiddleware,
    ShellToolMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
    ToolCallLimitMiddleware,
    ToolCallRequest,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph

from common import FAKE, get_model, get_small_model, last_text, run_demos


# 本章共用的几个示例工具
# 【文档原代码·有改动】agents · Tools：只改了 docstring
@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"


# 【补写】文档只引用了这个工具，没给定义
@tool
def database_tool(sql: str) -> str:
    """Run a read-only SQL query."""
    if "fail" in sql:
        raise ValueError("database temporarily unavailable")
    return "42 rows"


# 【文档原代码】middleware/built-in · Subagent
@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is sunny."


# 【文档原代码·有改动】middleware/built-in · Human-in-the-loop：加了 @tool
@tool
def your_read_email_tool(email_id: str) -> str:
    """Mock function to read an email by its ID."""
    return f"Email content for ID: {email_id}"


# 【文档原代码·有改动】middleware/built-in · Human-in-the-loop：加了 @tool
@tool
def your_send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Mock function to send an email."""
    return f"Email sent to {recipient} with subject '{subject}'"


# 【补写】发送一条消息的小工具
def _run(agent, text: str, **kwargs):
    config = {"configurable": {"thread_id": "mw"}}
    result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config=config, **kwargs)
    return result


# ---------------------------------------------------------------------------
# 容错：错误、重试、降级
# ---------------------------------------------------------------------------
# 【文档原代码·有改动】middleware/built-in · Tool error：末尾显式 return None
def on_error(exc: Exception, request: ToolCallRequest) -> str | None:
    if isinstance(exc, ValueError):
        return f"`{request.tool_call['name']}` failed with {type(exc).__name__}."
    return None  # 其他异常继续抛出


# 【文档原代码·有改动】middleware/built-in · Tool retry：合并 Tool error / Tool retry / Model retry / Model fallback；模型改成 get_model()
def demo_fault_tolerance():
    """ToolErrorMiddleware / ToolRetryMiddleware / ModelRetryMiddleware / ModelFallbackMiddleware"""
    agent = create_agent(
        model=get_model(),
        tools=[search, database_tool],
        middleware=[
            ToolErrorMiddleware(on_error),
            ToolRetryMiddleware(max_retries=3, backoff_factor=2.0, initial_delay=1.0),
            ModelRetryMiddleware(max_retries=3, backoff_factor=2.0, initial_delay=1.0),
            ModelFallbackMiddleware(get_small_model()),  # 主模型失败时依次尝试这些模型
        ],
    )
    print(last_text(_run(agent, "Run the SQL 'select fail' with database_tool and tell me what happened.")))


# ---------------------------------------------------------------------------
# 上下文管理：摘要、清理旧工具结果
# ---------------------------------------------------------------------------
# 【文档原代码·有改动】middleware/built-in · Summarization：合并 Summarization 和 Context editing；模型改成 get_model()；加了 checkpointer
def demo_context_management():
    """SummarizationMiddleware + ContextEditingMiddleware"""
    agent = create_agent(
        model=get_model(),
        tools=[search],
        middleware=[
            SummarizationMiddleware(model=get_small_model(), trigger=("tokens", 4000), keep=("messages", 20)),
            # token 超过阈值时，把较早的工具结果替换成占位符，只保留最近 3 个
            ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=100000, keep=3)]),
        ],
        checkpointer=InMemorySaver(),
    )
    print(last_text(_run(agent, "Search for three facts about Wisconsin.")))


# ---------------------------------------------------------------------------
# 限流：模型调用次数、工具调用次数
# ---------------------------------------------------------------------------
# 【文档原代码·有改动】middleware/built-in · Model call limit：合并 Model call limit 和 Tool call limit；模型改成 get_model()；run_limit 改成 1
def demo_limits():
    """ModelCallLimitMiddleware / ToolCallLimitMiddleware"""
    agent = create_agent(
        model=get_model(),
        checkpointer=InMemorySaver(),  # thread_limit 需要 checkpointer
        tools=[search],
        middleware=[
            ModelCallLimitMiddleware(thread_limit=10, run_limit=5, exit_behavior="end"),
            ToolCallLimitMiddleware(thread_limit=20, run_limit=10),  # 全局
            ToolCallLimitMiddleware(tool_name="search", thread_limit=5, run_limit=1),  # 单个工具
        ],
    )
    print(last_text(_run(agent, "Search for 'a', then 'b', then 'c' as three separate searches.")))


# ---------------------------------------------------------------------------
# 人工审批（第 13 章详细讲）
# ---------------------------------------------------------------------------
# 【文档原代码·有改动】middleware/built-in · Human-in-the-loop：模型改成 get_model()；加了调用
def demo_human_in_the_loop():
    """HumanInTheLoopMiddleware：发送邮件前暂停"""
    agent = create_agent(
        model=get_model(),
        tools=[your_read_email_tool, your_send_email_tool],
        checkpointer=InMemorySaver(),
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "your_send_email_tool": {"allowed_decisions": ["approve", "edit", "reject"]},
                    "your_read_email_tool": False,
                }
            ),
        ],
    )
    result = _run(agent, "Read email 7 and then send a reply to bob@example.com.", version="v2")
    print("interrupts:", result.interrupts)


# ---------------------------------------------------------------------------
# PII 检测
# ---------------------------------------------------------------------------
# 【文档原代码·有改动】middleware/built-in · Custom PII types：正则不再单独存变量
def detect_ssn(content: str) -> list[PIIMatch]:
    """Detect SSNs with validation."""
    matches: list[PIIMatch] = []
    for match in re.finditer(r"\d{3}-\d{2}-\d{4}", content):
        ssn = match.group(0)
        first_three = int(ssn[:3])
        if first_three not in [0, 666] and not (900 <= first_three <= 999):
            matches.append({"type": "ssn", "value": ssn, "start": match.start(), "end": match.end()})
    return matches


# 【文档原代码·有改动】middleware/built-in · Custom PII types：几段合并；模型改成 get_model()；re.compile 改成字符串（新版不接受 Pattern）
def demo_pii():
    """内置类型（email、credit_card）+ 自定义正则 / 函数；策略 redact / mask / hash / block"""
    agent = create_agent(
        model=get_model(),
        tools=[],
        middleware=[
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
            # 文档里也给了 re.compile(...) 写法，但当前版本（langchain 1.4.7）只接受
            # 正则字符串或函数，传编译好的 Pattern 会报 TypeError
            PIIMiddleware(
                "phone_number",
                detector=r"\+?\d{1,3}[\s.-]?\d{3,4}[\s.-]?\d{4}",
                strategy="mask",
            ),
            PIIMiddleware("ssn", detector=detect_ssn, strategy="hash"),
        ],
    )
    result = _run(agent, "My email is john.doe@example.com, card 5105-1051-0510-5100, SSN 123-45-6789. Repeat them back.")
    for m in result["messages"]:
        print(f"[{m.type}] {m.text}")

    blocker = create_agent(
        model=get_model(),
        tools=[],
        middleware=[PIIMiddleware("api_key", detector=r"sk-[a-zA-Z0-9]{32}", strategy="block", apply_to_input=True)],
    )
    try:
        _run(blocker, "my key is sk-" + "a" * 32)
    except Exception as e:  # strategy="block" 会直接抛错
        print("blocked:", type(e).__name__, e)


# ---------------------------------------------------------------------------
# 规划、工具筛选、工具模拟
# ---------------------------------------------------------------------------
# 【文档原代码·有改动】middleware/built-in · To-do list：合并 To-do list / LLM tool selector / LLM tool emulator；模型改成 get_model()；工具和问题是补写
def demo_planning_and_tool_selection():
    """TodoListMiddleware / LLMToolSelectorMiddleware / LLMToolEmulator"""
    agent = create_agent(model=get_model(), tools=[search], middleware=[TodoListMiddleware()])
    result = _run(agent, "Make a 3-step plan to learn LangChain, then execute step 1.")
    print("todos:", result.get("todos"))

    if FAKE:
        print("LLMToolSelector / LLMToolEmulator 需要真实模型，跳过。")
        return
    # 工具很多时，先让小模型挑出最相关的几个，再交给主模型
    agent = create_agent(
        model=get_model(),
        tools=[search, database_tool, get_weather, your_read_email_tool, your_send_email_tool],
        middleware=[LLMToolSelectorMiddleware(model=get_small_model(), max_tools=2, always_include=["search"])],
    )
    print(last_text(_run(agent, "What's the weather in Madison?")))

    # 不真正执行工具，而是让 LLM 编造工具结果（开发阶段做原型很方便）
    agent = create_agent(model=get_model(), tools=[get_weather, your_send_email_tool], middleware=[LLMToolEmulator()])
    print(last_text(_run(agent, "What's the weather in Tokyo?")))


# ---------------------------------------------------------------------------
# 文件与 Shell
# ---------------------------------------------------------------------------
# 【文档原代码·有改动】middleware/built-in · File search：合并 File search 和 Shell tool；模型改成 get_model()；用临时目录；use_ripgrep 改成 False
def demo_file_search_and_shell():
    """FilesystemFileSearchMiddleware（glob/grep）与 ShellToolMiddleware"""
    workspace = tempfile.mkdtemp(prefix="lc_workspace_")
    with open(f"{workspace}/notes.txt", "w") as f:
        f.write("TODO: finish chapter 10\n")

    agent = create_agent(
        model=get_model(),
        tools=[],
        middleware=[FilesystemFileSearchMiddleware(root_path=workspace, use_ripgrep=False)],
    )
    print(last_text(_run(agent, "Find any TODO in the workspace files.")))

    # 警告：HostExecutionPolicy 会在本机直接执行模型生成的命令，只在可信环境使用。
    # 更安全的选择：DockerExecutionPolicy / CodexSandboxExecutionPolicy。
    agent = create_agent(
        model=get_model(),
        tools=[],
        middleware=[ShellToolMiddleware(workspace_root=workspace, execution_policy=HostExecutionPolicy())],
    )
    print(last_text(_run(agent, "List the files in the workspace with ls.")))


# ---------------------------------------------------------------------------
# deepagents 提供的中间件（文件系统、子 agent）
# ---------------------------------------------------------------------------
# 【文档原代码·有改动】middleware/built-in · Short-term vs. long-term filesystem：合并 Filesystem 和 Subagent；模型改成 get_model()；StoreBackend 补了 namespace（新版必须传）
def demo_deepagents_middleware():
    """FilesystemMiddleware（CompositeBackend：/memories/ 存到 store）+ SubAgentMiddleware"""
    from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
    from deepagents.middleware import FilesystemMiddleware
    from deepagents.middleware.subagents import SubAgentMiddleware
    from langgraph.store.memory import InMemoryStore

    agent = create_agent(
        model=get_model(),
        store=InMemoryStore(),
        middleware=[
            FilesystemMiddleware(
                # /memories/ 下的文件跨对话保存在 store，其他文件只在本次 state 中
                # 新版 StoreBackend 必须指定 namespace（按用户/agent 隔离存储）
                backend=CompositeBackend(
                    default=StateBackend(),
                    routes={"/memories/": StoreBackend(namespace=lambda rt: ("demo-user", "filesystem"))},
                ),
                custom_tool_descriptions={"ls": "Use the ls tool when you need to see existing files."},
            ),
            SubAgentMiddleware(
                backend=StateBackend(),
                subagents=[
                    {
                        "name": "weather",
                        "description": "This subagent can get weather in cities.",
                        "system_prompt": "Use the get_weather tool to get the weather in a city.",
                        "tools": [get_weather],
                        "model": get_model(),
                        "middleware": [],
                    }
                ],
            ),
        ],
    )
    print(last_text(_run(agent, "Ask the weather subagent about Oslo and save the answer to /memories/oslo.txt")))


# ---------------------------------------------------------------------------
# 在 LangGraph 工作流里嵌入带中间件的 agent
# ---------------------------------------------------------------------------
# 【文档原代码·有改动】middleware/overview · Use middleware inside a LangGraph workflow：模型改成 get_model()；工具名和调用有调整
def demo_agent_inside_langgraph():
    """把 create_agent 的结果当作 StateGraph 的一个节点"""
    email_agent = create_agent(
        model=get_model(),
        tools=[your_read_email_tool, your_send_email_tool],
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"your_send_email_tool": True})],
    )

    # 【补写】文档写的是 assumes defined elsewhere
    def classify_node(state: AgentState) -> dict:
        print("[classify] 收到", len(state["messages"]), "条消息")
        return {}

    # 【补写】文档写的是 assumes defined elsewhere
    def route(state: AgentState) -> str:
        return "email_agent"

    graph = (
        StateGraph(AgentState)
        .add_node("classify", classify_node)
        .add_node("email_agent", email_agent)
        .add_edge(START, "classify")
        .add_conditional_edges("classify", route)
        .compile(checkpointer=InMemorySaver())
    )
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "Read email 3 and summarize it."}]},
        config={"configurable": {"thread_id": "graph"}},
    )
    print(result["messages"][-1].text)


# 【补写】命令行入口
if __name__ == "__main__":
    run_demos(globals())
