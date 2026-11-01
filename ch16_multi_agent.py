"""第 16 章 · Multi-agent：多个 agent 协作的五种模式。

文档：https://docs.langchain.com/oss/python/langchain/multi-agent/index
- Subagents：主 agent 把子 agent 当工具调用（集中控制）
- Handoffs：按状态切换"当前负责的 agent"（对话式交接）
- Skills：一个 agent 按需加载专门的提示词（轻量）
- Router：先分类，再并行分发给多个专门 agent，最后汇总
- Custom workflow：用 LangGraph 自己编排确定性流程 + agent 节点
"""

import operator
from collections.abc import Callable
from enum import Enum
from typing import Annotated, Literal

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.messages import AIMessage, SystemMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send
from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict

from common import FAKE, get_model, last_text, run_demos


@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"[search] Top result for '{query}': LangGraph is a low-level agent orchestration framework."


# ---------------------------------------------------------------------------
# 1. Subagents
# ---------------------------------------------------------------------------
def demo_subagents_tool_per_agent():
    """每个子 agent 包装成一个工具"""
    research_agent = create_agent(
        model=get_model(), tools=[web_search], system_prompt="You research topics using web_search.", name="research"
    )

    @tool("research", description="Research a topic and return findings")
    def call_research_agent(query: str) -> str:
        result = research_agent.invoke({"messages": [{"role": "user", "content": query}]})
        return result["messages"][-1].text

    main_agent = create_agent(model=get_model(), tools=[call_research_agent], name="main")
    print(last_text(main_agent.invoke({"messages": [{"role": "user", "content": "What is LangGraph? Use research."}]})))


class AgentName(str, Enum):
    RESEARCH = "research"
    WRITER = "writer"
    REVIEWER = "reviewer"


def demo_subagents_dispatch_tool():
    """单个 task 工具 + 枚举约束，按名字派发给子 agent"""
    model = get_model()
    subagents = {
        AgentName.RESEARCH: create_agent(model, tools=[web_search], system_prompt="Research and fact-finding."),
        AgentName.WRITER: create_agent(model, tools=[], system_prompt="Content creation and editing."),
        AgentName.REVIEWER: create_agent(model, tools=[], system_prompt="Code and document review."),
    }

    @tool
    def task(agent_name: AgentName, description: str) -> str:
        """Launch an ephemeral subagent for a task."""
        result = subagents[AgentName(agent_name)].invoke({"messages": [{"role": "user", "content": description}]})
        return result["messages"][-1].text

    main_agent = create_agent(
        model=model,
        tools=[task],
        system_prompt=(
            "You coordinate specialized sub-agents. Available agents:\n"
            "- research: Research and fact-finding\n"
            "- writer: Content creation and editing\n"
            "- reviewer: Code and document review\n"
            "Use the task tool to delegate work."
        ),
    )
    print(last_text(main_agent.invoke({"messages": [{"role": "user", "content": "Write a one-line tagline for LangChain."}]})))


class SharedState(AgentState):
    example_state_key: NotRequired[str]


def demo_subagent_inputs_outputs():
    """子 agent 输入取自主 agent 的 state，输出用 Command 回写额外字段"""
    subagent1 = create_agent(model=get_model(), tools=[], state_schema=SharedState)

    @tool("subagent1_name", description="Ask the helper subagent a question.")
    def call_subagent1(query: str, runtime: ToolRuntime[None, SharedState]) -> Command:
        result = subagent1.invoke({
            "messages": [{"role": "user", "content": query}],
            "example_state_key": runtime.state.get("example_state_key", "none"),
        })
        return Command(update={
            "example_state_key": "updated by subagent",
            "messages": [ToolMessage(content=result["messages"][-1].text, tool_call_id=runtime.tool_call_id)],
        })

    main = create_agent(get_model(), tools=[call_subagent1], state_schema=SharedState)
    result = main.invoke({"messages": [{"role": "user", "content": "Ask the helper what 2+2 is."}], "example_state_key": "hello"})
    print(last_text(result), "| example_state_key =", result.get("example_state_key"))


# ---------------------------------------------------------------------------
# 2. Handoffs（A）：单个 agent + 中间件，按 current_step 切换"角色"
# ---------------------------------------------------------------------------
class SupportState(AgentState):
    current_step: NotRequired[Literal["triage", "specialist"]]
    warranty_status: NotRequired[str]


@tool
def record_warranty_status(status: str, runtime: ToolRuntime[None, SupportState]) -> Command:
    """Record warranty status (in_warranty / out_of_warranty) and transition to the specialist step."""
    return Command(update={
        "messages": [ToolMessage(content=f"Warranty status recorded: {status}", tool_call_id=runtime.tool_call_id)],
        "warranty_status": status,
        "current_step": "specialist",  # 改变 state 触发"交接"
    })


@tool
def schedule_repair(device: str) -> str:
    """Schedule a repair appointment for a device."""
    return f"Repair scheduled for {device} next Monday."


STEP_CONFIG = {
    "triage": ("You are a triage agent. Ask whether the device is under warranty, then call record_warranty_status.", ["record_warranty_status"]),
    "specialist": ("You are a repair specialist. The warranty status is {warranty_status}. Offer to schedule a repair.", ["schedule_repair"]),
}


@wrap_model_call
def step_router(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    step = request.state.get("current_step", "triage")
    prompt, tool_names = STEP_CONFIG[step]
    return handler(request.override(
        system_message=SystemMessage(prompt.format(warranty_status=request.state.get("warranty_status", "unknown"))),
        tools=[t for t in request.tools if t.name in tool_names],
    ))


def demo_handoffs_single_agent():
    """同一个 agent，不同阶段换提示词和工具"""
    agent = create_agent(
        get_model(),
        tools=[record_warranty_status, schedule_repair],
        state_schema=SupportState,
        middleware=[step_router],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "support"}}
    for text in ("My laptop screen is broken.", "Yes, it's still under warranty.", "Please book the repair."):
        result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config)
        print(f"[{result.get('current_step', 'triage')}] {last_text(result)}")


# ---------------------------------------------------------------------------
# 2. Handoffs（B）：多个 agent 子图，工具返回 Command(goto=..., graph=Command.PARENT)
# ---------------------------------------------------------------------------
class MultiAgentState(AgentState):
    active_agent: NotRequired[str]


def _make_transfer_tool(target: str):
    @tool(f"transfer_to_{target}")
    def transfer(runtime: ToolRuntime) -> Command:
        """Transfer the conversation to another agent."""
        # 只把触发交接的 AI 消息 + 一条配对的 ToolMessage 交给下一个 agent
        last_ai_message = next(m for m in reversed(runtime.state["messages"]) if isinstance(m, AIMessage))
        transfer_message = ToolMessage(content=f"Transferred to {target}", tool_call_id=runtime.tool_call_id)
        return Command(
            goto=target,
            update={"active_agent": target, "messages": [last_ai_message, transfer_message]},
            graph=Command.PARENT,  # 跳到父图里的另一个节点
        )

    transfer.description = f"Transfer the conversation to the {target}."
    return transfer


def build_handoff_graph():
    model = get_model()
    support_agent = create_agent(
        model, tools=[_make_transfer_tool("sales_agent")], name="support_agent",
        system_prompt="You are tech support. If the user wants to buy something, transfer to sales.",
    )
    sales_agent = create_agent(
        model, tools=[_make_transfer_tool("support_agent")], name="sales_agent",
        system_prompt="You are sales. Quote prices. If the user has a technical problem, transfer to support.",
    )
    return (
        StateGraph(MultiAgentState)
        .add_node("support_agent", support_agent, destinations=("sales_agent", END))
        .add_node("sales_agent", sales_agent, destinations=("support_agent", END))
        # 从上次负责的 agent 继续
        .add_conditional_edges(START, lambda s: s.get("active_agent", "support_agent"), ["support_agent", "sales_agent"])
        .compile(checkpointer=InMemorySaver())
    )


def demo_handoffs_multi_agent():
    """support_agent ⇄ sales_agent 互相交接"""
    graph = build_handoff_graph()
    config = {"configurable": {"thread_id": "handoff"}}
    result = graph.invoke({"messages": [{"role": "user", "content": "I'd like to buy the pro plan. How much is it?"}]}, config)
    print(f"[{result.get('active_agent', 'support_agent')}] {result['messages'][-1].text}")


# ---------------------------------------------------------------------------
# 3. Skills：按需加载专门的提示词
# ---------------------------------------------------------------------------
SKILLS = {
    "write_sql": "You are a SQL expert. Always use explicit JOINs and LIMIT clauses. Explain the query briefly.",
    "review_legal_doc": "You are a legal reviewer. List risky clauses as bullet points with a severity.",
}


@tool
def load_skill(skill_name: str) -> str:
    """Load a specialized skill prompt.

    Available skills:
    - write_sql: SQL query writing expert
    - review_legal_doc: Legal document reviewer

    Returns the skill's prompt and context.
    """
    return SKILLS.get(skill_name, f"Unknown skill: {skill_name}")


def demo_skills():
    """agent 先调用 load_skill 拿到专门的指令，再完成任务"""
    agent = create_agent(
        model=get_model(),
        tools=[load_skill],
        system_prompt=(
            "You are a helpful assistant. You have access to two skills: "
            "write_sql and review_legal_doc. Use load_skill to access them."
        ),
    )
    print(last_text(agent.invoke({"messages": [{"role": "user", "content": "Write SQL to get the 5 newest users."}]})))


# ---------------------------------------------------------------------------
# 4. Router：分类 → 并行分发（Send）→ 汇总
# ---------------------------------------------------------------------------
class Classification(BaseModel):
    agent: Literal["github", "notion"] = Field(description="Which knowledge source to ask")
    query: str = Field(description="The sub-question for that source")


class ClassificationList(BaseModel):
    items: list[Classification]


class RouterState(TypedDict):
    query: str
    results: Annotated[list[str], operator.add]  # 多个分支的结果用 add 合并
    final_answer: str


@tool
def search_github(query: str) -> str:
    """Search GitHub issues and code."""
    return "GitHub: issue #123 says auth tokens expire after 1 hour."


@tool
def search_notion(query: str) -> str:
    """Search internal Notion docs."""
    return "Notion: the onboarding doc says to rotate tokens via the admin panel."


def build_router():
    model = get_model()
    agents = {
        "github": create_agent(model, tools=[search_github], system_prompt="Answer using GitHub search."),
        "notion": create_agent(model, tools=[search_notion], system_prompt="Answer using Notion search."),
    }

    def classify_query(query: str) -> list[Classification]:
        """Use LLM to classify query and determine which agents to invoke."""
        if FAKE:
            return [Classification(agent="github", query=query), Classification(agent="notion", query=query)]
        return model.with_structured_output(ClassificationList).invoke(
            f"Split this question into sub-questions for 'github' and/or 'notion': {query}"
        ).items

    def route_query(state: RouterState):
        return [Send(c.agent, {"query": c.query}) for c in classify_query(state["query"])]

    def make_node(name: str):
        def node(state: dict) -> dict:
            result = agents[name].invoke({"messages": [{"role": "user", "content": state["query"]}]})
            return {"results": [f"{name}: {result['messages'][-1].text}"]}
        return node

    def synthesize(state: RouterState) -> dict:
        answer = model.invoke(f"Combine these findings into one answer for '{state['query']}':\n" + "\n".join(state["results"]))
        return {"final_answer": answer.text}

    return (
        StateGraph(RouterState)
        .add_node("github", make_node("github"))
        .add_node("notion", make_node("notion"))
        .add_node("synthesize", synthesize)
        .add_conditional_edges(START, route_query, ["github", "notion"])
        .add_edge("github", "synthesize")
        .add_edge("notion", "synthesize")
        .add_edge("synthesize", END)
        .compile()
    )


def demo_router():
    """无状态路由：一次问题并行问两个知识源"""
    workflow = build_router()
    result = workflow.invoke({"query": "How do I refresh an expired auth token?", "results": []})
    print("\n".join(result["results"]))
    print("final:", result["final_answer"])


def demo_router_as_tool():
    """有状态：把 router 包装成工具，交给一个带记忆的对话 agent"""
    workflow = build_router()

    @tool
    def search_docs(query: str) -> str:
        """Search across multiple documentation sources."""
        return workflow.invoke({"query": query, "results": []})["final_answer"]

    conversational_agent = create_agent(
        get_model(),
        tools=[search_docs],
        system_prompt="You are a helpful assistant. Use search_docs to answer questions.",
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "router"}}
    print(last_text(conversational_agent.invoke({"messages": [{"role": "user", "content": "How do I fix expired tokens?"}]}, config)))


# ---------------------------------------------------------------------------
# 5. Custom workflow：LangGraph 编排 + agent 节点
# ---------------------------------------------------------------------------
class WorkflowState(TypedDict):
    query: str
    context: str
    answer: str


def demo_custom_workflow():
    """检索（确定性步骤）→ agent 回答（智能步骤）"""
    agent = create_agent(model=get_model(), tools=[web_search])

    def retrieve(state: WorkflowState) -> dict:
        return {"context": "LangChain 1.x centers on create_agent plus middleware."}

    def agent_node(state: WorkflowState) -> dict:
        """A LangGraph node that invokes a LangChain agent."""
        prompt = f"Context: {state['context']}\n\nQuestion: {state['query']}"
        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        return {"answer": result["messages"][-1].text}

    workflow = (
        StateGraph(WorkflowState)
        .add_node("retrieve", retrieve)
        .add_node("agent", agent_node)
        .add_edge(START, "retrieve")
        .add_edge("retrieve", "agent")
        .add_edge("agent", END)
        .compile()
    )
    print(workflow.invoke({"query": "What is the core API of LangChain 1.x?"})["answer"])


def demo_deep_agent():
    """deepagents.create_deep_agent：自带规划、虚拟文件系统、子 agent、上下文压缩"""
    from deepagents import create_deep_agent

    deep_agent = create_deep_agent(
        model=get_model(),
        tools=[web_search],
        system_prompt="You are a careful researcher. Save your final notes to /report.md.",
        checkpointer=InMemorySaver(),
    )
    result = deep_agent.invoke(
        {"messages": [{"role": "user", "content": "Research LangGraph in two steps and write a short report."}]},
        config={"configurable": {"thread_id": "deep"}},
    )
    print(last_text(result))
    print("files:", list(result.get("files", {}).keys()))


if __name__ == "__main__":
    run_demos(globals())
