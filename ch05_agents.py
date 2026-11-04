"""第 5 章 · Agents：create_agent 的全部参数一览。

文档：https://docs.langchain.com/oss/python/langchain/agents

create_agent(
    model,            # 模型（字符串或模型对象）
    tools,            # 工具列表
    system_prompt,    # 系统提示词
    response_format,  # 结构化输出（第 8 章）
    state_schema,     # 自定义 state（第 6 章）
    context_schema,   # 运行时 context（第 9 章）
    checkpointer,     # 短期记忆（第 6 章）
    store,            # 长期记忆（第 14 章）
    middleware,       # 中间件（第 10、11 章）
    name,             # 名字（多 agent 时区分来源）
)

代码来源标注（每个类、函数和主要常量前面都有一行）：
  【文档原代码】          和文档的代码完全一致（只比较语法，不比较注释和排版）
  【文档原代码·有改动】   来自文档，做了改动（标注里写了改了什么）
  【补写】                文档没有给出这段代码，是我写的
来源写成"页面 · 小节"，页面相对于 https://docs.langchain.com/oss/python/langchain/
注意：overview 页只有 get_weather 这一个例子，其余代码来自它链接的其他页面。
"""

from dataclasses import dataclass

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelRetryMiddleware,
    PIIMiddleware,
    TodoListMiddleware,
    ToolRetryMiddleware,
)
from langchain.messages import AIMessage, HumanMessage
from langchain.tools import tool
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
from typing_extensions import NotRequired

from common import FAKE, get_model, last_text, run_demos


# 【文档原代码】agents · Planning and delegation
@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"


# 【文档原代码·有改动】agents · System prompt：合并 Tools / System prompt / Name your agent；模型改成 get_model()；加了调用
def demo_core_components():
    """模型 + 工具 + 系统提示词"""
    agent = create_agent(
        model=get_model(),
        tools=[search],
        system_prompt="You are a helpful assistant. Be concise and accurate.",
        name="research_assistant",  # 起个名字，多 agent 流式输出时能看出是谁在说话
    )
    print(last_text(agent.invoke({"messages": [{"role": "user", "content": "Search for LangChain news"}]})))


# 【文档原代码】agents · Structured output
class Answer(BaseModel):
    summary: str
    confidence: float


# 【文档原代码·有改动】agents · Structured output：模型改成 get_model()；tools 改成 [search]；加了假模型跳过
def demo_structured_output():
    """response_format：结果出现在 result["structured_response"]"""
    if FAKE:
        print("假模型不支持结构化输出，跳过。")
        return
    agent = create_agent(model=get_model(), tools=[search], response_format=Answer)
    result = agent.invoke({"messages": [{"role": "user", "content": "Summarize AI trends"}]})
    print(result["structured_response"])


# 【文档原代码·有改动】agents · Agent state：字段改成 NotRequired（可以不传）
class MyState(AgentState):
    user_id: NotRequired[str]
    call_count: NotRequired[int]


# 【文档原代码·有改动】agents · Agent state：模型改成 get_model()；加了调用和 print
def demo_agent_state():
    """state_schema：在默认的 messages 之外加自定义字段"""
    agent = create_agent(model=get_model(), tools=[], state_schema=MyState)
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "user-123",
        "call_count": 0,
    })
    print({k: v for k, v in result.items() if k != "messages"})


# 【文档原代码】agents · Invocation
@dataclass
class Context:
    user_id: str


# 【文档原代码·有改动】agents · Invocation：合并两段示例；模型改成 get_model()；问题改了
def demo_invocation():
    """thread_id 让多轮对话共享历史；context 传入每次调用的只读信息"""
    agent = create_agent(
        model=get_model(),
        tools=[],
        context_schema=Context,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": str(uuid7())}}
    for question in ("My favourite city is Madison.", "What is my favourite city?"):
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config=config,
            context=Context(user_id="user-123"),
        )
        print(f"> {question}\n{last_text(result)}")


# 【文档原代码·有改动】agents · Streaming：文档只给了循环，agent 定义是补写；tool_calls 用 getattr 读取
def demo_streaming():
    """stream_events(version="v3")：每一步的完整 state 快照"""
    agent = create_agent(model=get_model(), tools=[search])
    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": "Search for AI news and summarize the findings"}]},
        version="v3",
    )
    for snapshot in stream.values:
        latest_message = snapshot["messages"][-1]
        if latest_message.content:
            if isinstance(latest_message, HumanMessage):
                print(f"User: {latest_message.content}")
            elif isinstance(latest_message, AIMessage):
                print(f"Agent: {latest_message.content}")
        elif getattr(latest_message, "tool_calls", None):
            print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")


# 【补写】把 agents 页几个 Configure the harness 中间件组合起来
def demo_configure_harness():
    """用中间件组装 harness：容错、护栏、人工审批、任务规划"""
    agent = create_agent(
        model=get_model(),
        tools=[search],
        middleware=[
            ModelRetryMiddleware(max_retries=3),  # 模型调用失败自动重试
            ToolRetryMiddleware(max_retries=2),  # 工具调用失败自动重试
            PIIMiddleware("email", strategy="redact", apply_to_input=True),  # 脱敏邮箱
            TodoListMiddleware(),  # 给 agent 一个 write_todos 工具做计划
            HumanInTheLoopMiddleware(interrupt_on={"search": False}),  # 这里设为不拦截
        ],
        checkpointer=InMemorySaver(),
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "My email is a@b.com. Plan and search for agent frameworks."}]},
        config={"configurable": {"thread_id": "harness"}},
    )
    print(last_text(result))
    print("todos:", result.get("todos"))


# 【文档原代码·有改动】agents · Planning and delegation：模型改成 get_model()；子 agent 模型同样改了；加了调用
def demo_deepagents_harness():
    """deepagents 中间件：虚拟文件系统 + 子 agent（需要 pip install deepagents）"""
    from deepagents.backends import StateBackend
    from deepagents.middleware import FilesystemMiddleware
    from deepagents.middleware.subagents import SubAgentMiddleware

    backend = StateBackend()
    model = get_model()
    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[
            FilesystemMiddleware(backend=backend),  # ls / read_file / write_file / edit_file ...
            TodoListMiddleware(),
            SubAgentMiddleware(
                backend=backend,
                subagents=[
                    {
                        "name": "researcher",
                        "description": "Searches and returns a structured summary.",
                        "system_prompt": "Use the search tool to research the question and summarize key points.",
                        "tools": [search],
                        "model": model,
                        "middleware": [],
                    }
                ],
            ),
        ],
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "Research LangGraph and save notes to /notes.md"}]})
    print(last_text(result))
    print("虚拟文件：", list(result.get("files", {}).keys()))


# 【补写】命令行入口
if __name__ == "__main__":
    run_demos(globals())
