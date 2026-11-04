"""第 7 章 · Streaming：实时看到 agent 在做什么。

文档：https://docs.langchain.com/oss/python/langchain/streaming

两套 API：
- agent.stream_events(..., version="v3")：高层 API，按 messages / tool_calls / values 分类迭代
- agent.stream(..., stream_mode=..., version="v2")：底层 API，stream_mode 可选
    "updates"  每个节点执行后的增量
    "messages" LLM 的 token
    "custom"   工具里 get_stream_writer() 推送的自定义数据

代码来源标注（每个类、函数和主要常量前面都有一行）：
  【文档原代码】          和文档的代码完全一致（只比较语法，不比较注释和排版）
  【文档原代码·有改动】   来自文档，做了改动（标注里写了改了什么）
  【补写】                文档没有给出这段代码，是我写的
来源写成"页面 · 小节"，页面相对于 https://docs.langchain.com/oss/python/langchain/
注意：overview 页只有 get_weather 这一个例子，其余代码来自它链接的其他页面。
"""

from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, HumanInTheLoopMiddleware, after_agent
from langchain.messages import AIMessage, AIMessageChunk, AnyMessage, ToolMessage
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime
from langgraph.types import Command, Interrupt
from pydantic import BaseModel

from common import FAKE, MODEL, get_model, get_small_model, run_demos


# 【文档原代码】streaming · Custom updates
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()  # 推送任意自定义数据
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"


# 【文档原代码】streaming · Streaming tool calls
def _render_message_chunk(token: AIMessageChunk) -> None:
    if token.text:
        print(token.text, end="|")
    if token.tool_call_chunks:
        print(token.tool_call_chunks)


# 【文档原代码】streaming · Streaming tool calls
def _render_completed_message(message: AnyMessage) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"Tool calls: {message.tool_calls}")
    if isinstance(message, ToolMessage):
        print(f"Tool response: {message.content_blocks}")


# ---------------------------------------------------------------------------
# 高层：stream_events v3
# ---------------------------------------------------------------------------
# 【文档原代码·有改动】streaming · Agent progress：模型改成 get_model()；最后加了 print
def demo_agent_progress():
    """stream_events v3：token 与工具调用交错输出"""
    agent = create_agent(model=get_model(), tools=[get_weather], checkpointer=InMemorySaver())
    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
        config={"configurable": {"thread_id": str(uuid7())}},
        version="v3",
    )
    for kind, item in stream.interleave("messages", "tool_calls"):
        if kind == "messages":
            for token in item.text:
                print(token, end="", flush=True)
        elif kind == "tool_calls":
            print(f"\nTool call: {item.tool_name}({item.input})")
            for delta in item.output_deltas:
                print(delta, end="", flush=True)
            print(f"\nTool result: {item.output}")
    final_state = stream.output
    print(f"\n最终 state 有 {len(final_state['messages'])} 条消息")


# ---------------------------------------------------------------------------
# 底层：stream + stream_mode
# ---------------------------------------------------------------------------
# 【文档原代码·有改动】streaming · LLM tokens：模型改成 get_model()；print 合并成一行
def demo_llm_tokens():
    """stream_mode="messages"：逐 token，metadata 里有节点名"""
    agent = create_agent(model=get_model(), tools=[get_weather])
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
        stream_mode="messages",
        version="v2",
    ):
        if chunk["type"] == "messages":
            token, metadata = chunk["data"]
            print(f"node: {metadata['langgraph_node']:<6} content: {token.content_blocks}")


# 【文档原代码·有改动】streaming · Custom updates：模型改成 get_model()；去掉了 type 判断
def demo_custom_updates():
    """stream_mode="custom"：接收工具内部推送的进度"""
    agent = create_agent(model=get_model(), tools=[get_weather])
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
        stream_mode="custom",
        version="v2",
    ):
        print(chunk["data"])


# 【文档原代码·有改动】streaming · Stream multiple modes：模型改成 get_model()；print 合并
def demo_multiple_modes():
    """同时订阅多个 stream_mode，用 chunk["type"] 区分"""
    agent = create_agent(model=get_model(), tools=[get_weather])
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
        stream_mode=["updates", "custom"],
        version="v2",
    ):
        print(f"stream_mode: {chunk['type']}\ncontent: {chunk['data']}\n")


# 【文档原代码·有改动】streaming · Accessing completed messages：合并 Streaming tool calls 和本节第三段；模型改成 get_model()
def demo_tool_call_streaming():
    """流式拼接工具调用参数，并在 updates 里拿到完整消息"""
    agent = create_agent(get_model(), tools=[get_weather])
    full_message = None
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "What is the weather in Boston?"}]},
        stream_mode=["messages", "updates"],
        version="v2",
    ):
        if chunk["type"] == "messages":
            token, metadata = chunk["data"]
            if isinstance(token, AIMessageChunk):
                _render_message_chunk(token)
                full_message = token if full_message is None else full_message + token
                if token.chunk_position == "last":  # 一条消息的最后一个 chunk
                    if full_message.tool_calls:
                        print(f"\n[拼好的] Tool calls: {full_message.tool_calls}")
                    full_message = None
        elif chunk["type"] == "updates":
            for source, update in chunk["data"].items():
                if source == "tools":
                    _render_completed_message(update["messages"][-1])


# 【文档原代码·有改动】streaming · Streaming thinking / reasoning tokens：模型名取自 .env；非 Anthropic 模型直接返回
def demo_reasoning_tokens():
    """思考（reasoning）token：Anthropic extended thinking 示例"""
    if not MODEL.startswith(("anthropic:", "claude")):
        print("这个 demo 需要 Anthropic 模型（LC_MODEL=anthropic:claude-sonnet-4-6）。")
        return
    from langchain_anthropic import ChatAnthropic

    model = ChatAnthropic(
        model_name=MODEL.removeprefix("anthropic:"),
        timeout=None,
        stop=None,
        thinking={"type": "enabled", "budget_tokens": 5000},
    )
    agent = create_agent(model=model, tools=[get_weather])
    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": "What is the weather in SF?"}]}, version="v3"
    )
    for message in stream.messages:
        for token in message.reasoning:
            print(f"[thinking] {token}", end="")
        for token in message.text:
            print(token, end="", flush=True)


# ---------------------------------------------------------------------------
# 常见组合：护栏结果、人工审批、子 agent
# ---------------------------------------------------------------------------
# 【文档原代码】streaming · Accessing completed messages
class ResponseSafety(BaseModel):
    """Evaluate a response as safe or unsafe."""

    evaluation: Literal["safe", "unsafe"]


# 【文档原代码·有改动】streaming · Accessing completed messages：模型改成 get_small_model()；加了假模型分支；合并判断和变量
@after_agent(can_jump_to=["end"])
def safety_guardrail(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Model-based guardrail: Use an LLM to evaluate response safety."""
    stream_writer = get_stream_writer()
    if not state["messages"] or not isinstance(state["messages"][-1], AIMessage):
        return None
    last_message = state["messages"][-1]
    if FAKE:
        stream_writer("（假模型模式跳过安全评估）")
        return None
    result = get_small_model().bind_tools([ResponseSafety], tool_choice="any").invoke([
        {"role": "system", "content": "Evaluate this AI response as generally safe or unsafe."},
        {"role": "user", "content": f"AI response: {last_message.text}"},
    ])
    stream_writer(result)  # 把评估结果也推到流里
    if result.tool_calls[0]["args"]["evaluation"] == "unsafe":
        last_message.content = "I cannot provide that response. Please rephrase your request."
    return None


# 【文档原代码·有改动】streaming · Accessing completed messages：模型改成 get_model()；custom 分支打印内容改了
def demo_guardrail_in_stream():
    """在流里同时拿到 token、完整消息和护栏评估结果"""
    agent = create_agent(model=get_model(), tools=[get_weather], middleware=[safety_guardrail])
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "What is the weather in Boston?"}]},
        stream_mode=["messages", "updates", "custom"],
        version="v2",
    ):
        if chunk["type"] == "messages":
            token, _ = chunk["data"]
            if isinstance(token, AIMessageChunk):
                _render_message_chunk(token)
        elif chunk["type"] == "updates":
            for source, update in chunk["data"].items():
                if source in ("model", "tools"):
                    _render_completed_message(update["messages"][-1])
        elif chunk["type"] == "custom":
            print(f"\n[guardrail] {chunk['data']}")


# 【文档原代码·有改动】streaming · Streaming with human-in-the-loop：去掉中间变量 interrupts
def _render_interrupt(interrupt: Interrupt) -> None:
    for request in interrupt.value["action_requests"]:
        print(request["description"])


# 【文档原代码·有改动】streaming · Streaming with human-in-the-loop：三段合并；模型改成 get_model()；没有中断时不恢复
def demo_stream_with_hitl():
    """流式 + 人工审批：先流到中断处，再用 Command(resume=...) 继续流"""
    agent = create_agent(
        get_model(),
        tools=[get_weather],
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"get_weather": True})],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "some_id"}}

    # 【文档原代码·有改动】streaming · Streaming with human-in-the-loop：把文档重复两次的流式循环封成函数
    def run(payload) -> list[Interrupt]:
        interrupts = []
        for chunk in agent.stream(payload, config=config, stream_mode=["messages", "updates"], version="v2"):
            if chunk["type"] == "messages":
                token, _ = chunk["data"]
                if isinstance(token, AIMessageChunk):
                    _render_message_chunk(token)
            elif chunk["type"] == "updates":
                for source, update in chunk["data"].items():
                    if source in ("model", "tools"):
                        _render_completed_message(update["messages"][-1])
                    if source == "__interrupt__":
                        interrupts.extend(update)
                        _render_interrupt(update[0])
        return interrupts

    interrupts = run({"messages": [{"role": "user", "content": "Can you look up the weather in Boston and San Francisco?"}]})

    # 每个中断对应一组决策：Boston 改参数，其余直接批准
    decisions = {}
    for interrupt in interrupts:
        decisions[interrupt.id] = {
            "decisions": [
                {"type": "edit", "edited_action": {"name": "get_weather", "args": {"city": "Boston, U.K."}}}
                if "boston" in request["description"].lower()
                else {"type": "approve"}
                for request in interrupt.value["action_requests"]
            ]
        }
    if decisions:
        print("\n--- 恢复执行 ---")
        run(Command(resume=decisions))


# 【文档原代码·有改动】streaming · Streaming from sub-agents：两段合并；模型改成 get_model()
def demo_subagent_streaming():
    """subgraphs=True + lc_agent_name：区分主 agent 和子 agent 的 token"""
    weather_agent = create_agent(model=get_model(), tools=[get_weather], name="weather_agent")

    # 【文档原代码】streaming · Streaming from sub-agents
    def call_weather_agent(query: str) -> str:
        """Query the weather agent."""
        result = weather_agent.invoke({"messages": [{"role": "user", "content": query}]})
        return result["messages"][-1].text

    agent = create_agent(model=get_model(), tools=[call_weather_agent], name="supervisor")
    current_agent = None
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "What is the weather in Boston?"}]},
        stream_mode=["messages", "updates"],
        subgraphs=True,
        version="v2",
    ):
        if chunk["type"] == "messages":
            token, metadata = chunk["data"]
            if (agent_name := metadata.get("lc_agent_name")) and agent_name != current_agent:
                print(f"\n🤖 {agent_name}: ")
                current_agent = agent_name
            if isinstance(token, AIMessageChunk):
                _render_message_chunk(token)
        elif chunk["type"] == "updates":
            for source, update in chunk["data"].items():
                if source in ("model", "tools"):
                    _render_completed_message(update["messages"][-1])


# 【文档原代码·有改动】streaming · v2 streaming format：模型改成 get_model()；print 内容有调整
def demo_v2_invoke():
    """invoke(version="v2") 返回 GraphOutput：.value 与 .interrupts"""
    agent = create_agent(model=get_model(), tools=[get_weather])
    result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]}, version="v2")
    print(type(result).__name__, "| interrupts:", result.interrupts)
    print(result.value["messages"][-1].text)


# 【补写】命令行入口
if __name__ == "__main__":
    run_demos(globals())
