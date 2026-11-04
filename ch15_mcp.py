"""第 15 章 · MCP（Model Context Protocol）：把外部 MCP 服务器的工具交给 agent。

文档：
- https://docs.langchain.com/oss/python/langchain/mcp/index
- https://docs.langchain.com/oss/python/langchain/mcp/tools
- https://docs.langchain.com/oss/python/langchain/mcp/connections

安装：pip install "langchain[mcp]"（自带 FastMCP）
MCPAdapter(target) 的 target 可以是：
    FastMCP 实例（进程内，适合测试） / Path("server.py")（stdio 子进程） /
    "https://..."（HTTP） / {"mcpServers": {...}}（多服务器配置）
所有 MCP 相关调用都是异步的。

代码来源标注（每个类、函数和主要常量前面都有一行）：
  【文档原代码】          和文档的代码完全一致（只比较语法，不比较注释和排版）
  【文档原代码·有改动】   来自文档，做了改动（标注里写了改了什么）
  【补写】                文档没有给出这段代码，是我写的
来源写成"页面 · 小节"，页面相对于 https://docs.langchain.com/oss/python/langchain/
注意：overview 页只有 get_weather 这一个例子，其余代码来自它链接的其他页面。
"""

import sys
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig
from langchain.mcp import MCPAdapter
from langchain.messages import ToolMessage
from langchain.tools import BaseTool
from langchain.tools.tool_node import ToolCallRequest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from ch15_mcp_server import server
from common import get_model, run_demos

SERVER_SCRIPT = Path(__file__).with_name("ch15_mcp_server.py")


# 【补写】生成消息的小工具
def _ask(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


# 【文档原代码·有改动】mcp/index · Quickstart：服务器改成本项目的 FastMCP 实例；模型改成 get_model()；问题来自 mcp/tools
async def demo_quickstart():
    """进程内连接 FastMCP 服务器，列出工具并交给 agent"""
    async with MCPAdapter(server) as adapter:
        tools = await adapter.list_tools()
        print("MCP 工具：", [t.name for t in tools])
        agent = create_agent(get_model(), tools)
        result = await agent.ainvoke(_ask("What is the forecast for Oslo?"))
    print(result["messages"][-1].text)


# 【补写】写法参考 mcp/index · Transports
async def demo_stdio_transport():
    """stdio：每个 adapter 启动一个子进程运行服务器脚本"""
    async with MCPAdapter(SERVER_SCRIPT) as adapter:
        tools = await adapter.list_tools()
        print("通过 stdio 拿到的工具：", [t.name for t in tools])


# 【文档原代码·有改动】mcp/tools · Structured content：合并 Multimodal / Structured / Errors；模型改成 get_model()；问题改了
async def demo_tool_outputs():
    """结构化内容（artifact）与服务器错误（status="error"）"""
    async with MCPAdapter(server) as adapter:
        tools = await adapter.list_tools()
        agent = create_agent(get_model(), tools)
        result = await agent.ainvoke(_ask("Look up user 42, then use the divide tool to calculate 10 divided by 0."))
    for message in result["messages"]:
        if not isinstance(message, ToolMessage):
            continue
        if message.status == "error":
            print("Tool reported error:", message.text)
        elif message.artifact is not None:
            print("Structured content:", message.artifact["structured_content"])
        for block in message.content_blocks:
            if block["type"] == "image":
                print("Image:", block.get("mime_type"))


# 【文档原代码】mcp/tools · Tool metadata
def is_destructive(tool: BaseTool) -> bool:
    """Read the MCP destructive hint off the adapter's tool metadata."""
    annotations = (tool.metadata or {}).get("mcp", {}).get("tool", {}).get("annotations", {})
    return annotations.get("destructive_hint", False)


# 【文档原代码·有改动】mcp/tools · Human-in-the-loop：模型改成 get_model()；interrupt_on 内联；调用和恢复来自文档下一段，改用 version="v2"
async def demo_gate_destructive_tools():
    """读取 MCP 工具的 destructive_hint，自动给危险工具加人工审批"""
    async with MCPAdapter(server) as adapter:
        tools = await adapter.list_tools()
        for t in tools:
            print(f"{t.name:<14} destructive={is_destructive(t)}")

        destructive = {t.name for t in tools if is_destructive(t)}

        # 【文档原代码】mcp/tools · Human-in-the-loop
        def needs_approval(request: ToolCallRequest) -> bool:
            return request.tool_call["name"] in destructive

        gate = InterruptOnConfig(allowed_decisions=["approve", "reject"], when=needs_approval)
        agent = create_agent(
            get_model(),
            tools,
            middleware=[HumanInTheLoopMiddleware(interrupt_on={t.name: gate for t in tools})],
            checkpointer=InMemorySaver(),
        )
        config: Any = {"configurable": {"thread_id": "mcp-hitl"}}
        result = await agent.ainvoke(_ask("Delete record 7."), config, version="v2")
        if result.interrupts:
            print("需要审批：", result.interrupts[0].value["action_requests"])
            result = await agent.ainvoke(Command(resume={"decisions": [{"type": "approve"}]}), config, version="v2")
        print(result.value["messages"][-1].text)


# 【文档原代码·有改动】mcp/tools · Elicitation：模型改成 get_model()；加了没有中断时的分支和 print
async def demo_elicitation():
    """服务器中途向用户提问 → interrupt → 用 responses 答复后继续"""
    async with MCPAdapter(server) as adapter:
        tools = await adapter.list_tools()
        agent = create_agent(get_model(), tools, checkpointer=InMemorySaver())
        config: Any = {"configurable": {"thread_id": "booking-1"}}
        paused = await agent.ainvoke(_ask("Book a table for 4."), config)
        if "__interrupt__" not in paused:
            print(paused["messages"][-1].text)
            return
        [interrupt] = paused["__interrupt__"]
        [question] = interrupt.value["requests"]
        print("服务器提问：", question)
        answer = {"action": "accept", "content": {"date": "2026-09-14"}}
        result = await agent.ainvoke(Command(resume={"responses": {question["key"]: answer}}), config)
        print(result["messages"][-1].text)


# 【文档原代码·有改动】mcp/connections · One aggregate connection with `MCPConfig`：两个配置都指向本项目的服务器；用 sys.executable；不建 agent，只列工具
async def demo_multiple_servers():
    """mcpServers 配置：多个服务器，工具名自动加上服务器前缀"""
    config = {
        "mcpServers": {
            "weather": {"command": sys.executable, "args": [str(SERVER_SCRIPT)]},
            "calc": {"command": sys.executable, "args": [str(SERVER_SCRIPT)]},
        }
    }
    async with MCPAdapter(config) as adapter:
        tools = await adapter.list_tools()
        print([t.name for t in tools])  # weather_get_forecast, calc_divide, ...


# 【补写】命令行入口
if __name__ == "__main__":
    run_demos(globals())
