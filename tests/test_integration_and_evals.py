"""第 20 章（下）· Integration testing + Agent evals：用真实模型测试。

文档：
- https://docs.langchain.com/oss/python/langchain/test/integration-testing
- https://docs.langchain.com/oss/python/langchain/test/evals

这些测试会调用真实 API（花钱），默认跳过。运行方式：
    LC_MODEL=openai:gpt-5.4-mini pytest -m integration -v
原则：断言"结构"（调用了哪个工具），不要断言模型的原话。

代码来源标注（每个类、函数和主要常量前面都有一行）：
  【文档原代码】          和文档的代码完全一致（只比较语法，不比较注释和排版）
  【文档原代码·有改动】   来自文档，做了改动（标注里写了改了什么）
  【补写】                文档没有给出这段代码，是我写的
来源写成"页面 · 小节"，页面相对于 https://docs.langchain.com/oss/python/langchain/
注意：overview 页只有 get_weather 这一个例子，其余代码来自它链接的其他页面。
"""

import os

import pytest
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain.tools import tool

pytestmark = pytest.mark.integration

MODEL = os.environ.get("LC_MODEL", "fake")


# 【文档原代码·有改动】test/integration-testing · Manage API keys：检查的条件从 API key 改成 LC_MODEL
@pytest.fixture(autouse=True)
def require_real_model():
    if MODEL == "fake":
        pytest.skip("设置 LC_MODEL=<供应商:模型> 并配置对应 API key 后再运行")


# 【文档原代码·有改动】test/evals · Trajectory match evaluator：加了返回类型
@tool
def get_weather(city: str) -> str:
    """Get weather information for a city."""
    return f"It's 75 degrees and sunny in {city}."


# 【补写】把文档里的列表推导抽成函数
def _tool_calls(messages) -> list[dict]:
    return [tc for msg in messages if isinstance(msg, AIMessage) for tc in (msg.tool_calls or [])]


# 【文档原代码·有改动】test/integration-testing · Separate unit and integration tests：模型名改成 LC_MODEL；marker 放到模块级
def test_agent_with_real_model():
    agent = create_agent(MODEL, tools=[get_weather])
    result = agent.invoke({"messages": [HumanMessage(content="What's the weather in SF?")]})
    assert len(result["messages"]) > 1


# 【文档原代码·有改动】test/integration-testing · Assert on structure, not content：模型名改成 LC_MODEL；列表推导改用 _tool_calls；content 改成 .text
def test_agent_calls_weather_tool():
    agent = create_agent(MODEL, tools=[get_weather])
    messages = agent.invoke({"messages": [HumanMessage(content="What's the weather in SF?")]})["messages"]
    assert any(tc["name"] == "get_weather" for tc in _tool_calls(messages))
    assert isinstance(messages[-1], AIMessage)
    assert len(messages[-1].text) > 0


# 【文档原代码·有改动】test/evals · Trajectory match evaluator：文档这里只给了开头，评估器调用和参考轨迹是补写
def test_trajectory_match():
    """agentevals：把真实轨迹与参考轨迹比较（pip install agentevals）"""
    match = pytest.importorskip("agentevals.trajectory.match")
    evaluator = match.create_trajectory_match_evaluator(trajectory_match_mode="superset")

    agent = create_agent(MODEL, tools=[get_weather])
    result = agent.invoke({"messages": [HumanMessage(content="What's the weather in San Francisco?")]})
    reference_trajectory = [
        HumanMessage(content="What's the weather in San Francisco?"),
        AIMessage(content="", tool_calls=[{"id": "call_1", "name": "get_weather", "args": {"city": "San Francisco"}}]),
        ToolMessage(content="It's 75 degrees and sunny in San Francisco.", tool_call_id="call_1"),
        AIMessage(content="The weather in San Francisco is 75 degrees and sunny."),
    ]
    evaluation = evaluator(outputs=result["messages"], reference_outputs=reference_trajectory)
    assert evaluation["score"] is True
