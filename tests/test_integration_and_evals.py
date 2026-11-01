"""第 20 章（下）· Integration testing + Agent evals：用真实模型测试。

文档：
- https://docs.langchain.com/oss/python/langchain/test/integration-testing
- https://docs.langchain.com/oss/python/langchain/test/evals

这些测试会调用真实 API（花钱），默认跳过。运行方式：
    LC_MODEL=openai:gpt-5.4-mini pytest -m integration -v
原则：断言"结构"（调用了哪个工具），不要断言模型的原话。
"""

import os

import pytest
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain.tools import tool

pytestmark = pytest.mark.integration

MODEL = os.environ.get("LC_MODEL", "fake")


@pytest.fixture(autouse=True)
def require_real_model():
    if MODEL == "fake":
        pytest.skip("设置 LC_MODEL=<供应商:模型> 并配置对应 API key 后再运行")


@tool
def get_weather(city: str) -> str:
    """Get weather information for a city."""
    return f"It's 75 degrees and sunny in {city}."


def _tool_calls(messages) -> list[dict]:
    return [tc for msg in messages if isinstance(msg, AIMessage) for tc in (msg.tool_calls or [])]


def test_agent_with_real_model():
    agent = create_agent(MODEL, tools=[get_weather])
    result = agent.invoke({"messages": [HumanMessage(content="What's the weather in SF?")]})
    assert len(result["messages"]) > 1


def test_agent_calls_weather_tool():
    agent = create_agent(MODEL, tools=[get_weather])
    messages = agent.invoke({"messages": [HumanMessage(content="What's the weather in SF?")]})["messages"]
    assert any(tc["name"] == "get_weather" for tc in _tool_calls(messages))
    assert isinstance(messages[-1], AIMessage)
    assert len(messages[-1].text) > 0


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
