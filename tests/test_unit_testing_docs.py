"""第 20 章（上）· Unit testing：文档里的例子原样变成测试。

文档：https://docs.langchain.com/oss/python/langchain/test/unit-testing
运行：pytest tests/test_unit_testing_docs.py -v
"""

import pytest
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, ToolCall
from langchain.tools import tool
from langchain_core.language_models import FakeListChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langgraph.checkpoint.memory import InMemorySaver


def test_generic_fake_chat_model():
    model = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[ToolCall(name="foo", args={"bar": "baz"}, id="call_1")]),
        "bar",
    ]))
    first = model.invoke("hello")
    assert first.tool_calls[0]["name"] == "foo"
    assert model.invoke("hello, again!").content == "bar"


def test_fake_list_chat_model():
    model = FakeListChatModel(responses=["Paris", "The Eiffel Tower"])
    assert model.invoke("What is the capital of France?").content == "Paris"
    assert model.invoke("What is a famous landmark there?").content == "The Eiffel Tower"


class FakeToolCallingModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"72°F and sunny in {city}"


@pytest.fixture
def weather_model():
    return FakeToolCallingModel(messages=iter([
        AIMessage(content="", tool_calls=[ToolCall(name="get_weather", args={"city": "Tokyo"}, id="call_1")]),
        AIMessage(content="It is 72°F and sunny in Tokyo."),
    ]))


def test_tool_is_called(weather_model):
    agent = create_agent(weather_model, tools=[get_weather])
    result = agent.invoke({"messages": [HumanMessage(content="Weather in Tokyo?")]})
    tool_call_messages = [m for m in result["messages"] if isinstance(m, AIMessage) and m.tool_calls]
    assert len(tool_call_messages) == 1
    assert tool_call_messages[0].tool_calls[0]["name"] == "get_weather"


def test_final_response(weather_model):
    agent = create_agent(weather_model, tools=[get_weather])
    result = agent.invoke({"messages": [HumanMessage(content="Weather in Tokyo?")]})
    assert "Tokyo" in result["messages"][-1].text


def test_checkpointer_persists_history():
    model = FakeListChatModel(responses=["Noted, you live in Sydney.", "It's GMT+10 for you."])
    agent = create_agent(model, tools=[], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "session-1"}}
    agent.invoke({"messages": [HumanMessage(content="I live in Sydney, Australia")]}, config=config)
    result = agent.invoke({"messages": [HumanMessage(content="What's my local time?")]}, config=config)
    # 第二次调用时，第一轮的消息仍在历史里
    assert any("Sydney" in m.text for m in result["messages"] if isinstance(m, HumanMessage))
    assert len(result["messages"]) == 4
