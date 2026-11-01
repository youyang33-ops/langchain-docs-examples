"""第 2 章 · Models：统一的聊天模型接口。

文档：https://docs.langchain.com/oss/python/langchain/models

要点：
- init_chat_model("供应商:模型名") 一行切换 OpenAI / Anthropic / Gemini / Bedrock ...
- 三种调用：invoke（一次性）、stream（流式）、batch（批量）
- bind_tools 让模型"决定"调用工具；with_structured_output 让模型输出固定结构
"""

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain.tools import tool
from langchain_core.callbacks import UsageMetadataCallbackHandler, get_usage_metadata_callback
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

from common import FAKE, MODEL, SMALL_MODEL, get_model, get_small_model, run_demos


# ---------------------------------------------------------------------------
# 初始化与参数
# ---------------------------------------------------------------------------
def demo_init():
    """init_chat_model 初始化模型并调用一次"""
    # 等价写法：from langchain_openai import ChatOpenAI; ChatOpenAI(model="gpt-5.5")
    model = get_model(
        temperature=0.7,
        timeout=30,
        max_tokens=1000,
        max_retries=6,  # 默认 6，网络不稳可调大
    )
    response = model.invoke("Why do parrots talk?")
    print(response.text)


# ---------------------------------------------------------------------------
# 调用方式：invoke / stream / batch
# ---------------------------------------------------------------------------
def demo_invoke():
    """invoke：字符串、字典列表、消息对象三种输入"""
    model = get_model()
    print(model.invoke("Why do parrots have colorful feathers?").text)

    conversation = [
        {"role": "system", "content": "You are a helpful assistant that translates English to French."},
        {"role": "user", "content": "Translate: I love programming."},
        {"role": "assistant", "content": "J'adore la programmation."},
        {"role": "user", "content": "Translate: I love building applications."},
    ]
    print(model.invoke(conversation).text)

    conversation = [
        SystemMessage("You are a helpful assistant that translates English to French."),
        HumanMessage("Translate: I love programming."),
        AIMessage("J'adore la programmation."),
        HumanMessage("Translate: I love building applications."),
    ]
    print(model.invoke(conversation).text)


def demo_stream():
    """stream：逐块输出，并把 chunk 累加成完整消息"""
    model = get_model()
    for chunk in model.stream("Why do parrots have colorful feathers?"):
        print(chunk.text, end="|", flush=True)
    print()

    # chunk 可以相加，得到完整的 AIMessageChunk
    full = None
    for chunk in model.stream("What color is the sky?"):
        full = chunk if full is None else full + chunk
    print(full.content_blocks)

    # 按内容块类型分别处理（推理 / 工具调用片段 / 文本）
    for chunk in model.stream("What color is the sky?"):
        for block in chunk.content_blocks:
            if block["type"] == "reasoning" and (reasoning := block.get("reasoning")):
                print(f"Reasoning: {reasoning}")
            elif block["type"] == "tool_call_chunk":
                print(f"Tool call chunk: {block}")
            elif block["type"] == "text":
                print(block["text"], end="")
    print()


def demo_batch():
    """batch / batch_as_completed：并行处理多个输入"""
    model = get_model()
    questions = [
        "Why do parrots have colorful feathers?",
        "How do airplanes fly?",
        "What is quantum computing?",
    ]
    for response in model.batch(questions):
        print("-", response.text[:80])
    # 谁先完成先返回谁，返回 (原始下标, 结果)
    for index, response in model.batch_as_completed(questions):
        print(index, response.text[:60])


# ---------------------------------------------------------------------------
# 工具调用与结构化输出（模型层面；agent 层面见第 4、8 章）
# ---------------------------------------------------------------------------
@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."


def demo_tool_calling():
    """bind_tools：模型只"提出"工具调用，不会自己执行"""
    model_with_tools = get_model().bind_tools([get_weather])
    response = model_with_tools.invoke("What's the weather like in Boston?")
    for tool_call in response.tool_calls:
        print(f"Tool: {tool_call['name']}  Args: {tool_call['args']}")


class Movie(BaseModel):
    """A movie with details."""

    title: str = Field(description="The title of the movie")
    year: int = Field(description="The year the movie was released")
    director: str = Field(description="The director of the movie")
    rating: float = Field(description="The movie's rating out of 10")


class MovieDict(TypedDict):
    """A movie with details."""

    title: Annotated[str, ..., "The title of the movie"]
    year: Annotated[int, ..., "The year the movie was released"]
    director: Annotated[str, ..., "The director of the movie"]
    rating: Annotated[float, ..., "The movie's rating out of 10"]


MOVIE_JSON_SCHEMA = {
    "title": "Movie",
    "description": "A movie with details",
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "The title of the movie"},
        "year": {"type": "integer", "description": "The year the movie was released"},
        "director": {"type": "string", "description": "The director of the movie"},
        "rating": {"type": "number", "description": "The movie's rating out of 10"},
    },
    "required": ["title", "year", "director", "rating"],
}


def demo_structured_output():
    """with_structured_output：Pydantic / TypedDict / JSON Schema 三种 schema"""
    if FAKE:
        print("假模型不支持结构化输出，跳过。")
        return
    model = get_model()
    print(model.with_structured_output(Movie).invoke("Provide details about the movie Inception"))
    print(model.with_structured_output(MovieDict).invoke("Provide details about the movie Inception"))
    print(
        model.with_structured_output(MOVIE_JSON_SCHEMA, method="json_schema").invoke(
            "Provide details about the movie Inception"
        )
    )


# ---------------------------------------------------------------------------
# 进阶：profile、推理、token 用量、运行配置、可配置模型
# ---------------------------------------------------------------------------
def demo_profile_and_reasoning():
    """model.profile 能力描述 + 读取推理（reasoning）内容块"""
    model = get_model()
    print("profile:", getattr(model, "profile", None))
    response = model.invoke("Why do parrots have colorful feathers?")
    reasoning_steps = [b for b in response.content_blocks if b["type"] == "reasoning"]
    print("reasoning:", " ".join(step.get("reasoning", "") for step in reasoning_steps) or "（无）")


def demo_token_usage():
    """usage_metadata 与两种用量回调"""
    model = get_model()
    print(model.invoke("Hello!").usage_metadata)

    callback = UsageMetadataCallbackHandler()
    model.invoke("Hello", config={"callbacks": [callback]})
    get_small_model().invoke("Hello", config={"callbacks": [callback]})
    print(callback.usage_metadata)  # 按模型名汇总

    with get_usage_metadata_callback() as cb:
        model.invoke("Hello")
        print(cb.usage_metadata)


def demo_invocation_config():
    """RunnableConfig：run_name / tags / metadata（会出现在 LangSmith 追踪里）"""
    response = get_model().invoke(
        "Tell me a joke",
        config={
            "run_name": "joke_generation",
            "tags": ["humor", "demo"],
            "metadata": {"user_id": "123"},
        },
    )
    print(response.text)


def demo_configurable_model():
    """不指定模型名，调用时再通过 config 选择模型"""
    if FAKE:
        print("假模型模式下跳过（需要真实模型名）。")
        return
    configurable_model = init_chat_model(temperature=0)
    for name in (MODEL, SMALL_MODEL):
        reply = configurable_model.invoke(
            "what's your name", config={"configurable": {"model": name}}
        )
        print(name, "->", reply.text[:80])


def demo_dynamic_model_selection():
    """中间件里按对话长度切换模型（第 9、11 章还会展开）"""
    basic_model = get_small_model()
    advanced_model = get_model()

    @wrap_model_call
    def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
        """Choose model based on conversation complexity."""
        model = advanced_model if len(request.state["messages"]) > 10 else basic_model
        return handler(request.override(model=model))

    agent = create_agent(model=basic_model, tools=[], middleware=[dynamic_model_selection])
    result = agent.invoke({"messages": [{"role": "user", "content": "Hi!"}]})
    print(result["messages"][-1].text)


if __name__ == "__main__":
    run_demos(globals())
