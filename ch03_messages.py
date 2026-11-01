"""第 3 章 · Messages：模型的输入输出单位。

文档：https://docs.langchain.com/oss/python/langchain/messages

四种消息：SystemMessage / HumanMessage / AIMessage / ToolMessage。
每条消息都有 content（供应商原生格式）和 content_blocks（统一的标准格式）。
"""

from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.load import dumpd, load

from common import get_model, run_demos


def demo_basic_usage():
    """三种输入写法：字符串 / 消息对象 / 字典"""
    model = get_model()
    print(model.invoke("Write a haiku about spring").text)

    messages = [
        SystemMessage("You are a poetry expert"),
        HumanMessage("Write a haiku about spring"),
        AIMessage("Cherry blossoms bloom..."),
    ]
    print(model.invoke(messages).text)

    messages = [
        {"role": "system", "content": "You are a poetry expert"},
        {"role": "user", "content": "Write a haiku about spring"},
        {"role": "assistant", "content": "Cherry blossoms bloom..."},
    ]
    print(model.invoke(messages).text)


def demo_message_types():
    """系统消息、带元数据的人类消息、手动构造的 AI 消息"""
    model = get_model()
    system_msg = SystemMessage("""
You are a senior Python developer with expertise in web frameworks.
Always provide code examples and explain your reasoning.
Be concise but thorough in your explanations.
""")
    human_msg = HumanMessage(
        content="How do I create a REST API?",
        name="alice",  # 可选：区分不同用户
        id="msg_123",  # 可选：用于追踪
    )
    response = model.invoke([system_msg, human_msg])
    print(type(response).__name__, "->", response.text[:200])

    # 手动插入一条 AI 消息，当作历史记录
    history = [
        SystemMessage("You are a helpful assistant"),
        HumanMessage("Can you help me?"),
        AIMessage("I'd be happy to help you with that question!"),
        HumanMessage("Great! What's 2+2?"),
    ]
    print(model.invoke(history).text)


def demo_tool_calls_and_tool_message():
    """AIMessage.tool_calls 与配对的 ToolMessage（tool_call_id 必须一致）"""
    model = get_model()

    def get_weather(location: str) -> str:
        """Get the weather at a location."""
        ...

    response = model.bind_tools([get_weather]).invoke("What's the weather in Paris?")
    for tool_call in response.tool_calls:
        print(f"Tool: {tool_call['name']}  Args: {tool_call['args']}  ID: {tool_call['id']}")

    # 手动构造一次完整的"调用工具 → 返回结果"
    ai_message = AIMessage(
        content=[],
        tool_calls=[{"name": "get_weather", "args": {"location": "San Francisco"}, "id": "call_123"}],
    )
    tool_message = ToolMessage(content="Sunny, 72°F", tool_call_id="call_123")
    messages = [
        HumanMessage("What's the weather in San Francisco?"),
        ai_message,
        tool_message,
    ]
    print(model.invoke(messages).text)


def demo_usage_and_chunks():
    """usage_metadata，以及流式 chunk 的累加"""
    model = get_model()
    print(model.invoke("Hello!").usage_metadata)

    full_message = None
    for chunk in model.stream("Hi"):
        full_message = chunk if full_message is None else full_message + chunk
    print(full_message.text)


def demo_content_blocks():
    """content（原生格式）与 content_blocks（标准格式）的对应关系"""
    # Anthropic 的 thinking 块 → 标准的 reasoning 块
    message = AIMessage(
        content=[
            {"type": "thinking", "thinking": "...", "signature": "WaUjzkyp..."},
            {"type": "text", "text": "..."},
        ],
        response_metadata={"model_provider": "anthropic"},
    )
    print(message.content_blocks)

    # OpenAI 的 reasoning summary → 同样的 reasoning 块
    message = AIMessage(
        content=[
            {
                "type": "reasoning",
                "id": "rs_abc123",
                "summary": [
                    {"type": "summary_text", "text": "summary 1"},
                    {"type": "summary_text", "text": "summary 2"},
                ],
            },
            {"type": "text", "text": "...", "id": "msg_abc123"},
        ],
        response_metadata={"model_provider": "openai"},
    )
    print(message.content_blocks)

    # 构造消息时也可以直接给标准内容块
    human_message = HumanMessage(
        content_blocks=[
            {"type": "text", "text": "Hello, how are you?"},
            {"type": "image", "url": "https://example.com/image.jpg"},
        ]
    )
    print(human_message.content_blocks)


# 多模态输入的几种写法（URL / base64 / 供应商文件 ID），只展示结构，不发送
MULTIMODAL_EXAMPLES = {
    "image_url": {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the content of this image."},
            {"type": "image", "url": "https://example.com/path/to/image.jpg"},
        ],
    },
    "image_base64": {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the content of this image."},
            {"type": "image", "base64": "AAAAIGZ0eXBtcDQy...", "mime_type": "image/jpeg"},
        ],
    },
    "pdf_file": {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the content of this document."},
            {"type": "file", "url": "https://example.com/path/to/document.pdf"},
        ],
    },
    "audio": {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the content of this audio."},
            {"type": "audio", "base64": "AAAAIGZ0eXBtcDQy...", "mime_type": "audio/wav"},
        ],
    },
    "video_file_id": {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the content of this video."},
            {"type": "video", "file_id": "file-abc123"},
        ],
    },
}


def demo_multimodal():
    """多模态消息结构：图片 / 文件 / 音频 / 视频"""
    for name, message in MULTIMODAL_EXAMPLES.items():
        print(name, "->", [block["type"] for block in message["content"]])
    print("要真正发送，请把 URL 换成真实图片，再 get_model().invoke([message])。")


def demo_serialization():
    """dumpd / load：消息与字典互转（存数据库、跨进程传递）"""
    message = HumanMessage("What is the capital of France?")
    serialized = dumpd(message)
    restored = load(serialized, allowed_objects="messages")  # 只允许反序列化消息类型，更安全
    print(serialized)
    print(restored == message)


if __name__ == "__main__":
    run_demos(globals())
