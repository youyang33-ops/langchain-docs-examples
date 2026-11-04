"""第 8 章 · Structured output：让 agent 返回固定结构的数据。

文档：https://docs.langchain.com/oss/python/langchain/structured-output

两种策略：
- ProviderStrategy：用供应商原生的结构化输出（OpenAI/Anthropic/Gemini 等支持时自动选择）
- ToolStrategy：把 schema 伪装成一个工具让模型"调用"，任何支持工具调用的模型都能用
schema 可以是 Pydantic / dataclass / TypedDict / JSON Schema。
结果在 result["structured_response"]。

代码来源标注（每个类、函数和主要常量前面都有一行）：
  【文档原代码】          和文档的代码完全一致（只比较语法，不比较注释和排版）
  【文档原代码·有改动】   来自文档，做了改动（标注里写了改了什么）
  【补写】                文档没有给出这段代码，是我写的
来源写成"页面 · 小节"，页面相对于 https://docs.langchain.com/oss/python/langchain/
注意：overview 页只有 get_weather 这一个例子，其余代码来自它链接的其他页面。
"""

from dataclasses import dataclass
from typing import Literal, Union

from langchain.agents import create_agent
from langchain.agents.structured_output import (
    MultipleStructuredOutputsError,
    ProviderStrategy,
    StructuredOutputValidationError,
    ToolStrategy,
)
from langchain.messages import ToolMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from common import FAKE, get_model, run_demos

# 【文档原代码·有改动】structured-output · Provider strategy：把文档的问题抽成常量
CONTACT_PROMPT = "Extract contact info from: John Doe, john@example.com, (555) 123-4567"
# 【文档原代码·有改动】structured-output · Tool calling strategy：把文档的问题抽成常量
REVIEW_PROMPT = "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"


# 【补写】创建 agent 并调用，加了假模型跳过
def _ask(response_format, prompt: str, **kwargs):
    if FAKE:  # 假模型永远不会产出结构化结果，agent 会一直重试
        return {"structured_response": "（假模型模式跳过）", "messages": []}
    agent = create_agent(model=get_model(), tools=[], response_format=response_format, **kwargs)
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    return result


# ---------------------------------------------------------------------------
# ProviderStrategy
# ---------------------------------------------------------------------------
# 【文档原代码】structured-output · Provider strategy
class ContactInfo(BaseModel):
    """Contact information for a person."""

    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")


# 【文档原代码·有改动】structured-output · Provider strategy：文档叫 ContactInfo，这里改名；去掉字段注释
@dataclass
class ContactInfoDC:
    """Contact information for a person."""

    name: str
    email: str
    phone: str


# 【文档原代码·有改动】structured-output · Provider strategy：文档叫 ContactInfo，这里改名；去掉字段注释
class ContactInfoTD(TypedDict):
    """Contact information for a person."""

    name: str
    email: str
    phone: str


# 【文档原代码·有改动】structured-output · Provider strategy：变量改名
CONTACT_JSON_SCHEMA = {
    "title": "ContactInfo",
    "type": "object",
    "description": "Contact information for a person.",
    "properties": {
        "name": {"type": "string", "description": "The name of the person"},
        "email": {"type": "string", "description": "The email address of the person"},
        "phone": {"type": "string", "description": "The phone number of the person"},
    },
    "required": ["name", "email", "phone"],
}


# 【文档原代码·有改动】structured-output · Provider strategy：四段示例合并成循环；模型通过 _ask 取得
def demo_provider_strategy():
    """直接传 schema 类型会自动选择 ProviderStrategy"""
    for fmt in (ContactInfo, ContactInfoDC, ContactInfoTD, ProviderStrategy(CONTACT_JSON_SCHEMA)):
        print(_ask(fmt, CONTACT_PROMPT)["structured_response"])


# ---------------------------------------------------------------------------
# ToolStrategy
# ---------------------------------------------------------------------------
# 【文档原代码】structured-output · Tool calling strategy
class ProductReview(BaseModel):
    """Analysis of a product review."""

    rating: int | None = Field(description="The rating of the product", ge=1, le=5)
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the review")
    key_points: list[str] = Field(description="The key points of the review. Lowercase, 1-3 words each.")


# 【文档原代码】structured-output · Tool calling strategy
class CustomerComplaint(BaseModel):
    """A customer complaint about a product or service."""

    issue_type: Literal["product", "service", "shipping", "billing"] = Field(description="The type of issue")
    severity: Literal["low", "medium", "high"] = Field(description="The severity of the complaint")
    description: str = Field(description="Brief description of the complaint")


# 【文档原代码·有改动】structured-output · Tool calling strategy：变量改名；去掉字段 description
PRODUCT_REVIEW_SCHEMA = {
    "title": "ProductReview",
    "type": "object",
    "description": "Analysis of a product review.",
    "properties": {
        "rating": {"type": ["integer", "null"], "minimum": 1, "maximum": 5},
        "sentiment": {"type": "string", "enum": ["positive", "negative"]},
        "key_points": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["sentiment", "key_points"],
}


# 【文档原代码·有改动】structured-output · Tool calling strategy：几段示例合并；最后那句投诉是补写
def demo_tool_strategy():
    """ToolStrategy：Pydantic、JSON Schema，以及 Union 多选一"""
    print(_ask(ToolStrategy(ProductReview), REVIEW_PROMPT)["structured_response"])
    print(_ask(ToolStrategy(PRODUCT_REVIEW_SCHEMA), REVIEW_PROMPT)["structured_response"])
    # Union：模型自己判断这是评价还是投诉
    print(_ask(ToolStrategy(Union[ProductReview, CustomerComplaint]), REVIEW_PROMPT)["structured_response"])
    print(_ask(
        ToolStrategy(Union[ProductReview, CustomerComplaint]),
        "My package arrived 2 weeks late and the box was crushed!",
    )["structured_response"])


# 【文档原代码】structured-output · Custom tool message content
class MeetingAction(BaseModel):
    """Action items extracted from a meeting transcript."""

    task: str = Field(description="The specific task to be completed")
    assignee: str = Field(description="Person responsible for the task")
    priority: Literal["low", "medium", "high"] = Field(description="Priority level")


# 【文档原代码·有改动】structured-output · Custom tool message content：模型通过 _ask 取得；加了 print
def demo_tool_message_content():
    """tool_message_content：自定义结构化输出后写入历史的那条 ToolMessage"""
    result = _ask(
        ToolStrategy(schema=MeetingAction, tool_message_content="Action item captured and added to meeting notes!"),
        "From our meeting: Sarah needs to update the project timeline as soon as possible",
    )
    print(result["structured_response"])
    print([m.content for m in result["messages"] if isinstance(m, ToolMessage)])


# ---------------------------------------------------------------------------
# 错误处理：schema 校验失败 / 返回了多个结构化输出
# ---------------------------------------------------------------------------
# 【文档原代码·有改动】structured-output · Multiple structured outputs error：文档叫 ContactInfo，这里改名
class ContactInfoShort(BaseModel):
    name: str = Field(description="Person's name")
    email: str = Field(description="Email address")


# 【文档原代码】structured-output · Multiple structured outputs error
class EventDetails(BaseModel):
    event_name: str = Field(description="Name of the event")
    date: str = Field(description="Event date")


# 【文档原代码】structured-output · Schema validation error
class ProductRating(BaseModel):
    rating: int | None = Field(description="Rating from 1-5", ge=1, le=5)
    comment: str = Field(description="Review comment")


# 【文档原代码·有改动】structured-output · Error handling strategies：elif / else 改成提前 return；去掉 str()
def custom_error_handler(error: Exception) -> str:
    if isinstance(error, StructuredOutputValidationError):
        return "There was an issue with the format. Try again."
    if isinstance(error, MultipleStructuredOutputsError):
        return "Multiple structured outputs were returned. Pick the most relevant one."
    return f"Error: {error}"


# 【文档原代码·有改动】structured-output · Error handling strategies：几段示例合并；模型通过 _ask 取得
def demo_error_handling():
    """handle_errors：默认 True 自动重试；也可以给字符串、异常类型或函数"""
    # 10/10 超出 1-5 范围 → 校验失败 → 把错误反馈给模型重试
    result = _ask(
        ToolStrategy(ProductRating),
        "Parse this: Amazing product, 10/10!",
        system_prompt="You are a helpful assistant that parses product reviews. Do not make any field or value up.",
    )
    print(result["structured_response"])

    # 其他写法：
    ToolStrategy(schema=ProductRating, handle_errors="Please provide a valid rating between 1-5 and include a comment.")
    ToolStrategy(schema=ProductRating, handle_errors=ValueError)
    ToolStrategy(schema=ProductRating, handle_errors=(ValueError, TypeError))
    ToolStrategy(schema=ProductRating, handle_errors=False)  # 出错直接抛异常

    result = _ask(
        ToolStrategy(schema=Union[ContactInfoShort, EventDetails], handle_errors=custom_error_handler),
        "Extract info: John Doe (john@email.com) is organizing Tech Conference on March 15th",
    )
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            print("ToolMessage:", msg.content)
    print(result["structured_response"])


# 【补写】命令行入口
if __name__ == "__main__":
    if FAKE:
        print("假模型无法产生结构化输出，本章请用真实模型运行。")
    run_demos(globals())
