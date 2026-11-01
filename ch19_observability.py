"""第 19 章 · Observability：用 LangSmith 追踪 agent 的每一步。

文档：https://docs.langchain.com/oss/python/langchain/observability

在 .env 里设置：
    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=lsv2_...
    LANGSMITH_PROJECT=langchain-learning   # 可选
之后所有 agent / 模型调用都会自动上传追踪，在 https://smith.langchain.com 查看。
测试相关的内容（第 20 章）在 tests/ 目录，用 pytest 运行。
"""

import os

import langsmith as ls
from langchain.agents import create_agent

from common import get_model, last_text, run_demos


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {to}"


def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"


def build_agent():
    return create_agent(
        model=get_model(),
        tools=[send_email, search_web],
        system_prompt="You are a helpful assistant that can send emails and search the web.",
    )


def demo_status():
    """检查追踪是否开启"""
    print("LANGSMITH_TRACING =", os.getenv("LANGSMITH_TRACING"))
    print("LANGSMITH_API_KEY 已设置 =", bool(os.getenv("LANGSMITH_API_KEY")))
    print("LANGSMITH_PROJECT =", os.getenv("LANGSMITH_PROJECT", "default"))


def demo_auto_tracing():
    """开启环境变量后，普通的 invoke 就会被追踪"""
    agent = build_agent()
    response = agent.invoke({
        "messages": [{"role": "user", "content": "Search for the latest AI news and email a summary to john@example.com"}]
    })
    print(last_text(response))


def demo_selective_tracing():
    """tracing_context：只追踪代码里的某一段"""
    agent = build_agent()
    with ls.tracing_context(enabled=True):
        agent.invoke({"messages": [{"role": "user", "content": "Send a test email to alice@example.com"}]})
    # 这一次不会被追踪（前提是没有设置 LANGSMITH_TRACING）
    agent.invoke({"messages": [{"role": "user", "content": "Send another email"}]})
    print("完成。去 LangSmith 看看是否只有第一次调用被记录。")


def demo_tags_and_metadata():
    """给追踪加 tags / metadata，方便筛选"""
    agent = build_agent()
    agent.invoke(
        {"messages": [{"role": "user", "content": "Send a welcome email"}]},
        config={
            "tags": ["production", "email-assistant", "v1.0"],
            "metadata": {"user_id": "user_123", "session_id": "session_456", "environment": "production"},
        },
    )
    with ls.tracing_context(
        enabled=True,
        tags=["production", "email-assistant", "v1.0"],
        metadata={"user_id": "user_123", "session_id": "session_456", "environment": "production"},
    ):
        agent.invoke({"messages": [{"role": "user", "content": "Send a welcome email"}]})
    print("完成。")


if __name__ == "__main__":
    run_demos(globals())
