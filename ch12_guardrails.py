"""第 12 章 · Guardrails：护栏。

文档：https://docs.langchain.com/oss/python/langchain/guardrails

两类护栏：
- 确定性（关键词、正则）：快、便宜、可预测
- 基于模型（让另一个 LLM 审查）：能理解语义，但更慢更贵
放的位置：before_agent（拦截输入）/ after_agent（审查最终输出）/ 工具调用前（人工审批）。

代码来源标注（每个类、函数和主要常量前面都有一行）：
  【文档原代码】          和文档的代码完全一致（只比较语法，不比较注释和排版）
  【文档原代码·有改动】   来自文档，做了改动（标注里写了改了什么）
  【补写】                文档没有给出这段代码，是我写的
来源写成"页面 · 小节"，页面相对于 https://docs.langchain.com/oss/python/langchain/
注意：overview 页只有 get_weather 这一个例子，其余代码来自它链接的其他页面。
"""

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    HumanInTheLoopMiddleware,
    PIIMiddleware,
    after_agent,
    before_agent,
    hook_config,
)
from langchain.messages import AIMessage
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.types import Command

from common import FAKE, get_model, get_small_model, last_text, run_demos


# 【补写】文档只引用了这个工具，没给定义
@tool
def search_tool(query: str) -> str:
    """Search the web."""
    return f"Results for {query}"


# 【补写】文档只引用了这个工具，没给定义
@tool
def calculator_tool(expression: str) -> str:
    """Evaluate a simple arithmetic expression."""
    return str(eval(expression, {"__builtins__": {}}))


# 【补写】文档只引用了这个工具，没给定义
@tool("send_email")
def send_email_tool(to: str, body: str) -> str:
    """Send an email."""
    return f"sent to {to}"


# 【补写】文档只引用了这个工具，没给定义
@tool("delete_database")
def delete_database_tool() -> str:
    """Delete the whole database."""
    return "database deleted"


# 【补写】调用 agent 的小工具
def _ask(agent, text: str, **kwargs):
    return agent.invoke({"messages": [{"role": "user", "content": text}]}, **kwargs)


# 【文档原代码·有改动】guardrails · PII detection：模型改成 get_model()；工具改了；加了 print
def demo_pii():
    """内置 PII 护栏：脱敏邮箱、遮盖信用卡、拦截 API key"""
    agent = create_agent(
        model=get_model(),
        tools=[search_tool],
        middleware=[
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
            PIIMiddleware("api_key", detector=r"sk-[a-zA-Z0-9]{32}", strategy="block", apply_to_input=True),
        ],
    )
    result = _ask(agent, "My email is john.doe@example.com and card is 5105-1051-0510-5100")
    print("模型实际看到的输入：", result["messages"][0].text)


# 【文档原代码·有改动】guardrails · Human-in-the-loop：模型改成 get_model()；改用 version="v2" 并判断是否中断
def demo_human_approval():
    """人工审批护栏：敏感工具执行前暂停，批准后继续"""
    agent = create_agent(
        model=get_model(),
        tools=[search_tool, send_email_tool, delete_database_tool],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={"send_email": True, "delete_database": True, "search": False}
            )
        ],
        checkpointer=InMemorySaver(),  # 暂停/恢复需要 checkpointer
    )
    config = {"configurable": {"thread_id": "some_id"}}
    result = _ask(agent, "Send an email to the team at team@example.com saying hi", config=config, version="v2")
    if result.interrupts:
        print("等待审批：", result.interrupts[0].value["action_requests"])
        result = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=config, version="v2")
    print(result.value["messages"][-1].text)


# ---------------------------------------------------------------------------
# 自定义护栏：before_agent 关键词过滤
# ---------------------------------------------------------------------------
# 【文档原代码·有改动】guardrails · Before agent guardrails：两个判断合并；content 改成 .text；循环方式不变
class ContentFilterMiddleware(AgentMiddleware):
    """Deterministic guardrail: Block requests containing banned keywords."""

    # 【文档原代码】guardrails · Before agent guardrails
    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    # 【文档原代码·有改动】guardrails · Before agent guardrails：两个判断合并；content 改成 .text
    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"] or state["messages"][0].type != "human":
            return None
        content = state["messages"][0].text.lower()
        for keyword in self.banned_keywords:
            if keyword in content:
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": "I cannot process requests containing inappropriate content. Please rephrase your request.",
                    }],
                    "jump_to": "end",  # 直接结束，模型根本不会被调用
                }
        return None


# 【文档原代码】guardrails · Before agent guardrails
banned_keywords = ["hack", "exploit", "malware"]


# 【文档原代码·有改动】guardrails · Before agent guardrails：判断合并；循环改成 any；回复文字缩短；docstring 改了
@before_agent(can_jump_to=["end"])
def content_filter(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """同一个护栏的装饰器写法"""
    if not state["messages"] or state["messages"][0].type != "human":
        return None
    content = state["messages"][0].text.lower()
    if any(keyword in content for keyword in banned_keywords):
        return {
            "messages": [{"role": "assistant", "content": "I cannot process requests containing inappropriate content."}],
            "jump_to": "end",
        }
    return None


# 【文档原代码·有改动】guardrails · Before agent guardrails：模型改成 get_model()；两种写法合并成循环
def demo_before_agent_filter():
    """类写法与装饰器写法的关键词过滤"""
    for mw in (ContentFilterMiddleware(banned_keywords=banned_keywords), content_filter):
        agent = create_agent(model=get_model(), tools=[search_tool, calculator_tool], middleware=[mw])
        print(last_text(_ask(agent, "How do I hack into a database?")))


# ---------------------------------------------------------------------------
# 自定义护栏：after_agent 用模型审查输出
# ---------------------------------------------------------------------------
# 【文档原代码·有改动】guardrails · After agent guardrails：模型改成 get_small_model()；判断合并；content 改成 .text；加了 print
class SafetyGuardrailMiddleware(AgentMiddleware):
    """Model-based guardrail: Use an LLM to evaluate response safety."""

    # 【文档原代码·有改动】guardrails · After agent guardrails：模型改成 get_small_model()
    def __init__(self):
        super().__init__()
        self.safety_model = get_small_model()

    # 【文档原代码·有改动】guardrails · After agent guardrails：判断合并；content 改成 .text；加了 print
    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"] or not isinstance(state["messages"][-1], AIMessage):
            return None
        last_message = state["messages"][-1]
        safety_prompt = f"""Evaluate if this response is safe and appropriate.
Respond with only 'SAFE' or 'UNSAFE'.

Response: {last_message.text}"""
        result = self.safety_model.invoke([{"role": "user", "content": safety_prompt}])
        print(f"[safety] {result.text}")
        if "UNSAFE" in result.text:
            last_message.content = "I cannot provide that response. Please rephrase your request."
        return None


# 【文档原代码·有改动】guardrails · After agent guardrails：包成工厂函数，让模型延迟初始化
def make_safety_guardrail():
    """同一个护栏的装饰器写法"""
    safety_model = get_small_model()

    # 【文档原代码·有改动】guardrails · After agent guardrails：判断合并；提示词写成一行；content 改成 .text
    @after_agent(can_jump_to=["end"])
    def safety_guardrail(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"] or not isinstance(state["messages"][-1], AIMessage):
            return None
        last_message = state["messages"][-1]
        result = safety_model.invoke([{
            "role": "user",
            "content": f"Evaluate if this response is safe and appropriate.\nRespond with only 'SAFE' or 'UNSAFE'.\n\nResponse: {last_message.text}",
        }])
        if "UNSAFE" in result.text:
            last_message.content = "I cannot provide that response. Please rephrase your request."
        return None

    return safety_guardrail


# 【文档原代码·有改动】guardrails · After agent guardrails：模型改成 get_model()；两种写法合并成循环；问题改了
def demo_after_agent_safety():
    """after_agent：最终回答交给小模型审查（类写法与装饰器写法）"""
    for mw in (SafetyGuardrailMiddleware(), make_safety_guardrail()):
        agent = create_agent(model=get_model(), tools=[search_tool, calculator_tool], middleware=[mw])
        print(last_text(_ask(agent, "Give me a fun fact about cats.")))


# 【文档原代码·有改动】guardrails · Combine multiple guardrails：模型改成 get_model()；两个 PII 合并成一个（同名中间件不能重复）；加了调用
def demo_layered_guardrails():
    """多层护栏组合：关键词 → PII → 人工审批 → 模型审查"""
    agent = create_agent(
        model=get_model(),
        tools=[search_tool, send_email_tool],
        middleware=[
            ContentFilterMiddleware(banned_keywords=["hack", "exploit"]),  # 第 1 层：输入过滤
            # 第 2 层：输入输出都脱敏。文档写成两个 PIIMiddleware("email")，
            # 但同名中间件不能重复添加，所以合并成一个实例
            PIIMiddleware("email", strategy="redact", apply_to_input=True, apply_to_output=True),
            HumanInTheLoopMiddleware(interrupt_on={"send_email": True}),  # 第 3 层：人工审批
            SafetyGuardrailMiddleware(),  # 第 4 层：模型审查
        ],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "layers"}}
    result = _ask(agent, "Summarize what a guardrail is. My email is me@example.com.", config=config)
    for m in result["messages"]:
        print(f"[{m.type}] {m.text[:150]}")
    if FAKE:
        print("（假模型模式：安全审查结果无意义）")


# 【补写】命令行入口
if __name__ == "__main__":
    run_demos(globals())
