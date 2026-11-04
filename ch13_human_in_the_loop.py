"""第 13 章 · Human-in-the-loop：执行敏感操作前让人审批。

文档：https://docs.langchain.com/oss/python/langchain/human-in-the-loop

流程：agent 想调用被拦截的工具 → 产生 interrupt 并暂停（state 存进 checkpointer）
      → 人做决策 → agent.invoke(Command(resume=...)) 从断点继续。
四种决策：approve（批准）/ edit（改参数后执行）/ reject（拒绝并说明原因）/ respond（人直接给出工具结果）

代码来源标注（每个类、函数和主要常量前面都有一行）：
  【文档原代码】          和文档的代码完全一致（只比较语法，不比较注释和排版）
  【文档原代码·有改动】   来自文档，做了改动（标注里写了改了什么）
  【补写】                文档没有给出这段代码，是我写的
来源写成"页面 · 小节"，页面相对于 https://docs.langchain.com/oss/python/langchain/
注意：overview 页只有 get_weather 这一个例子，其余代码来自它链接的其他页面。
"""

import json

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, ToolCallRequest
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import ValidationError

from common import get_model, run_demos


# 【文档原代码】human-in-the-loop · Pause once per tool call
@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    return f"Wrote {len(content)} characters to {path}"


# 【文档原代码·有改动】human-in-the-loop · Pause once per tool call：按是否 DELETE 返回不同结果
@tool
def execute_sql(query: str) -> str:
    """Run a SQL query."""
    return "Deleted 42 rows" if "DELETE" in query.upper() else "3 rows"


# 【文档原代码】human-in-the-loop · Pause once per tool call
@tool
def read_data(table: str) -> str:
    """Read rows from a table."""
    return f"3 rows from {table}"


# 【补写】本章工具列表
TOOLS = [write_file, execute_sql, read_data]


# 【补写】在终端里询问审批决策
def _ask_decision(action: dict, allowed: list[str]) -> dict:
    """在终端里让你亲自做决策。"""
    print(f"\n⚠️  {action['description']}")
    choice = ""
    while choice not in allowed:
        choice = input(f"选择 {allowed}: ").strip()
    if choice == "approve":
        return {"type": "approve"}
    if choice == "edit":
        args = json.loads(input(f"新的参数 JSON（原参数 {json.dumps(action['args'])}）: "))
        return {"type": "edit", "edited_action": {"name": action["name"], "args": args}}
    if choice == "reject":
        return {"type": "reject", "message": input("拒绝原因: ")}
    return {"type": "respond", "message": input("你给出的工具结果: ")}


# 【文档原代码·有改动】human-in-the-loop · Responding to interrupts：模型改成 get_model()；固定决策改成终端交互，并循环处理中断
def demo_batch_mode():
    """默认模式：一次中断里包含本轮所有待审批的调用，按顺序给出决策"""
    agent = create_agent(
        model=get_model(),
        tools=TOOLS,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "write_file": True,  # 允许全部四种决策
                    "execute_sql": {"allowed_decisions": ["approve", "reject"]},  # 不允许改参数
                    "read_data": False,  # 安全操作，不拦截
                },
                description_prefix="Tool execution pending approval",
            ),
        ],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "batch"}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Delete records older than 30 days from the records table."}]},
        config=config,
        version="v2",
    )
    while result.interrupts:
        request = result.interrupts[0].value
        allowed = {c["action_name"]: c["allowed_decisions"] for c in request["review_configs"]}
        decisions = [_ask_decision(a, allowed[a["name"]]) for a in request["action_requests"]]
        result = agent.invoke(Command(resume={"decisions": decisions}), config=config, version="v2")
    print("\n最终回答：", result.value["messages"][-1].text)


# 【文档原代码·有改动】human-in-the-loop · Conditional interrupts：去掉中间变量 path
def writes_outside_workspace(request: ToolCallRequest) -> bool:
    """Pause writes to paths outside the workspace directory."""
    return not request.tool_call["args"].get("path", "").startswith("/workspace/")


# 【文档原代码·有改动】human-in-the-loop · Conditional interrupts：去掉中间变量 query
def is_write_query(request: ToolCallRequest) -> bool:
    """Pause SQL that isn't a read-only SELECT."""
    return not request.tool_call["args"].get("query", "").lstrip().upper().startswith("SELECT")


# 【文档原代码·有改动】human-in-the-loop · Conditional interrupts：模型改成 get_model()；调用部分是补写
def demo_conditional_interrupts():
    """when=...：只有满足条件的调用才需要审批（只读 SELECT 自动放行）"""
    agent = create_agent(
        model=get_model(),
        tools=TOOLS,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "write_file": {"allowed_decisions": ["approve", "edit", "reject"], "when": writes_outside_workspace},
                    "execute_sql": {"allowed_decisions": ["approve", "reject"], "when": is_write_query},
                },
            ),
        ],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "conditional"}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Run SELECT count(*) FROM orders, then write the count to /tmp/count.txt"}]},
        config=config,
        version="v2",
    )
    for interrupt in result.interrupts:
        print("需要审批：", [a["description"] for a in interrupt.value["action_requests"]])
    if not result.interrupts:
        print(result.value["messages"][-1].text)


# 【文档原代码·有改动】human-in-the-loop · Pause once per tool call：合并本节和 Resume / Correct 几段；模型改成 get_model()
def demo_per_call_mode():
    """interrupt_mode="per_call"：每个调用一个 interrupt，按 interrupt.id 分别答复"""
    agent = create_agent(
        model=get_model(),
        tools=TOOLS,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "write_file": True,
                    "execute_sql": {"allowed_decisions": ["approve", "reject"]},
                    "read_data": False,
                },
                interrupt_mode="per_call",
            ),
        ],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "per-call"}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Archive old orders to orders_archive.csv, then delete them with SQL."}]},
        config=config,
        version="v2",
    )
    for interrupt in result.interrupts:
        print(interrupt.value)

    # 拒绝 SQL，批准写文件
    decisions = {}
    for interrupt in result.interrupts:
        if interrupt.value["name"] == "execute_sql":
            decisions[interrupt.id] = {"type": "reject", "message": "Archive the rows before deleting them."}
        else:
            decisions[interrupt.id] = {"type": "approve"}

    # 演示：edit 时少给了必填参数 → ValidationError，什么都没保存，可以重新答复
    write_interrupt = next((i for i in result.interrupts if i.value["name"] == "write_file"), None)
    if write_interrupt:
        bad_edit = {"type": "edit", "edited_action": {"name": "write_file", "args": {"path": "archive/orders_2024.csv"}}}
        try:
            agent.invoke(Command(resume={write_interrupt.id: bad_edit}), config=config, version="v2")
        except ValidationError as error:
            for problem in error.errors():
                print("校验失败：", ".".join(str(p) for p in problem["loc"]), problem["msg"])

    if decisions:
        result = agent.invoke(Command(resume=decisions), config=config, version="v2")
        print(result.value["messages"][-1].text)


# 【文档原代码·有改动】human-in-the-loop · Streaming with human-in-the-loop：模型改成 get_model()；只有中断时才恢复；agent 配置是补写
def demo_streaming_with_hitl():
    """stream_events v3 + stream.interrupted / stream.interrupts"""
    agent = create_agent(
        model=get_model(),
        tools=TOOLS,
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"execute_sql": True})],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "stream"}}
    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": "Delete old records from the database"}]},
        config=config,
        version="v3",
    )
    for message in stream.messages:
        for token in message.text:
            print(token, end="", flush=True)
    if stream.interrupted:
        print(f"\n\nInterrupt: {stream.interrupts}")
        stream = agent.stream_events(Command(resume={"decisions": [{"type": "approve"}]}), config=config, version="v3")
        for message in stream.messages:
            for token in message.text:
                print(token, end="", flush=True)
    print()


# 【补写】命令行入口
if __name__ == "__main__":
    run_demos(globals())
