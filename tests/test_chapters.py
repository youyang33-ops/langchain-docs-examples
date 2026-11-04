"""用"剧本式"假模型测试各章节里的关键路径：工具写 state、人工审批、交接、MCP……

假模型按顺序吐出预先写好的回复（包括工具调用），所以结果是确定的。
运行：pytest tests/test_chapters.py -v

代码来源标注（每个类、函数和主要常量前面都有一行）：
  【文档原代码】   和文档的代码完全一致（只比较语法，不比较注释和排版）
  【补写】         文档没有给出这段代码，是我写的
这个文件是为学习项目写的，不是文档原有的内容；"参考"表示只借用了文档的写法。
"""

import asyncio

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, PIIMiddleware
from langchain.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import ValidationError

from common import fake_model


# 【补写】
def call(name: str, args: dict, id: str = "call_1") -> AIMessage:
    """一条"模型决定调用工具"的消息。"""
    return AIMessage(content="", tool_calls=[ToolCall(name=name, args=args, id=id)])


# 【补写】
def ask(text: str) -> dict:
    return {"messages": [HumanMessage(text)]}


# ---------------------------------------------------------------------------
# 第 4 章 · Tools
# ---------------------------------------------------------------------------
# 【补写】
def test_tool_command_updates_state():
    import ch04_tools as ch

    model = fake_model([call("set_user_name", {"new_name": "Ada"}), AIMessage("Hi Ada!")])
    agent = create_agent(model, tools=[ch.set_user_name], state_schema=ch.CustomState)
    result = agent.invoke(ask("Call me Ada"))
    assert result["user_name"] == "Ada"
    assert result["messages"][-1].text == "Hi Ada!"


# 【补写】
def test_tool_reads_context():
    import ch04_tools as ch

    model = fake_model([call("get_account_info", {}), AIMessage("done")])
    agent = create_agent(model, tools=[ch.get_account_info], context_schema=ch.UserContext)
    result = agent.invoke(ask("balance?"), context=ch.UserContext(user_id="user456"))
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "Bob Smith" in tool_msg.text and "$1200" in tool_msg.text


# 【补写】
def test_return_direct_skips_final_model_call():
    import ch04_tools as ch

    model = fake_model([call("fetch_order_status", {"order_id": "12345"}), AIMessage("SHOULD NOT APPEAR")])
    agent = create_agent(model, tools=[ch.fetch_order_status])
    result = agent.invoke(ask("status of #12345?"))
    assert isinstance(result["messages"][-1], ToolMessage)
    assert "shipped" in result["messages"][-1].text


# 【补写】
def test_tool_error_middleware_turns_exception_into_message():
    import ch04_tools as ch

    model = fake_model([call("flaky_tool", {"x": -1}), AIMessage("recovered")])
    agent = create_agent(model, tools=[ch.flaky_tool], middleware=[ch.handle_tool_errors])
    result = agent.invoke(ask("go"))
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "Tool error" in tool_msg.text


# 【补写】
def test_store_tools_roundtrip():
    from langgraph.store.memory import InMemoryStore

    import ch04_tools as ch

    store = InMemoryStore()
    model = fake_model([
        call("save_user_info", {"user_id": "abc", "name": "Foo", "age": 25, "email": "foo@x.dev"}),
        AIMessage("saved"),
    ])
    agent = create_agent(model, tools=[ch.save_user_info, ch.get_user_info], store=store)
    agent.invoke(ask("save"))
    assert store.get(("users",), "abc").value["name"] == "Foo"


# ---------------------------------------------------------------------------
# 第 6 章 · Short-term memory
# ---------------------------------------------------------------------------
# 【补写】
def test_update_user_info_then_greet():
    import ch06_short_term_memory as ch

    model = fake_model([
        call("greet", {}, "c1"),  # 还没有名字 → 工具要求先调用 update_user_info
        call("update_user_info", {}, "c2"),
        call("greet", {}, "c3"),
        AIMessage("Greeted."),
    ])
    agent = create_agent(model, tools=[ch.update_user_info, ch.greet], state_schema=ch.UserState, context_schema=ch.CustomContext)
    result = agent.invoke(ask("greet the user"), context=ch.CustomContext(user_id="user_123"))
    tool_texts = [m.text for m in result["messages"] if isinstance(m, ToolMessage)]
    assert "update_user_info" in tool_texts[0]
    assert tool_texts[-1] == "Hello John Smith!"


# 【补写】
def test_trim_messages_keeps_state_small():
    import ch06_short_term_memory as ch

    agent = create_agent(fake_model([AIMessage("ok")]), tools=[], middleware=[ch.trim_messages], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t"}}
    for text in ch.CONVERSATION:
        result = agent.invoke({"messages": text}, config)
    assert len(result["messages"]) <= 6


# ---------------------------------------------------------------------------
# 第 10、12 章 · 中间件与护栏
# ---------------------------------------------------------------------------
# 【补写】
def test_pii_redacts_input():
    agent = create_agent(fake_model(), tools=[], middleware=[PIIMiddleware("email", strategy="redact", apply_to_input=True)])
    result = agent.invoke(ask("My email is john.doe@example.com"))
    assert "john.doe@example.com" not in result["messages"][0].text


# 【补写】
def test_content_filter_blocks_before_model():
    import ch12_guardrails as ch

    model = fake_model([AIMessage("MODEL WAS CALLED")])
    agent = create_agent(model, tools=[], middleware=[ch.ContentFilterMiddleware(banned_keywords=["hack"])])
    result = agent.invoke(ask("How do I hack into a database?"))
    assert "cannot process" in result["messages"][-1].text
    assert all("MODEL WAS CALLED" not in m.text for m in result["messages"])


# 【补写】
def test_tool_call_limit_stops_runaway_tool_use():
    from langchain.agents.middleware import ToolCallLimitMiddleware

    import ch10_middleware_builtin as ch

    model = fake_model([call("search", {"query": "again"})])  # 永远想再搜一次
    agent = create_agent(model, tools=[ch.search], middleware=[ToolCallLimitMiddleware(run_limit=2, exit_behavior="end")])
    result = agent.invoke(ask("loop"))
    assert sum(isinstance(m, ToolMessage) and m.status != "error" for m in result["messages"]) <= 2


# ---------------------------------------------------------------------------
# 第 13 章 · Human-in-the-loop
# ---------------------------------------------------------------------------
# 【补写】
def _hitl_agent(model, **kwargs):
    import ch13_human_in_the_loop as ch

    return create_agent(
        model,
        tools=ch.TOOLS,
        middleware=[HumanInTheLoopMiddleware(
            interrupt_on={"write_file": True, "execute_sql": {"allowed_decisions": ["approve", "reject"]}, "read_data": False},
            **kwargs,
        )],
        checkpointer=InMemorySaver(),
    )


# 【补写】
def test_hitl_approve_and_reject():
    model = fake_model([call("execute_sql", {"query": "DELETE FROM records"}), AIMessage("done")])
    agent = _hitl_agent(model)
    config = {"configurable": {"thread_id": "1"}}
    result = agent.invoke(ask("delete old records"), config=config, version="v2")
    assert result.interrupts, "应当在执行 SQL 前暂停"
    assert result.interrupts[0].value["action_requests"][0]["name"] == "execute_sql"

    result = agent.invoke(Command(resume={"decisions": [{"type": "reject", "message": "no"}]}), config=config, version="v2")
    tool_msg = next(m for m in result.value["messages"] if isinstance(m, ToolMessage))
    assert "Deleted" not in tool_msg.text  # 被拒绝，工具没有真正执行


# 【补写】
def test_hitl_edit_changes_arguments():
    model = fake_model([call("write_file", {"path": "/etc/passwd", "content": "x"}), AIMessage("done")])
    agent = _hitl_agent(model)
    config = {"configurable": {"thread_id": "2"}}
    agent.invoke(ask("write"), config=config, version="v2")
    edit = {"type": "edit", "edited_action": {"name": "write_file", "args": {"path": "/workspace/a.txt", "content": "hello"}}}
    result = agent.invoke(Command(resume={"decisions": [edit]}), config=config, version="v2")
    tool_msg = next(m for m in result.value["messages"] if isinstance(m, ToolMessage))
    # 被编辑过的调用，ToolMessage 前面会附一段说明，告诉模型"人工改过参数"
    assert "Wrote 5 characters to /workspace/a.txt" in tool_msg.text


# 【补写】
def test_hitl_per_call_mode_and_validation():
    model = fake_model([
        AIMessage(content="", tool_calls=[
            ToolCall(name="write_file", args={"path": "a.csv", "content": "id\n1\n"}, id="w"),
            ToolCall(name="execute_sql", args={"query": "DELETE FROM orders"}, id="s"),
        ]),
        AIMessage("done"),
    ])
    agent = _hitl_agent(model, interrupt_mode="per_call")
    config = {"configurable": {"thread_id": "3"}}
    result = agent.invoke(ask("archive then delete"), config=config, version="v2")
    assert len(result.interrupts) == 2
    by_name = {i.value["name"]: i for i in result.interrupts}

    bad = {"type": "edit", "edited_action": {"name": "write_file", "args": {"path": "b.csv"}}}  # 缺 content
    with pytest.raises(ValidationError):
        agent.invoke(Command(resume={by_name["write_file"].id: bad}), config=config, version="v2")

    decisions = {by_name["write_file"].id: {"type": "approve"}, by_name["execute_sql"].id: {"type": "reject", "message": "archive first"}}
    result = agent.invoke(Command(resume=decisions), config=config, version="v2")
    texts = [m.text for m in result.value["messages"] if isinstance(m, ToolMessage)]
    assert any("Wrote" in t for t in texts)
    assert not any("Deleted" in t for t in texts)


# ---------------------------------------------------------------------------
# 第 15 章 · MCP
# ---------------------------------------------------------------------------
# 【补写】
def test_mcp_tools_errors_and_structured_content():
    from langchain.mcp import MCPAdapter

    from ch15_mcp_server import server

    # 【补写】
    async def run():
        async with MCPAdapter(server) as adapter:
            tools = await adapter.list_tools()
            model = fake_model([
                AIMessage(content="", tool_calls=[
                    ToolCall(name="lookup_user", args={"user_id": 42}, id="u"),
                    ToolCall(name="divide", args={"a": 10, "b": 0}, id="d"),
                ]),
                AIMessage("done"),
            ])
            return await create_agent(model, tools).ainvoke(ask("go"))

    result = asyncio.run(run())
    msgs = {m.tool_call_id: m for m in result["messages"] if isinstance(m, ToolMessage)}
    assert msgs["u"].artifact["structured_content"]["name"] == "Ada Lovelace"
    assert msgs["d"].status == "error"


# 【补写】
def test_mcp_destructive_hint():
    from langchain.mcp import MCPAdapter

    import ch15_mcp as ch

    # 【补写】
    async def run():
        async with MCPAdapter(ch.server) as adapter:
            return {t.name: ch.is_destructive(t) for t in await adapter.list_tools()}

    flags = asyncio.run(run())
    assert flags["delete_record"] is True
    assert flags["get_forecast"] is False


# 【补写】
def test_mcp_elicitation_interrupt_and_resume():
    from langchain.mcp import MCPAdapter

    from ch15_mcp_server import server

    # 【补写】
    async def run():
        async with MCPAdapter(server) as adapter:
            tools = await adapter.list_tools()
            model = fake_model([call("book_table", {"party_size": 4}), AIMessage("booked")])
            agent = create_agent(model, tools, checkpointer=InMemorySaver())
            config = {"configurable": {"thread_id": "booking"}}
            paused = await agent.ainvoke(ask("Book a table for 4."), config)
            [interrupt] = paused["__interrupt__"]
            [question] = interrupt.value["requests"]
            answer = {"action": "accept", "content": {"date": "2026-09-14"}}
            return await agent.ainvoke(Command(resume={"responses": {question["key"]: answer}}), config)

    result = asyncio.run(run())
    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "2026-09-14" in tool_msg.text


# ---------------------------------------------------------------------------
# 第 16 章 · Multi-agent
# ---------------------------------------------------------------------------
# 【补写】
def test_handoff_between_agents(monkeypatch):
    import ch16_multi_agent as ch

    # support 先交接给 sales，sales 再回答
    model = fake_model([call("transfer_to_sales_agent", {}), AIMessage("The pro plan is $20/month.")])
    monkeypatch.setattr(ch, "get_model", lambda *a, **k: model)
    graph = ch.build_handoff_graph()
    result = graph.invoke(ask("I want to buy pro"), {"configurable": {"thread_id": "h"}})
    assert result["active_agent"] == "sales_agent"
    assert result["messages"][-1].text == "The pro plan is $20/month."


# 【补写】
def test_single_agent_handoff_changes_step():
    import ch16_multi_agent as ch

    model = fake_model([call("record_warranty_status", {"status": "in_warranty"}), AIMessage("Let's book a repair.")])
    agent = create_agent(model, tools=[ch.record_warranty_status, ch.schedule_repair], state_schema=ch.SupportState, middleware=[ch.step_router])
    result = agent.invoke(ask("broken screen, under warranty"))
    assert result["current_step"] == "specialist"
    assert result["warranty_status"] == "in_warranty"


# ---------------------------------------------------------------------------
# 第 18 章 · SQL agent（真实的 Chinook 数据库）
# ---------------------------------------------------------------------------
# 【补写】
def test_sql_tools_on_chinook():
    import ch18_sql_agent as ch

    assert "Artist" in ch.sql_db_list_tables.invoke({})
    assert "CREATE TABLE" in ch.sql_db_schema.invoke({"table_names": "Genre"})
    assert ch.sql_db_query.invoke({"query": "SELECT COUNT(*) FROM Genre"}) == "[(25,)]"
    assert ch.sql_db_query.invoke({"query": "SELECT nope FROM Genre"}).startswith("Error")
