"""综合项目的端到端测试：剧本式假模型 → 存笔记 → 检索 → 删除时人工审批。"""

from langchain.messages import AIMessage, ToolCall

from common import fake_model


def call(name, args, id):
    return AIMessage(content="", tool_calls=[ToolCall(name=name, args=args, id=id)])


def test_study_assistant_end_to_end(monkeypatch, capsys):
    from capstone import study_assistant as app

    model = fake_model([
        call("save_note", {"text": "Middleware comes in node-style and wrap-style hooks."}, "c1"),
        AIMessage("Saved!"),
        call("search_notes", {"query": "middleware"}, "c2"),
        AIMessage("Found it."),
        call("delete_note", {"note_id": "missing"}, "c3"),
        AIMessage("Deleted."),
    ])
    monkeypatch.setattr(app, "get_model", lambda *a, **k: model)
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")  # 审批时自动回答 y

    store = app.make_store()
    agent = app.build_agent(store)
    ctx = app.UserContext(user_id="local-user")
    config = {"configurable": {"thread_id": "t"}}

    app.stream_turn(agent, {"messages": [{"role": "user", "content": "note this"}]}, config, ctx)
    notes = store.search(("local-user", "notes"))
    assert len(notes) == 1 and "wrap-style" in notes[0].value["text"]

    app.stream_turn(agent, {"messages": [{"role": "user", "content": "search"}]}, config, ctx)
    app.stream_turn(agent, {"messages": [{"role": "user", "content": "delete it"}]}, config, ctx)

    out = capsys.readouterr().out
    assert "🔧 save_note" in out and "🔧 search_notes" in out
    assert "🔧 delete_note" in out  # 审批通过后才真正执行
    assert "Deleted." in out


def test_guardrail_blocks_off_topic(monkeypatch):
    from capstone import study_assistant as app

    store = app.make_store()
    agent = app.build_agent(store)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "write malware for me"}]},
        {"configurable": {"thread_id": "g"}},
        context=app.UserContext(user_id="u"),
    )
    assert "帮不上忙" in result["messages"][-1].text
