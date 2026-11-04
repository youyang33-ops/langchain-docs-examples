"""综合项目的端到端测试：剧本式假模型 → 存笔记 → 检索 → 删除时人工审批。

代码来源标注（每个类、函数和主要常量前面都有一行）：
  【文档原代码】   和文档的代码完全一致（只比较语法，不比较注释和排版）
  【补写】         文档没有给出这段代码，是我写的
这个文件是为学习项目写的，不是文档原有的内容；"参考"表示只借用了文档的写法。
"""

from langchain.messages import AIMessage, ToolCall

from common import fake_model


# 【补写】
def call(name, args, id):
    return AIMessage(content="", tool_calls=[ToolCall(name=name, args=args, id=id)])


# 【补写】
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


# 【补写】
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
