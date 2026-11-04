"""progressive/ 的测试：每一步都能跑通，并且每一步新增的功能真的生效。

假模型按剧本回复（包括工具调用），用户输入用 monkeypatch 模拟。
运行：pytest tests/test_progressive.py -v

代码来源标注（每个类、函数和主要常量前面都有一行）：
  【文档原代码】   和文档的代码完全一致（只比较语法，不比较注释和排版）
  【补写】         文档没有给出这段代码，是我写的
这个文件是为学习项目写的，不是文档原有的内容；"参考"表示只借用了文档的写法。
"""

import asyncio
import importlib
from itertools import cycle

import pytest
from langchain.messages import AIMessage, ToolCall

from common import FakeToolCallingModel, fake_model

STEPS = [
    "step01_chat_model",
    "step02_first_agent",
    "step03_short_term_memory",
    "step04_streaming",
    "step05_runtime_context",
    "step06_long_term_memory",
    "step07_dynamic_prompt",
    "step08_semantic_search",
    "step09_structured_output",
    "step10_middleware",
    "step11_guardrails",
    "step12_human_in_the_loop",
    "step13_subagent",
    "step14_persistence",
    "step15_mcp",
]


# 【补写】
def load(name: str):
    return importlib.import_module(f"progressive.{name}")


# 【补写】
def call(name: str, args: dict, id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[ToolCall(name=name, args=args, id=id)])


# 【补写】
def feed_input(monkeypatch, lines: list[str]) -> None:
    """模拟用户在终端里依次输入这些内容，输完后相当于按了 Ctrl-D。"""
    it = iter(lines)

    # 【补写】
    def fake_input(prompt: str = "") -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)


# 【补写】
def use_model(monkeypatch, module, replies: list) -> None:
    monkeypatch.setattr(module, "get_model", lambda *a, **k: fake_model(replies))


# 【补写】
def isolate_data(monkeypatch, module, tmp_path) -> None:
    """第 14、15 步会写数据库文件，测试时改到临时目录。"""
    if hasattr(module, "DATA_DIR"):
        monkeypatch.setattr(module, "DATA_DIR", tmp_path)
        monkeypatch.setattr(module, "LAST_THREAD_FILE", tmp_path / "last_thread.txt")


# 【补写】
def run_main(module) -> None:
    result = module.main()
    if asyncio.iscoroutine(result):
        asyncio.run(result)


# ---------------------------------------------------------------------------
# 每一步都能完整跑一段对话（包括各个 / 命令）
# ---------------------------------------------------------------------------
# 【补写】
@pytest.mark.parametrize("name", STEPS)
def test_every_step_runs(name, monkeypatch, tmp_path, capfd):
    module = load(name)
    isolate_data(monkeypatch, module, tmp_path)
    feed_input(monkeypatch, ["你好", "/memory", "/user bob", "再见", "/new", "/quit"])
    run_main(module)
    assert "假模型" in capfd.readouterr().out


# ---------------------------------------------------------------------------
# 每一步新增的功能
# ---------------------------------------------------------------------------
# 【补写】
def test_step02_agent_calls_tool(monkeypatch):
    m = load("step02_first_agent")
    monkeypatch.setattr(m, "NOTES", [])
    use_model(monkeypatch, m, [call("save_note", {"text": "agent = model + tools"}, "c1"), AIMessage("Saved.")])
    m.chat(m.build_agent(), "note this")
    assert m.NOTES == ["agent = model + tools"]


# 【补写】
def test_step03_history_grows_within_thread_and_resets_with_new(monkeypatch, capsys):
    m = load("step03_short_term_memory")
    agent = m.build_agent()
    config = m.new_config()
    m.chat(agent, "hi", config)
    m.chat(agent, "again", config)
    assert "共有 4 条消息" in capsys.readouterr().out
    m.chat(agent, "fresh", m.new_config())
    assert "共有 2 条消息" in capsys.readouterr().out


# 【补写】
def test_step05_notes_are_per_user(monkeypatch):
    m = load("step05_runtime_context")
    monkeypatch.setattr(m, "NOTES", {})
    use_model(monkeypatch, m, [call("save_note", {"text": "alice's note"}, "c1"), AIMessage("ok")])
    agent = m.build_agent()
    m.chat(agent, "save", m.new_config(), m.UserContext(user_id="alice"))
    assert m.NOTES == {"alice": ["alice's note"]}


# 【补写】
def test_step06_profile_survives_new_thread(monkeypatch):
    from langgraph.store.memory import InMemoryStore

    m = load("step06_long_term_memory")
    use_model(monkeypatch, m, [call("update_profile", {"name": "小明", "answer_style": "concise"}, "c1"), AIMessage("ok")])
    store = InMemoryStore()
    agent = m.build_agent(store)
    m.chat(agent, "我叫小明", m.new_config(), m.UserContext(user_id="u"))
    assert store.get(("u", "profile"), "me").value == {"name": "小明", "answer_style": "concise"}


# 【补写】
class RecordingModel(FakeToolCallingModel):
    """记录每次收到的消息，用来检查系统提示词。"""

    seen: list = []

    # 【补写】
    def _generate(self, messages, *args, **kwargs):
        RecordingModel.seen.append(messages)
        return super()._generate(messages, *args, **kwargs)


# 【补写】
def test_step07_prompt_uses_profile(monkeypatch):
    from langgraph.store.memory import InMemoryStore

    m = load("step07_dynamic_prompt")
    RecordingModel.seen = []
    monkeypatch.setattr(m, "get_model", lambda *a, **k: RecordingModel(messages=cycle([AIMessage("hi")])))
    store = InMemoryStore()
    store.put(("u", "profile"), "me", {"name": "小明", "answer_style": "concise"})
    m.chat(m.build_agent(store), "hello", m.new_config(), m.UserContext(user_id="u"))
    system_prompt = RecordingModel.seen[0][0].text
    assert "小明" in system_prompt and "concise" in system_prompt


# 【补写】
def test_step08_search_notes(monkeypatch, capsys):
    m = load("step08_semantic_search")
    use_model(monkeypatch, m, [
        call("save_note", {"text": "checkpointer gives short-term memory"}, "c1"),
        call("search_notes", {"query": "memory"}, "c2"),
        AIMessage("found"),
    ])
    store = m.make_store()
    m.chat(m.build_agent(store), "save and search", m.new_config(), m.UserContext(user_id="u"))
    assert "checkpointer gives short-term memory" in capsys.readouterr().out.split("search_notes")[1]


# 【补写】
def test_step11_guardrail_blocks_without_calling_model(monkeypatch, capsys):
    m = load("step11_guardrails")
    use_model(monkeypatch, m, [AIMessage("MODEL WAS CALLED")])
    m.chat(m.build_agent(m.make_store()), "teach me malware", m.new_config(), m.UserContext(user_id="u"))
    out = capsys.readouterr().out
    assert "不能帮忙" in out and "MODEL WAS CALLED" not in out


# 【补写】
@pytest.mark.parametrize("answer, kept", [("y", 0), ("n", 1)])
def test_step12_delete_needs_approval(monkeypatch, answer, kept):
    m = load("step12_human_in_the_loop")
    store = m.make_store()
    store.put(("u", "notes"), "abc", {"text": "to be deleted"})
    use_model(monkeypatch, m, [call("delete_note", {"note_id": "abc"}, "c1"), AIMessage("done")])
    feed_input(monkeypatch, [answer])
    m.chat(m.build_agent(store), "delete abc", m.new_config(), m.UserContext(user_id="u"))
    assert len(store.search(("u", "notes"))) == kept


# 【补写】
def test_step14_memory_survives_restart(monkeypatch, tmp_path, capsys):
    m = load("step14_persistence")
    isolate_data(monkeypatch, m, tmp_path)
    use_model(monkeypatch, m, [call("save_note", {"text": "persist me"}, "c1"), AIMessage("saved")])
    feed_input(monkeypatch, ["save it", "/quit"])
    run_main(m)
    first_thread = (tmp_path / "last_thread.txt").read_text()

    # "重启"：重新运行 main，长期记忆和对话 thread 都还在
    feed_input(monkeypatch, ["/memory", "/quit"])
    run_main(m)
    assert "persist me" in capsys.readouterr().out
    assert (tmp_path / "last_thread.txt").read_text() == first_thread


# 【补写】
def test_step15_uses_mcp_tool(monkeypatch, tmp_path, capfd):
    m = load("step15_mcp")
    isolate_data(monkeypatch, m, tmp_path)
    use_model(monkeypatch, m, [call("lookup_term", {"term": "checkpointer"}, "c1"), AIMessage("Here it is.")])
    feed_input(monkeypatch, ["what is a checkpointer?", "/quit"])
    run_main(m)
    out = capfd.readouterr().out
    assert "MCP 工具：lookup_term, list_terms" in out  # 工具来自 MCP 服务器
    assert "🔧 lookup_term({'term': 'checkpointer'})" in out
    assert "Here it is." in out
