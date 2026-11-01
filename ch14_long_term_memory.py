"""第 14 章 · Long-term memory：跨对话的记忆（Store）。

文档：https://docs.langchain.com/oss/python/langchain/long-term-memory

- 短期记忆（checkpointer）按 thread_id 隔离；长期记忆（store）按 namespace + key 存 JSON
- store.put / get / search；配置 embedding 后 search 支持语义检索
- 工具通过 runtime.store 读写；生产环境用 PostgresStore
"""

from collections.abc import Sequence
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.embeddings import DeterministicFakeEmbedding
from langgraph.store.base import IndexConfig
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict

from common import get_model, last_text, run_demos


def demo_store_basics():
    """put / get / search，以及带向量索引的语义搜索"""
    def embed(texts: Sequence[str]) -> list[list[float]]:
        # 文档里的占位实现；真实项目换成 OpenAIEmbeddings 等
        return [[1.0, 2.0] for _ in texts]

    store = InMemoryStore(index=IndexConfig(embed=embed, dims=2))
    namespace = ("my-user", "chitchat")  # (用户 ID, 应用场景)
    store.put(
        namespace,
        "a-memory",
        {"rules": ["User likes short, direct language", "User only speaks English & python"], "my-key": "my-value"},
    )
    print("get:", store.get(namespace, "a-memory").value)
    items = store.search(namespace, filter={"my-key": "my-value"}, query="language preferences")
    print("search:", [(i.key, i.score) for i in items])


def demo_semantic_search():
    """用 LangChain 的 Embeddings 对象做真正的语义检索"""
    embeddings = DeterministicFakeEmbedding(size=256)  # 离线可跑；换成 OpenAIEmbeddings 效果更好
    store = InMemoryStore(index={"embed": embeddings, "dims": 256, "fields": ["text"]})
    ns = ("u1", "memories")
    for i, text in enumerate([
        "I prefer Python over Java",
        "My cat is named Mochi",
        "I study at UW-Madison",
    ]):
        store.put(ns, f"m{i}", {"text": text})
    for hit in store.search(ns, query="what programming language do I like?", limit=2):
        print(round(hit.score or 0, 3), hit.value["text"])


@dataclass
class Context:
    user_id: str


class UserInfo(TypedDict):
    name: str


@tool
def get_user_info(runtime: ToolRuntime[Context]) -> str:
    """Look up user info."""
    assert runtime.store is not None
    user_info = runtime.store.get(("users",), runtime.context.user_id)
    return str(user_info.value) if user_info else "Unknown user"


@tool
def save_user_info(user_info: UserInfo, runtime: ToolRuntime[Context]) -> str:
    """Save user info."""
    assert runtime.store is not None
    runtime.store.put(("users",), runtime.context.user_id, dict(user_info))
    return "Successfully saved user info."


def demo_read_in_tools():
    """预先写入 store，工具按 context.user_id 读取"""
    store = InMemoryStore()
    store.put(("users",), "user_123", {"name": "John Smith", "language": "English"})
    agent = create_agent(model=get_model(), tools=[get_user_info], store=store, context_schema=Context)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "look up user information"}]},
        context=Context(user_id="user_123"),
    )
    print(last_text(result))


def demo_write_in_tools():
    """agent 用工具把信息写进 store；新对话也能读到"""
    store = InMemoryStore()
    agent = create_agent(
        model=get_model(), tools=[save_user_info, get_user_info], store=store, context_schema=Context
    )
    ctx = Context(user_id="user_123")
    agent.invoke({"messages": [{"role": "user", "content": "My name is John Smith"}]}, context=ctx)
    print("store:", store.get(("users",), "user_123"))
    # 全新的对话（没有 checkpointer，不共享消息历史），仍然能从 store 读到
    result = agent.invoke({"messages": [{"role": "user", "content": "What's my name?"}]}, context=ctx)
    print(last_text(result))


def demo_postgres_store():
    """生产环境：PostgresStore（需要 Postgres 与 langgraph-checkpoint-postgres）"""
    print("""示例代码：

from langgraph.store.postgres import PostgresStore

DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"
with PostgresStore.from_conn_string(DB_URI) as store:
    store.setup()
    store.put(("users",), "user_123", {"name": "John Smith", "language": "English"})
    agent = create_agent(model, tools=[get_user_info], store=store, context_schema=Context)
""")


if __name__ == "__main__":
    run_demos(globals())
