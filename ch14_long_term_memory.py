"""第 14 章 · Long-term memory：跨对话的记忆（Store）。

文档：https://docs.langchain.com/oss/python/langchain/long-term-memory

- 短期记忆（checkpointer）按 thread_id 隔离；长期记忆（store）按 namespace + key 存 JSON
- store.put / get / search；配置 embedding 后 search 支持语义检索
- 工具通过 runtime.store 读写；生产环境用 PostgresStore

代码来源标注（每个类、函数和主要常量前面都有一行）：
  【文档原代码】          和文档的代码完全一致（只比较语法，不比较注释和排版）
  【文档原代码·有改动】   来自文档，做了改动（标注里写了改了什么）
  【补写】                文档没有给出这段代码，是我写的
来源写成"页面 · 小节"，页面相对于 https://docs.langchain.com/oss/python/langchain/
注意：overview 页只有 get_weather 这一个例子，其余代码来自它链接的其他页面。
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


# 【文档原代码·有改动】long-term-memory · Memory storage：包成函数；打印结果
def demo_store_basics():
    """put / get / search，以及带向量索引的语义搜索"""
    # 【文档原代码】long-term-memory · Memory storage
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


# 【补写】用 Embeddings 对象做语义检索
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


# 【文档原代码】long-term-memory · Read long-term memory in tools
@dataclass
class Context:
    user_id: str


# 【文档原代码】long-term-memory · Write long-term memory from tools
class UserInfo(TypedDict):
    name: str


# 【文档原代码】long-term-memory · Read long-term memory in tools
@tool
def get_user_info(runtime: ToolRuntime[Context]) -> str:
    """Look up user info."""
    assert runtime.store is not None
    user_info = runtime.store.get(("users",), runtime.context.user_id)
    return str(user_info.value) if user_info else "Unknown user"


# 【文档原代码】long-term-memory · Write long-term memory from tools
@tool
def save_user_info(user_info: UserInfo, runtime: ToolRuntime[Context]) -> str:
    """Save user info."""
    assert runtime.store is not None
    runtime.store.put(("users",), runtime.context.user_id, dict(user_info))
    return "Successfully saved user info."


# 【文档原代码·有改动】long-term-memory · Read long-term memory in tools：模型改成 get_model()；工具移到模块级
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


# 【文档原代码·有改动】long-term-memory · Write long-term memory from tools：模型改成 get_model()；多加了第二次对话
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


# 【文档原代码·有改动】long-term-memory · Read long-term memory in tools：不运行，只打印 Postgres 版示例
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


# 【补写】命令行入口
if __name__ == "__main__":
    run_demos(globals())
