"""第 17 章 · Knowledge base：语义搜索引擎 + RAG agent。

文档：https://docs.langchain.com/oss/python/langchain/knowledge-base

流程：Document → 切分（TextSplitter）→ 向量化（Embeddings）→ 存入 VectorStore
      → similarity_search / Retriever → 作为工具交给 agent（Agentic RAG）

示例数据是 Nike 2023 年报 PDF；第一次运行 download_pdf 会下载到 data/ 目录：
    python ch17_knowledge_base_rag.py download_pdf
没有 PDF 时自动退回到几段内置文本，保证离线也能跑。
"""

import os
import urllib.request
from pathlib import Path

import pypdf
from langchain.agents import create_agent
from langchain.embeddings import init_embeddings
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.runnables import chain
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common import FAKE, get_model, last_text, run_demos

PDF_URL = "https://raw.githubusercontent.com/langchain-ai/langchain/v0.3/docs/docs/example_data/nke-10k-2023.pdf"
PDF_PATH = Path(__file__).with_name("data") / "nke-10k-2023.pdf"
EMBEDDING_MODEL = os.getenv("LC_EMBEDDINGS", "openai:text-embedding-3-small")

FALLBACK_DOCS = [
    Document(page_content="Dogs are great companions, known for their loyalty and friendliness.", metadata={"source": "mammal-pets-doc"}),
    Document(page_content="Cats are independent pets that often enjoy their own space.", metadata={"source": "mammal-pets-doc"}),
    Document(page_content="Goldfish are popular pets for beginners, requiring relatively simple care.", metadata={"source": "fish-pets-doc"}),
    Document(page_content="Parrots are intelligent birds capable of mimicking human speech.", metadata={"source": "bird-pets-doc"}),
    Document(page_content="Rabbits are social animals that need plenty of space to hop around.", metadata={"source": "mammal-pets-doc"}),
]


def get_embeddings():
    """真实模式用 init_embeddings；假模型模式用确定性的假向量（只保证能跑，不保证相关性）"""
    if FAKE:
        return DeterministicFakeEmbedding(size=256)
    return init_embeddings(EMBEDDING_MODEL)


def load_pdf_pages(file_path: str) -> list[Document]:
    reader = pypdf.PdfReader(file_path)
    return [
        Document(page_content=page.extract_text() or "", metadata={"source": file_path, "page": i})
        for i, page in enumerate(reader.pages)
    ]


def load_documents() -> list[Document]:
    if PDF_PATH.exists():
        return load_pdf_pages(str(PDF_PATH))
    print(f"（没有找到 {PDF_PATH.name}，使用内置示例文本。运行 download_pdf 可下载年报。）")
    return FALLBACK_DOCS


def build_vector_store() -> InMemoryVectorStore:
    docs = load_documents()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    all_splits = text_splitter.split_documents(docs)
    print(f"{len(docs)} 个文档 → 切成 {len(all_splits)} 块")
    vector_store = InMemoryVectorStore(get_embeddings())
    vector_store.add_documents(documents=all_splits)
    return vector_store


def demo_download_pdf():
    """下载示例 PDF（Nike 2023 10-K，约 2.4MB）到 data/"""
    PDF_PATH.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve(PDF_URL, PDF_PATH)
    print(f"已保存到 {PDF_PATH}")


def demo_documents_and_embeddings():
    """Document 结构 + 向量长度"""
    embeddings = get_embeddings()
    vector_1 = embeddings.embed_query(FALLBACK_DOCS[0].page_content)
    vector_2 = embeddings.embed_query(FALLBACK_DOCS[1].page_content)
    assert len(vector_1) == len(vector_2)
    print(f"Generated vectors of length {len(vector_1)}\n")
    print(vector_1[:10])


def demo_similarity_search():
    """按字符串搜、带分数搜、按向量搜"""
    vector_store = build_vector_store()
    pdf = PDF_PATH.exists()
    q1 = "How many distribution centers does Nike have in the US?" if pdf else "Which pet can talk?"
    q2 = "What was Nike's revenue in 2023?" if pdf else "Which pet is good for beginners?"
    q3 = "How were Nike's margins impacted in 2023?" if pdf else "Which animal likes its own space?"

    print("[search]", vector_store.similarity_search(q1)[0].page_content[:300], "\n")
    doc, score = vector_store.similarity_search_with_score(q2)[0]
    print(f"[score={score:.3f}]", doc.page_content[:300], "\n")
    embedding = get_embeddings().embed_query(q3)
    print("[by vector]", vector_store.similarity_search_by_vector(embedding)[0].page_content[:300])


async def demo_async_search():
    """异步版本 asimilarity_search"""
    vector_store = build_vector_store()
    results = await vector_store.asimilarity_search("When was Nike incorporated?")
    print(results[0].page_content[:300])


def demo_retrievers():
    """@chain 自定义 retriever 与 vector_store.as_retriever()"""
    vector_store = build_vector_store()

    @chain
    def retriever(query: str) -> list[Document]:
        return vector_store.similarity_search(query, k=1)

    questions = ["Which pet can talk?", "Which pet needs space to hop?"]
    print([d[0].page_content for d in retriever.batch(questions)])

    retriever2 = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    print([d[0].page_content for d in retriever2.batch(questions)])


def demo_rag_agent():
    """Agentic RAG：把检索包装成工具，agent 自己决定何时检索、检索什么"""
    vector_store = build_vector_store()

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information to help answer a query."""
        docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(f"Source: {d.metadata}\nContent: {d.page_content}" for d in docs)
        return serialized, docs  # content 给模型看，artifact（原始 Document）留给程序用

    agent = create_agent(
        get_model(),
        tools=[retrieve_context],
        system_prompt="You have access to a tool that retrieves context from a document. Use it to answer user queries. Cite the page if available.",
    )
    question = "What was Nike's revenue in 2023?" if PDF_PATH.exists() else "Which pets are good for beginners?"
    print(last_text(agent.invoke({"messages": [{"role": "user", "content": question}]})))


if __name__ == "__main__":
    run_demos(globals())
