"""第 15 步配套的 MCP 服务器：一个 LangChain 术语表。

它是一个独立的程序，step15_mcp.py 会通过 stdio 把它作为子进程启动。
任何支持 MCP 的客户端（Claude Desktop、Cursor……）也都能用它。
单独测试：python progressive/step15_mcp_server.py   （会等待 stdin 上的 MCP 消息）

代码来源标注（每个类、函数和主要常量前面都有一行）：
  【文档原代码】   和文档的代码完全一致（只比较语法，不比较注释和排版）
  【补写】         文档没有给出这段代码，是我写的
这个文件是为学习项目写的，不是文档原有的内容；"参考"表示只借用了文档的写法。
"""

from fastmcp import FastMCP

server = FastMCP("langchain-glossary")

# 【补写】术语表
GLOSSARY = {
    "agent": "A model running in a loop, calling tools until it can answer. Built with create_agent.",
    "tool": "A function the model can choose to call. Defined with @tool; its docstring is the description.",
    "middleware": "Hooks around the agent loop (before_model, wrap_tool_call, ...) for retries, guardrails, prompts.",
    "checkpointer": "Saves agent state after every step, keyed by thread_id. Gives short-term memory.",
    "store": "Long-term key-value memory organized by namespace, shared across threads. Supports semantic search.",
    "context": "Read-only per-invocation data (like user_id) passed with context=; tools read it via ToolRuntime.",
    "interrupt": "A pause in the graph waiting for human input; resume with Command(resume=...).",
    "mcp": "Model Context Protocol: a standard way to expose tools to any AI application.",
}


# 【补写】术语表 MCP 工具
@server.tool
def lookup_term(term: str) -> str:
    """Look up the definition of a LangChain term in the official glossary."""
    definition = GLOSSARY.get(term.strip().lower())
    return definition or f"'{term}' is not in the glossary. Known terms: {', '.join(GLOSSARY)}"


# 【补写】术语表 MCP 工具
@server.tool
def list_terms() -> list[str]:
    """List every term available in the glossary."""
    return sorted(GLOSSARY)


# 【补写】命令行入口
if __name__ == "__main__":
    server.run()
