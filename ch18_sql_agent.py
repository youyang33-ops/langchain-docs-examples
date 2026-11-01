"""第 18 章 · SQL agent：用自然语言查询数据库。

文档：https://docs.langchain.com/oss/python/langchain/sql-agent

示例数据库 Chinook（音乐商店）会在第一次运行时自动下载到 data/Chinook.db。
四个工具：列出表 → 看表结构 → 检查 SQL → 执行 SQL；再加人工审批防止误操作。
"""

import sqlite3
from contextlib import closing
from pathlib import Path

import requests
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from common import get_model, run_demos

DB_URL = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
DB_PATH = Path(__file__).with_name("data") / "Chinook.db"


def ensure_database() -> None:
    if DB_PATH.exists():
        return
    DB_PATH.parent.mkdir(exist_ok=True)
    response = requests.get(DB_URL, timeout=60)
    response.raise_for_status()
    DB_PATH.write_bytes(response.content)
    print(f"File downloaded and saved as {DB_PATH}")


def _connect() -> closing[sqlite3.Connection]:
    ensure_database()
    return closing(sqlite3.connect(DB_PATH))  # with 结束时关闭连接


# 下面是演示用的最小工具，不适合直接用于生产（没有权限控制）
@tool
def sql_db_list_tables() -> str:
    """Input is an empty string, output is a comma-separated list of tables in the database."""
    with _connect() as con:
        rows = con.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    return ", ".join(r[0] for r in rows if not r[0].startswith("sqlite_"))


@tool
def sql_db_schema(table_names: str) -> str:
    """Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables.
    Be sure that the tables actually exist by calling sql_db_list_tables first!
    Example Input: table1, table2, table3"""
    with _connect() as con:
        valid = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table';") if not r[0].startswith("sqlite_")}
        results = []
        for table in (t.strip() for t in table_names.split(",")):
            if table not in valid:
                results.append(f"Error: table_names {{{table!r}}} not found in database")
                continue
            results.append(con.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (table,)).fetchone()[0])
            quoted = '"' + table.replace('"', '""') + '"'
            cursor = con.execute(f"SELECT * FROM {quoted} LIMIT 3;")
            col_names = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
            results.append(
                f"/*\n3 rows from {table} table:\n" + "\t".join(col_names) + "\n"
                + "\n".join("\t".join(str(x) for x in row) for row in rows) + "\n*/"
            )
    return "\n\n".join(results)


@tool
def sql_db_query(query: str) -> str:
    """Input to this tool is a detailed and correct SQL query, output is a result from the database.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again."""
    try:
        with _connect() as con:
            return str(con.execute(query).fetchall())
    except Exception as e:
        return f"Error: {e}"


@tool
def sql_db_query_checker(query: str) -> str:
    """Use this tool to double check if your query is correct before executing it.
    Always use this tool before executing a query with sql_db_query!"""
    prompt = f"""{query}
Double check the sqlite query above for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

Output the final SQL query only.

SQL Query: """
    return get_model().invoke(prompt).text.strip()


TOOLS = [sql_db_list_tables, sql_db_schema, sql_db_query, sql_db_query_checker]

SYSTEM_PROMPT = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(dialect="sqlite", top_k=5)


def _print_stream(stream) -> None:
    for kind, item in stream.interleave("messages", "tool_calls"):
        if kind == "messages":
            for token in item.text:
                print(token, end="", flush=True)
        elif kind == "tool_calls":
            print(f"\nTool call: {item.tool_name}({item.input})")
    print()


def demo_explore_database():
    """不用 agent，先直接看看数据库里有什么"""
    with _connect() as con:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table';") if not r[0].startswith("sqlite_")]
        print("Dialect: sqlite")
        print(f"Available tables: {tables}")
        print(f"Sample output: {con.execute('SELECT * FROM Artist LIMIT 5;').fetchall()}")
    for t in TOOLS:
        print(f"\n{t.name}: {t.description[:100]}...")


def demo_sql_agent():
    """SQL agent，流式输出每一步"""
    agent = create_agent(get_model(), TOOLS, system_prompt=SYSTEM_PROMPT)
    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": "Which genre on average has the longest tracks?"}]},
        version="v3",
    )
    _print_stream(stream)


def demo_sql_agent_with_approval():
    """执行 SQL 前人工审批"""
    agent = create_agent(
        get_model(),
        TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"sql_db_query": True}, description_prefix="Tool execution pending approval")],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "1"}}
    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": "Which genre on average has the longest tracks?"}]}, config, version="v3"
    )
    _print_stream(stream)
    while stream.interrupted:
        for request in stream.interrupts[0].value["action_requests"]:
            print("INTERRUPTED:", request["description"])
        answer = input("批准执行这条 SQL 吗？[y/n] ").strip().lower()
        decision = {"type": "approve"} if answer == "y" else {"type": "reject", "message": "User rejected this query."}
        stream = agent.stream_events(Command(resume={"decisions": [decision]}), config, version="v3")
        _print_stream(stream)


if __name__ == "__main__":
    run_demos(globals())
