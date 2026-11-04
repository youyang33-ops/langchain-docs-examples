"""第 15 章配套的 MCP 服务器（FastMCP）。

既可以被 ch15_mcp.py 以"进程内"方式直接导入，也可以单独启动：
    python ch15_mcp_server.py           # stdio 传输（被 MCPAdapter(Path(...)) 拉起）
    python ch15_mcp_server.py http      # HTTP 传输，地址 http://127.0.0.1:8000/mcp

代码来源标注（每个类、函数和主要常量前面都有一行）：
  【文档原代码】   和文档的代码完全一致（只比较语法，不比较注释和排版）
  【补写】         文档没有给出这段代码，是我写的
这个文件是为学习项目写的，不是文档原有的内容；"参考"表示只借用了文档的写法。
"""

import sys

import mcp.types as mcp_types
from fastmcp import Context, FastMCP

server = FastMCP("demo-tools")


# 【补写】文档没给服务器端代码
@server.tool
def get_forecast(city: str) -> str:
    """Get the weather forecast for a city."""
    return f"Forecast for {city}: 18°C, light rain in the afternoon."


# 【补写】文档没给服务器端代码
@server.tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        # 服务器报错会变成 status="error" 的 ToolMessage，模型能读到并自我纠正
        raise ValueError("Cannot divide by zero")
    return a / b


# 【补写】文档没给服务器端代码
@server.tool
def lookup_user(user_id: int) -> dict:
    """Look up a user record by ID."""
    # 返回 dict 时 FastMCP 会生成 structured content，出现在 ToolMessage.artifact 里
    return {"id": user_id, "name": "Ada Lovelace", "plan": "pro"}


# 【补写】文档没给服务器端代码
@server.tool(annotations={"destructiveHint": True, "readOnlyHint": False})
def delete_record(record_id: int) -> str:
    """Permanently delete a record."""
    return f"Record {record_id} deleted."


# 【补写】文档没给服务器端代码；写法来自 FastMCP 源码里的 InputRequiredResult
@server.tool
async def book_table(party_size: int, ctx: Context) -> str | mcp_types.InputRequiredResult:
    """Book a restaurant table. Asks the user for a date if needed."""
    # Elicitation（工具执行中途向用户提问）。新版 MCP 协议是无状态的，服务器不能主动
    # 回调客户端，所以 ctx.elicit() 不可用；改用"guard"写法：
    #   第一轮：返回 InputRequiredResult，列出要问的问题 → LangChain 把它变成 interrupt
    #   第二轮：客户端带着答案重新调用本工具，从 ctx.input_responses 读取答案
    responses = ctx.input_responses or {}
    if "date" not in responses:
        return mcp_types.InputRequiredResult(
            input_requests={
                "date": mcp_types.ElicitRequest(
                    params=mcp_types.ElicitRequestFormParams(
                        message="Which date should I book?",
                        requested_schema={
                            "type": "object",
                            "properties": {"date": {"type": "string", "description": "YYYY-MM-DD"}},
                            "required": ["date"],
                        },
                    )
                )
            }
        )
    answer = responses["date"]
    if answer.action != "accept":
        return "Booking cancelled."
    return f"Booked a table for {party_size} on {answer.content['date']}."


# 【补写】命令行入口
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "http":
        server.run(transport="http", host="127.0.0.1", port=8000)
    else:
        server.run()
