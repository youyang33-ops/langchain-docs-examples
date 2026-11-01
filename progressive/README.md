# 渐进式项目：15 步做出一个完整的学习助手

章节文件（`ch01`–`ch19`）按 API 分类讲解；这里换一种学法：**从零开始做同一个应用**，每一步只加一个新概念。

- 每一步都是**完整、可独立运行**的文件，不依赖上一步的文件。
- 和上一步相比新增或修改的行，都在行尾或上一行标了 `# ★ 新增` / `# ★ 修改`。
- 解释性注释只写在引入该概念的那一步；后面的步骤保持干净，越往后越接近真实项目的样子。

## 怎么学

```bash
python progressive/step01_chat_model.py          # 运行某一步，和它聊几句
```

```bash
diff -u progressive/step03_short_term_memory.py progressive/step04_streaming.py   # 看这一步改了什么
```

建议节奏：先读文件开头的注释（目标、知识点、"试试"），运行并照着"试试"操作，然后和上一步做 diff，最后去对应的章节文件看更多用法。
VS Code 里可以选中两个文件 → 右键 → "Compare Selected"，看起来比命令行 diff 更清楚。

## 15 步路线

| 步 | 文件 | 新增了什么 | 对应章节 |
|---|---|---|---|
| 1 | `step01_chat_model.py` | 直接调用聊天模型，自己维护消息历史，流式输出 | ch02、ch03 |
| 2 | `step02_first_agent.py` | `create_agent` + 两个 `@tool`（记笔记、列笔记） | ch01、ch04、ch05 |
| 3 | `step03_short_term_memory.py` | `checkpointer` + `thread_id`：记住本次对话；`/new` | ch06 |
| 4 | `step04_streaming.py` | `stream_events(version="v3")`：实时显示 token 和工具调用 | ch07 |
| 5 | `step05_runtime_context.py` | `context_schema` + `ToolRuntime`：按用户分开存笔记；`/user` | ch04、ch09 |
| 6 | `step06_long_term_memory.py` | `Store`：笔记和用户资料跨对话保存；`/memory` | ch14 |
| 7 | `step07_dynamic_prompt.py` | `@dynamic_prompt`：根据用户资料生成系统提示词 | ch09、ch11 |
| 8 | `step08_semantic_search.py` | 向量索引 + `search_notes`：按意思检索笔记（RAG） | ch14、ch17 |
| 9 | `step09_structured_output.py` | `make_quiz`：用 Pydantic 结构化输出出选择题 | ch02、ch08 |
| 10 | `step10_middleware.py` | 摘要、重试、工具调用上限 + 自己写的计时中间件 | ch10、ch11 |
| 11 | `step11_guardrails.py` | `@before_agent` 关键词拦截 + PII 脱敏 | ch12 |
| 12 | `step12_human_in_the_loop.py` | `delete_note` 执行前在终端里审批 | ch13 |
| 13 | `step13_subagent.py` | `research` 子 agent | ch16 |
| 14 | `step14_persistence.py` | SQLite 持久化：关掉程序再打开，记忆还在 | ch06、ch14 |
| 15 | `step15_mcp.py` + `step15_mcp_server.py` | 通过 MCP 接入独立的术语表服务器；整个程序改成 async | ch15 |

第 15 步就是最终的完整应用。它和 `capstone/study_assistant.py` 功能相近，但多了持久化和 MCP。

## 依赖

第 1–13 步只需要根目录 `requirements.txt` 里的包。第 14、15 步还要装：

```bash
pip install langgraph-checkpoint-sqlite
```

第 14、15 步的数据保存在 `progressive/data/`。删掉这个目录就能从头开始。

## 离线运行

`.env` 里设置 `LC_MODEL=fake` 后，每一步都能离线跑通，但回答没有实际内容。
`tests/test_progressive.py` 用"剧本式"假模型验证了每一步新增的功能：

```bash
pytest tests/test_progressive.py -v
```

## 写代码时踩到的坑

- **`item.output` 是 `None`**：`stream.interleave` 给出工具调用时，工具可能还没执行完；要先把 `item.output_deltas` 读完（第 4 步）。
- **护栏拦截后看不到回复**：被 `before_agent` 拦截时，回复不是模型生成的，流里没有 token，要从 `agent.get_state(config)` 里取（第 11 步）。
- **改成 async 后报错**（第 15 步）：
  - 在事件循环里访问 store，要用 `aget` / `asearch`；
  - `@wrap_tool_call` 必须写成 `async def`；
  - 异步流没有 `interleave`，`interrupted()` 和 `interrupts()` 要 `await`。
- **MCP 服务器在 pytest 里启动失败**：stdio 子进程需要真实的 stderr，测试里要用 `capfd`，不能用 `capsys`。
