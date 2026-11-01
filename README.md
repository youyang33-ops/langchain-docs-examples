# LangChain 1.x 学习项目

这个项目把 [LangChain Python 文档](https://docs.langchain.com/oss/python/langchain/overview)里的核心代码整理成一套**能直接运行**的项目：

- **19 个章节文件，共 126 个 demo**：顺序与文档目录一致，每个 demo 对应文档里的一段代码，注释是中文。
- **一个综合项目** `capstone/study_assistant.py`：把各章知识点串成一个能用的命令行"学习助手"。
- **一套测试** `tests/`：既是文档里 Testing 章节的示例，也用来验证本项目的代码。全部离线运行。

代码基于 langchain 1.4.7 / langgraph 1.2.18（2026 年 10 月）编写并测试通过。

## 快速开始

```bash
conda create -n lc python=3.11 -y && conda activate lc
pip install -r requirements.txt
cp .env.example .env          # 然后填写 LC_MODEL 和对应的 API key
```

```bash
python ch01_quickstart.py              # 列出本章所有 demo
python ch01_quickstart.py basic_agent  # 运行一个 demo
python ch04_tools.py all               # 依次运行本章全部 demo
```

**没有 API key 也能学：** 在 `.env` 里写 `LC_MODEL=fake`，所有 demo 都会改用离线假模型运行。
这样可以先把代码结构和执行流程看明白，但回答内容没有意义，结构化输出相关的 demo 也会跳过。

## 学习路线

| 章节 | 文件 | 对应文档 | 学到什么 |
|---|---|---|---|
| 1 | `ch01_quickstart.py` | [Overview](https://docs.langchain.com/oss/python/langchain/overview)、[Quickstart](https://docs.langchain.com/oss/python/langchain/quickstart) | 第一个 agent；Agent = 模型 + 外壳（提示词、工具、中间件） |
| 2 | `ch02_models.py` | [Models](https://docs.langchain.com/oss/python/langchain/models) | `init_chat_model`、invoke / stream / batch、bind_tools、结构化输出、token 用量 |
| 3 | `ch03_messages.py` | [Messages](https://docs.langchain.com/oss/python/langchain/messages) | 四种消息、`content_blocks`、多模态、序列化 |
| 4 | `ch04_tools.py` | [Tools](https://docs.langchain.com/oss/python/langchain/tools) | `@tool`、`ToolRuntime`（state / context / store）、`Command`、动态工具 |
| 5 | `ch05_agents.py` | [Agents](https://docs.langchain.com/oss/python/langchain/agents) | `create_agent` 全部参数 |
| 6 | `ch06_short_term_memory.py` | [Short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory) | checkpointer、thread_id、裁剪 / 删除 / 摘要消息 |
| 7 | `ch07_streaming.py` | [Streaming](https://docs.langchain.com/oss/python/langchain/streaming) | `stream_events` v3、各种 stream_mode、流式中的人工审批 |
| 8 | `ch08_structured_output.py` | [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output) | ProviderStrategy、ToolStrategy、出错重试 |
| 9 | `ch09_runtime_and_context.py` | [Runtime](https://docs.langchain.com/oss/python/langchain/runtime)、[Context engineering](https://docs.langchain.com/oss/python/langchain/context-engineering) | 用 State / Store / Context 动态调整提示词、工具、模型、输出格式 |
| 10 | `ch10_middleware_builtin.py` | [Middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in) | 内置中间件：重试、降级、限流、PII、摘要、Shell、文件搜索…… |
| 11 | `ch11_middleware_custom.py` | [Custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom) | 节点式 / 包裹式钩子、改 state、jump_to |
| 12 | `ch12_guardrails.py` | [Guardrails](https://docs.langchain.com/oss/python/langchain/guardrails) | 确定性护栏、模型审查护栏、多层组合 |
| 13 | `ch13_human_in_the_loop.py` | [Human-in-the-loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop) | approve / edit / reject / respond，按条件或逐个调用中断（可在终端里亲自审批） |
| 14 | `ch14_long_term_memory.py` | [Long-term memory](https://docs.langchain.com/oss/python/langchain/long-term-memory) | Store、namespace、语义检索 |
| 15 | `ch15_mcp.py` + `ch15_mcp_server.py` | [MCP](https://docs.langchain.com/oss/python/langchain/mcp/index) | `MCPAdapter`、自己写 FastMCP 服务器、危险工具审批、服务器中途向用户提问 |
| 16 | `ch16_multi_agent.py` | [Multi-agent](https://docs.langchain.com/oss/python/langchain/multi-agent/index) | Subagents、Handoffs、Skills、Router、自定义工作流、Deep Agents |
| 17 | `ch17_knowledge_base_rag.py` | [Knowledge base](https://docs.langchain.com/oss/python/langchain/knowledge-base) | 切分 → 向量化 → 检索 → Agentic RAG |
| 18 | `ch18_sql_agent.py` | [SQL agent](https://docs.langchain.com/oss/python/langchain/sql-agent) | 用自然语言查 Chinook 数据库（首次运行自动下载） |
| 19 | `ch19_observability.py` | [Observability](https://docs.langchain.com/oss/python/langchain/observability) | LangSmith 追踪、tags / metadata |
| 20 | `tests/` | [Testing](https://docs.langchain.com/oss/python/langchain/test/index) | 假模型单元测试、集成测试、轨迹评估 |
| ★ | `capstone/study_assistant.py` | — | 综合项目 |

建议学习顺序：1 → 2 → 3 → 4 → 5 打基础；6、7、8 是日常最常用的功能；9 → 10 → 11 → 12 → 13 是"控制 agent 行为"的核心；14 → 18 是具体应用；最后读综合项目和 `tests/`。

## 项目结构

```
common.py                  # 共用配置：从 .env 读取模型、离线假模型、命令行 demo 运行器
ch01_… ~ ch19_….py         # 各章节
capstone/study_assistant.py
tests/
  test_unit_testing_docs.py      # 文档 Unit testing 章节的例子
  test_chapters.py               # 用"剧本式"假模型验证各章关键路径（审批、交接、MCP……）
  test_capstone.py               # 综合项目端到端测试
  test_integration_and_evals.py  # 真实模型测试（默认跳过）
data/                      # 第 17、18 章下载的数据（自动生成）
```

## 综合项目：学习助手

```bash
python capstone/study_assistant.py
```

可以帮你记笔记、语义检索笔记、出选择题、调用子 agent 讲解概念；它会记住你的名字和偏好，删除笔记前会先请你确认。
文件开头的注释列出了每个功能用到了哪一章的知识点。

## 测试

```bash
pytest                                                    # 离线测试，几秒钟跑完
LC_MODEL=openai:gpt-5.4-mini pytest -m integration -v     # 用真实模型（会产生费用）
```

`tests/test_chapters.py` 很值得一读：假模型会按预先写好的顺序回复，包括工具调用，所以人工审批、Agent 交接这类流程可以写成确定的测试。

## 和文档不一致的地方

文档里有几处代码在当前版本跑不通，这里已经修正，代码注释里也做了说明：

| 位置 | 文档写法 | 实际情况 |
|---|---|---|
| Middleware · PII | `detector=re.compile(...)` | 只接受正则字符串或函数，传编译好的 Pattern 会报 TypeError |
| Guardrails · 组合 | 两个 `PIIMiddleware("email")` 分别管输入和输出 | 同名中间件不能重复添加，要合并成一个 `apply_to_input=True, apply_to_output=True` |
| Middleware · Filesystem | `StoreBackend()` | 新版 deepagents 必须传 `namespace=` |
| MCP · Elicitation | 服务器端示例未给出 | 新版 MCP 协议下 `ctx.elicit()` 不可用，需要返回 `InputRequiredResult`（见 `ch15_mcp_server.py`） |
| Human-in-the-loop · edit | — | 被人工修改过参数的调用，ToolMessage 前面会自动加一段说明 |

## 没有收录的内容

- **重复的代码**：文档里同一段代码会给 OpenAI、Anthropic、Gemini、Bedrock 等十几个供应商各写一遍。本项目只保留一份，换供应商改 `.env` 里的 `LC_MODEL` 即可。
- **前端页面**（`frontend/*`、Generative UI 等）：都是 TypeScript / React 代码，和 Python 项目不是一类。
- **部署、Studio、Voice agent**：需要 LangSmith 部署或语音服务，不适合作为本地学习示例。
- **Postgres 版本**的 checkpointer / store：只以打印示例代码的方式给出（第 6、14 章），因为需要本地数据库。
