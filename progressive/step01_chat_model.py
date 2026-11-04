"""第 1 步 · 只有模型的聊天机器人

目标：先不用 agent，直接和聊天模型对话，自己维护消息历史。
新增知识点：init_chat_model、消息列表、model.stream 流式输出      （对应 ch02、ch03）
运行：python progressive/step01_chat_model.py
试试：说"我叫小明"，再问"我叫什么？"——它记得，因为我们每次都把完整历史发给模型。

代码来源标注（每个类、函数和主要常量前面都有一行）：
  【文档原代码】   和文档的代码完全一致（只比较语法，不比较注释和排版）
  【补写】         文档没有给出这段代码，是我写的
这个文件是为学习项目写的，不是文档原有的内容；"参考"表示只借用了文档的写法。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402

from common import get_model  # noqa: E402

# 【补写】学习助手的提示词
SYSTEM_PROMPT = "You are a friendly study assistant for someone learning LangChain. Reply in the user's language."


# 【补写】聊天一轮
def chat(model, history: list, text: str) -> None:
    """把用户输入加入历史，流式打印回复，再把回复加入历史。"""
    history.append(HumanMessage(text))
    print("助手: ", end="", flush=True)
    full = None
    for chunk in model.stream(history):
        print(chunk.text, end="", flush=True)
        full = chunk if full is None else full + chunk  # chunk 可以相加
    print()
    history.append(AIMessage(full.text))


# 【补写】命令行主循环
def main() -> None:
    model = get_model()
    history = [SystemMessage(SYSTEM_PROMPT)]
    print("学习助手 v1（输入 /quit 退出）")
    while True:
        try:
            text = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text == "/quit":
            break
        if text:
            chat(model, history, text)


# 【补写】命令行入口
if __name__ == "__main__":
    main()
