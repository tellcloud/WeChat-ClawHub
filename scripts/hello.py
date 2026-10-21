# script: name=hello; keywords=hello,问候,你好; help=/hello <名字> — 问候示例脚本
"""hello 示例脚本：演示 WeChat-ClawHub 的脚本执行协议。

协议约定（由 ``modules/script`` 定义，无需 import 任何项目代码）：

- **参数**：用户输入的参数按序挂在本脚本的 ``sys.argv``
  （``/hello 世界`` → ``argv = ["hello.py", "世界"]``）；
- **上下文**：stdin 收到一行 JSON，含 ``command`` / ``args`` /
  ``session_key`` / ``user_id`` / ``message_id`` / ``is_group`` 等字段，
  脚本可据此实现按会话 / 用户区分的逻辑（本脚本只读取不强制）；
- **回复**：stdout 打印的文本（去首尾空白后）就是回复给微信用户的内容；
- **诊断**：stderr 的内容不会打扰用户，仅进入日志；
- **失败**：非零退出码即失败，用户会收到「执行失败（退出码 N）」+ stderr
  末尾片段；``/hello boom`` 可用于演示该路径。

元信息注释（文件首行 ``# script:``）让本脚本被 ``scripts/`` 目录自动
扫描注册，支持键：``name`` / ``keywords`` / ``help`` / ``description`` /
``permission`` / ``timeout`` / ``enabled``，分号分隔。

在微信中试试::

    /hello 世界
    hello 世界          # 关键词「hello」同样触发
    问候 ClawHub        # 关键词「问候」同样触发
    /hello boom         # 演示失败路径（退出码 1）
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    # 1. 用户参数（argv[0] 是脚本自身路径）
    args = sys.argv[1:]
    name = args[0].strip() if args and args[0].strip() else "world"

    # 演示失败路径：/hello boom → 退出码 1，stderr 片段反馈给用户
    if name == "boom":
        print("演示失败路径：这是一个故意的错误", file=sys.stderr)
        return 1

    # 2. 执行上下文（stdin JSON；手动运行 / 无输入时容忍为空）
    context: dict = {}
    try:
        raw = sys.stdin.read()
        if raw.strip():
            context = json.loads(raw)
    except (OSError, ValueError):
        context = {}

    # 3. 回复文本：stdout 即回复
    reply = f"Hello, {name}！这是 hello 示例脚本的回复。"
    if context.get("is_group"):
        reply += "（群聊里也能用）"
    print(reply)

    # 4. 诊断信息进 stderr（不打扰用户）
    print(f"[hello] 收到上下文: command={context.get('command')!r} args={args!r}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
