"""Script 模块数据模型：脚本定义、执行参数与执行结果。

对应 README 目录结构中的 ``modules/script/models.py``，是 script 模块
唯一的数据结构来源：

- :class:`ScriptType`      — 脚本解释器类型（当前落地 Python）；
- :class:`ScriptSpec`      — 一条脚本指令的完整定义（注册表基本单元）；
- :class:`ExecutionContext`— 执行上下文（会话、用户、触发消息）；
- :class:`ExecutionRequest`— 一次执行请求（指令 + 参数 + 上下文）；
- :class:`ScriptResult`    — 执行结果（状态 + 回复文本 + 原始输出）；
- :class:`ScriptConfig`    — 模块级默认配置（超时 / 输出 / 并发等）。

设计约定：

- **二选一来源**：脚本要么是落盘的 ``entry`` 文件（子进程执行），要么是
  内存中的 ``handler``（内置指令），由模型校验强制保证；
- **通信协议**：脚本通过 argv 接收参数、通过 stdin 接收 JSON 上下文、
  通过 stdout 输出回复文本，协议示例见 ``scripts/hello.py``；
- **权限声明**：``permission`` 档位是脚本对自身权限的声明，实际预检由
  :mod:`modules.script.permission` 完成；
- 全部模型为 pydantic v2 ``BaseModel``，与 messaging / transport 一致。
"""

from __future__ import annotations

import sys
from collections.abc import Awaitable, Callable
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .permission import PermissionLevel

#: 默认环境变量白名单（沙箱只透传这些变量，其余一律剥除，防密钥泄漏）。
#: Windows 相关项（SYSTEMROOT / PATHEXT 等）缺失时自动跳过。
DEFAULT_ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH",
    "HOME",
    "USERPROFILE",
    "LANG",
    "LC_ALL",
    "TEMP",
    "TMP",
    "TMPDIR",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "PATHEXT",
    "COMSPEC",
)


class ScriptType(StrEnum):
    """脚本解释器类型。

    当前执行器完整落地 ``PYTHON``；``NODE`` / ``SHELL`` 为 README 规划中的
    扩展位（数据结构先行，执行器后续适配）。
    """

    PYTHON = "python"
    NODE = "node"
    SHELL = "shell"


class ScriptSpec(BaseModel):
    """一条脚本指令的完整定义（注册表的基本单元）。

    ``entry`` 与 ``handler`` 二选一：

    - ``entry``：落盘脚本文件路径，由执行器以子进程方式启动；
    - ``handler``：进程内内置处理器（如 ``/help``），不落盘。

    字段说明：

    - ``name``：指令主名（``/`` 后第一个 token），注册时归一化为小写；
    - ``keywords``：触发同一条命令的关键词（中文别名等），用于无前缀
      的关键词模糊匹配，与 messaging 层 ``CommandSpec`` 对齐；
    - ``fixed_args``：执行时前置到用户参数之前的内置参数；
    - ``timeout_seconds`` / ``max_output_bytes`` / ``max_memory_mb`` /
      ``max_cpu_seconds``：覆盖模块级默认值的单脚本资源限制；
    - ``origin``：来源描述（``builtin`` / 目录路径 / ``manual``），仅供日志
      与排查使用。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(min_length=1, description="指令主名，如 help / weather")
    keywords: tuple[str, ...] = Field(default=(), description="关键词别名（小写比较）")
    help_text: str = Field(default="", description="帮助菜单展示文本")
    description: str = Field(default="", description="脚本功能详细描述")

    script_type: ScriptType = Field(default=ScriptType.PYTHON, description="解释器类型")
    entry: Path | None = Field(default=None, description="落盘脚本路径（二选一）")
    handler: ScriptHandler | None = Field(
        default=None, description="内置处理器（二选一）"
    )

    fixed_args: tuple[str, ...] = Field(default=(), description="前置内置参数")
    timeout_seconds: float | None = Field(
        default=None, gt=0, description="超时覆盖（秒）"
    )
    max_output_bytes: int | None = Field(
        default=None, gt=0, description="输出上限覆盖（字节）"
    )
    max_memory_mb: int | None = Field(
        default=None, gt=0, description="内存上限（MB，仅 POSIX 生效）"
    )
    max_cpu_seconds: int | None = Field(
        default=None, gt=0, description="CPU 时间上限（秒，仅 POSIX 生效）"
    )

    permission: PermissionLevel = Field(
        default=PermissionLevel.STANDARD, description="权限档位声明"
    )
    enabled: bool = Field(default=True, description="是否启用")
    origin: str = Field(default="", description="来源描述（日志用）")

    @model_validator(mode="after")
    def _check_entry_or_handler(self) -> ScriptSpec:
        """保证 entry 与 handler 恰好提供一个。"""
        if (self.entry is None) == (self.handler is None):
            raise ValueError("entry 与 handler 必须二选一（且只能提供一个）")
        # Path("") 会被规范成 Path(".")，同样视为空路径
        if self.entry is not None and str(self.entry).strip() in {"", "."}:
            raise ValueError("entry 不能为空路径")
        return self

    @property
    def is_builtin(self) -> bool:
        """是否为进程内内置指令（无落盘脚本）。"""
        return self.handler is not None


class ScriptStatus(StrEnum):
    """脚本执行状态。"""

    SUCCESS = "success"  #: 正常结束（退出码 0 或 handler 返回）
    FAILED = "failed"  #: 脚本异常结束（非零退出码 / 输出超限）
    TIMEOUT = "timeout"  #: 执行超时，已被强制终止
    DENIED = "denied"  #: 权限预检未通过，未启动
    NOT_FOUND = "not_found"  #: 指令未注册
    ERROR = "error"  #: 系统错误（启动失败、内部异常等）


class ExecutionContext(BaseModel):
    """执行上下文：随指令一起透传给脚本的关键信息。

    脚本通过 stdin 收到该结构的 JSON 序列化（见 ``scripts/hello.py``），
    用于实现按会话 / 用户区分的逻辑（如限流、个性化回复）。
    """

    session_key: str = Field(default="", description="会话键 group:<id> / user:<id>")
    user_id: str = Field(default="", description="触发用户完整 ID（xxx@im.wechat）")
    command: str = Field(default="", description="归一化后的指令名")
    args: list[str] = Field(default_factory=list, description="指令参数")
    message_id: str = Field(default="", description="触发消息 ID")
    is_group: bool = Field(default=False, description="是否群聊触发")
    extra: dict[str, str] = Field(default_factory=dict, description="附加透传字段")


class ExecutionRequest(BaseModel):
    """一次脚本执行请求。"""

    command: str = Field(description="指令名（注册表中的 name）")
    args: list[str] = Field(default_factory=list, description="用户参数")
    context: ExecutionContext = Field(
        default_factory=ExecutionContext, description="执行上下文"
    )


#: 内置脚本处理器：接收 (参数列表, 执行上下文)，返回回复文本。
#: 允许同步或异步实现，由执行器统一 await。
#: 定义在 ExecutionContext 之后：前向引用可被 pydantic 解析。
ScriptHandler = Callable[
    ["list[str]", "ExecutionContext"], str | Awaitable[str]
]


class ScriptResult(BaseModel):
    """脚本执行结果。

    ``reply`` 始终填充（失败时也是面向用户的友好说明），上层（main.py）
    只需把它发给用户；``stdout`` / ``stderr`` 保留原始输出供日志排查。
    """

    #: 状态；构造中间结果时允许缺省（ERROR 是安全的默认值，最终由
    #: 执行器映射覆盖）
    status: ScriptStatus = Field(default=ScriptStatus.ERROR, description="执行状态")
    command: str = ""
    reply: str = Field(default="", description="面向用户的回复文本")
    stdout: str = Field(default="", description="脚本标准输出（已截断）")
    stderr: str = Field(default="", description="脚本标准错误（已截断）")
    exit_code: int | None = Field(default=None, description="子进程退出码")
    duration_ms: int = Field(default=0, description="执行耗时（毫秒）")
    truncated: bool = Field(default=False, description="输出是否触顶截断")
    reason: str = Field(default="", description="失败原因（人类可读）")

    @property
    def ok(self) -> bool:
        """是否执行成功。"""
        return self.status is ScriptStatus.SUCCESS


class ScriptConfig(BaseModel):
    """Script 模块级默认配置。

    README 配置表中的 ``script.timeout``（毫秒）与 ``script.workDir``
    在此对应 ``timeout_seconds`` 与 ``scripts_dir``。
    """

    #: 用户脚本目录（自动扫描加载 ``*.py``，缺失时静默跳过）
    scripts_dir: Path = Field(default=Path("scripts"), description="用户脚本目录")
    #: 沙箱工作根目录（每次执行在其下创建独立临时目录）
    work_root: Path = Field(
        default=Path("Local/script_runs"), description="沙箱工作根目录"
    )
    #: 单脚本执行超时（秒），超时后强制终止整个进程树
    timeout_seconds: float = Field(default=30.0, gt=0, description="默认超时（秒）")
    #: 单脚本 stdout / stderr 各自捕获上限（字节），超出部分丢弃并标记截断
    max_output_bytes: int = Field(
        default=64 * 1024, gt=0, description="输出捕获上限（字节）"
    )
    #: 回复给用户的文本长度上限（微信长文本体验差，超出截断并提示）
    max_reply_length: int = Field(default=2000, gt=0, description="回复长度上限")
    #: 并发执行上限（信号量控制，防止脚本洪水挤占宿主）
    max_concurrent: int = Field(default=4, ge=1, description="并发执行上限")
    #: 目录加载脚本的默认权限档位
    default_permission: PermissionLevel = Field(
        default=PermissionLevel.STANDARD, description="目录脚本默认权限档位"
    )
    #: 执行 Python 脚本的解释器（默认当前解释器）
    python_executable: str = Field(
        default_factory=lambda: sys.executable, description="Python 解释器路径"
    )
    #: 是否以 ``python -I -u`` 隔离非缓冲模式启动（``-I`` 忽略环境变量与
    #: 用户 site 目录，``-u`` 保证输出实时可捕获）
    isolated_python: bool = Field(default=True, description="隔离模式启动 Python")
    #: 是否保留沙箱工作目录（调试用，生产保持 False）
    keep_workdir: bool = Field(default=False, description="保留沙箱工作目录")
    #: 是否对脚本源码做静态违规扫描（权限预检的一部分）
    static_scan: bool = Field(default=True, description="静态源码扫描开关")
    #: 沙箱环境变量白名单
    env_allowlist: tuple[str, ...] = Field(
        default=DEFAULT_ENV_ALLOWLIST, description="环境变量白名单"
    )
