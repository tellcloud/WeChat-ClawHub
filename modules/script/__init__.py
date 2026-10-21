"""Script 模块：脚本执行能力（指令注册 + 沙箱执行 + 权限控制）。

对应 README 目录结构中的 ``modules/script/``，公开接口::

    from modules.script import ScriptService, ScriptConfig

    service = ScriptService.from_directory("scripts")          # 自动扫描脚本
    service.register_into(router.registry)                     # 同步到消息路由
    reply = await service.handle("weather", ["北京"], context)  # 执行指令

模块划分：

- :mod:`~modules.script.models`     — 数据模型（脚本定义 / 执行参数 / 结果）；
- :mod:`~modules.script.registry`   — 脚本注册表（注册 / 解析 / 目录扫描）；
- :mod:`~modules.script.permission` — 权限（档位 / 策略 / 静态预检）；
- :mod:`~modules.script.sandbox`    — 沙箱（隔离进程 / 限制资源）；
- :mod:`~modules.script.executor`   — 执行器（启动脚本并收敛结果）；
- :class:`ScriptService`（本文件）   — 门面：组合上述组件，对接 main.py。

设计要点：

- 脚本通信协议：argv 传参、stdin 收 JSON 上下文、stdout 出回复文本，
  示例见 ``scripts/hello.py``；
- 三级匹配（精确 → 唯一前缀 → 关键词）与 messaging 路由层保持一致；
- 执行过程不抛异常：任何失败都收敛为带 ``reply`` 的
  :class:`~modules.script.models.ScriptResult`。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from .executor import (
    NOT_FOUND_REPLY_TEMPLATE,
    ExecutorOptions,
    ScriptExecutor,
    not_found_reply,
)
from .models import (
    ExecutionContext,
    ExecutionRequest,
    ScriptConfig,
    ScriptHandler,
    ScriptResult,
    ScriptSpec,
    ScriptStatus,
    ScriptType,
)
from .permission import (
    PermissionChecker,
    PermissionDecision,
    PermissionLevel,
    PermissionPolicy,
    PermissionSet,
)
from .registry import ScriptRegistry, default_script_registry
from .sandbox import ResourceLimits, Sandbox, SandboxOutcome

if TYPE_CHECKING:  # 仅类型标注依赖，运行期惰性 import
    from messaging.router import CommandRegistry, CommandSpec

logger = logging.getLogger(__name__)

#: 内置 help 指令的帮助文本
HELP_HELP_TEXT = "/help — 显示所有可用指令"


class ScriptService:
    """Script 模块门面：main.py 只需持有这一个对象。

    组合注册表 + 执行器 + 权限 + 沙箱，并提供两个对接点：

    - :meth:`register_into`：把脚本指令同步成 messaging 命令，让路由层
      的三级匹配认识它们；
    - :meth:`handle`：执行一条路由到 SCRIPT 的指令，返回面向用户的
      回复文本（不抛异常）。

    :param registry: 脚本注册表，缺省空表（可再
        :meth:`ScriptRegistry.load_directory` 或直接 ``register``）；
    :param config: 模块配置；
    :param sandbox: 沙箱实例（缺省按 ``config`` 创建）；
    :param checker: 权限预检器（缺省按默认策略创建）；
    :param unimplemented: 已知但由尚未就绪的模块负责的指令 → 提示文案
        （如 ``{"role": "暂未完成AI模块的开发工作"}``）；
    :param extra_help_lines: 帮助菜单补充行（如 role / agent 命令）。
    """

    def __init__(
        self,
        registry: ScriptRegistry | None = None,
        *,
        config: ScriptConfig | None = None,
        sandbox: Sandbox | None = None,
        checker: PermissionChecker | None = None,
        unimplemented: Mapping[str, str] | None = None,
        extra_help_lines: Sequence[str] = (),
    ) -> None:
        self.config = config or ScriptConfig()
        self.registry = registry if registry is not None else default_script_registry()
        self.executor = ScriptExecutor(
            self.registry, config=self.config, sandbox=sandbox, checker=checker
        )
        #: 已注册到尚未就绪模块的指令 → 提示文案（小写键）
        self.unimplemented: dict[str, str] = {
            name.strip().lower(): text for name, text in (unimplemented or {}).items()
        }
        #: 帮助菜单补充行（mutable，供外部在 register_into 后补充）
        self.extra_help_lines: list[str] = list(extra_help_lines)
        #: register_into 记录的 messaging 注册表（load_scripts 时重新同步）
        self._messaging_registry: CommandRegistry | None = None
        self._ensure_builtin_help()

    # -- 构建 ---------------------------------------------------------------

    @classmethod
    def from_directory(
        cls,
        scripts_dir: Path | str,
        *,
        config: ScriptConfig | None = None,
        sandbox: Sandbox | None = None,
        checker: PermissionChecker | None = None,
        unimplemented: Mapping[str, str] | None = None,
        extra_help_lines: Sequence[str] = (),
    ) -> ScriptService:
        """从脚本目录构建服务：自动扫描 ``*.py`` 注册为指令。

        目录不存在时静默跳过（首次运行没有 ``scripts/`` 属正常）。
        """
        cfg = config or ScriptConfig(scripts_dir=Path(scripts_dir))
        service = cls(
            config=cfg,
            sandbox=sandbox,
            checker=checker,
            unimplemented=unimplemented,
            extra_help_lines=extra_help_lines,
        )
        service.load_scripts()
        return service

    def load_scripts(self) -> list[ScriptSpec]:
        """（重新）扫描脚本目录并注册；返回本次注册的 spec 列表。

        若此前调用过 :meth:`register_into`，会顺势把新指令再同步一次。
        """
        specs = self.registry.load_directory(
            self.config.scripts_dir, self.config.default_permission
        )
        if specs and self._messaging_registry is not None:
            self.register_into(self._messaging_registry)
        return specs

    # -- 对接点 ---------------------------------------------------------------

    def register_into(self, messaging_registry: CommandRegistry) -> list[CommandSpec]:
        """把脚本指令同步成 messaging 命令（让路由层认识它们）。

        只同步启用中的指令；messaging 注册表中同名命令会被覆盖
        （如内置 ``help`` 的帮助文本以 script 模块为准）。

        :param messaging_registry:
            :class:`~messaging.router.CommandRegistry` 实例；
        :returns: 同步过去的 messaging ``CommandSpec`` 列表。
        """
        from messaging.router import CommandSpec, RouteTarget  # 惰性依赖

        self._messaging_registry = messaging_registry
        synced: list[CommandSpec] = []
        for spec in self.registry.commands():
            if not spec.enabled:
                continue
            synced.append(
                messaging_registry.register(
                    CommandSpec(
                        name=spec.name,
                        keywords=spec.keywords,
                        help_text=spec.help_text or f"/{spec.name} — 自定义脚本",
                        target=RouteTarget.SCRIPT,
                    )
                )
            )
        return synced

    async def handle(
        self,
        command: str,
        args: Sequence[str] | None = None,
        context: ExecutionContext | None = None,
    ) -> str:
        """执行一条路由到 SCRIPT 的指令，返回回复文本。

        分流逻辑：已注册脚本 → 执行器；已知但未就绪的模块指令 →
        提示文案；其余 → 未找到 + 帮助菜单提示。不抛异常。
        """
        name = (command or "").strip().lower()
        if not name:
            return self.render_help()

        spec = self.registry.resolve_exact(name) or self.registry.resolve_prefix(name)
        if spec is not None:
            result = await self.executor.execute(
                ExecutionRequest(
                    command=name,
                    args=list(args or []),
                    context=context or ExecutionContext(),
                )
            )
            return result.reply

        if name in self.unimplemented:
            logger.info("指令 /%s 由未就绪模块负责，返回占位提示", name)
            return self.unimplemented[name]

        return not_found_reply(name)

    def render_help(self) -> str:
        """渲染完整帮助菜单（脚本指令 + 补充行）。"""
        return self.registry.render_help(self.extra_help_lines)

    # -- 内部 ----------------------------------------------------------------

    def _ensure_builtin_help(self) -> None:
        """确保内置 ``/help`` 存在（由本服务渲染，可合并补充行）。"""

        def help_handler(args: list[str], ctx: ExecutionContext) -> str:
            return self.render_help()

        self.registry.register_handler(
            "help",
            help_handler,
            keywords=("帮助", "菜单"),
            help_text=HELP_HELP_TEXT,
            description="显示所有可用指令",
            origin="builtin",
        )


__all__ = [
    # 门面
    "ScriptService",
    # 注册表
    "ScriptRegistry",
    "default_script_registry",
    # 执行器
    "ScriptExecutor",
    "ExecutorOptions",
    "not_found_reply",
    "NOT_FOUND_REPLY_TEMPLATE",
    # 沙箱
    "Sandbox",
    "SandboxOutcome",
    "ResourceLimits",
    # 权限
    "PermissionChecker",
    "PermissionPolicy",
    "PermissionSet",
    "PermissionLevel",
    "PermissionDecision",
    # 模型
    "ScriptConfig",
    "ScriptSpec",
    "ScriptType",
    "ScriptStatus",
    "ScriptResult",
    "ScriptHandler",
    "ExecutionContext",
    "ExecutionRequest",
    # 其他
    "HELP_HELP_TEXT",
]
