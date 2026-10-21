"""Script 模块执行器：负责启动并执行 Python 脚本。

对应 README 目录结构中的 ``modules/script/executor.py``。

职责（一条指令的完整生命周期）：

1. **解析**：在 :class:`~modules.script.registry.ScriptRegistry` 中
   精确 → 唯一前缀解析指令，未命中返回 ``NOT_FOUND``；
2. **预检**：启用状态 + :class:`~modules.script.permission.PermissionChecker`
   权限校验，任一不通过返回 ``DENIED``（不启动进程）；
3. **执行**：

   - 内置 ``handler``：进程内直接调用（同步 / 异步均可）；
   - 落盘 ``entry``：经 :class:`~modules.script.sandbox.Sandbox` 以
     ``python -I -u`` 子进程启动，argv 传参、stdin 传 JSON 上下文、
     stdout 传回复；
4. **收敛**：把退出码 / 超时 / 输出超限 / 启动失败统一映射为
   :class:`~modules.script.models.ScriptResult`，``reply`` 始终是面向
   用户的友好文本，原始输出留在 ``stdout`` / ``stderr`` 供日志排查。

并发由信号量控制（``ScriptConfig.max_concurrent``），防止脚本洪水拖垮
宿主；执行过程不向调用方抛异常（内部错误收敛为 ``ERROR`` 结果）。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from pathlib import Path

from pydantic import BaseModel, Field

from .models import (
    ExecutionContext,
    ExecutionRequest,
    ScriptConfig,
    ScriptResult,
    ScriptSpec,
    ScriptStatus,
)
from .permission import PermissionChecker, PermissionPolicy
from .registry import ScriptRegistry
from .sandbox import ResourceLimits, Sandbox, SandboxOutcome

logger = logging.getLogger(__name__)

#: 未找到指令时的回复模板（``{command}`` 为指令名）
NOT_FOUND_REPLY_TEMPLATE = "未找到脚本指令 /{command}，发送 /help 查看可用指令"

#: 失败回复中引用的 stderr 片段长度上限（字符）
_STDERR_SNIPPET_LENGTH = 300


def not_found_reply(command: str) -> str:
    """生成「未找到指令」的标准回复。"""
    return NOT_FOUND_REPLY_TEMPLATE.format(command=command)


class ExecutorOptions(BaseModel):
    """执行器可调参数（均可选，缺省取 :class:`ScriptConfig`）。"""

    #: 单脚本默认超时（秒）
    timeout_seconds: float | None = Field(default=None, gt=0)
    #: 单脚本输出捕获上限（字节）
    max_output_bytes: int | None = Field(default=None, gt=0)


class ScriptExecutor:
    """脚本执行器：把一条执行请求跑成 :class:`ScriptResult`。

    用法::

        executor = ScriptExecutor(registry, config=ScriptConfig())
        result = await executor.execute_command("weather", ["北京"], context)
        if result.ok:
            await sender.send_text(user_id, result.reply)

    :param registry: 脚本注册表；
    :param config: 模块配置（超时 / 输出 / 并发 / 解释器等默认值）；
    :param sandbox: 沙箱实例，缺省按 ``config`` 创建；
    :param checker: 权限预检器，缺省按默认策略创建；
    :param options: 细项覆盖（一般不需要，直接改 ``config`` 即可）。
    """

    def __init__(
        self,
        registry: ScriptRegistry,
        *,
        config: ScriptConfig | None = None,
        sandbox: Sandbox | None = None,
        checker: PermissionChecker | None = None,
        options: ExecutorOptions | None = None,
    ) -> None:
        self.registry = registry
        self.config = config or ScriptConfig()
        self.sandbox = sandbox or Sandbox(self.config)
        self.checker = checker or PermissionChecker(PermissionPolicy())
        self.options = options or ExecutorOptions()
        #: 并发闸门：最多 max_concurrent 个脚本同时执行
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)

    # -- 主入口 ---------------------------------------------------------------

    async def execute(self, request: ExecutionRequest) -> ScriptResult:
        """执行一条请求（不抛异常，错误收敛为 ``ERROR`` 结果）。"""
        command = request.command.strip().lower()
        started = time.monotonic()

        spec = self.registry.resolve_exact(command) or self.registry.resolve_prefix(
            command
        )
        if spec is None:
            logger.info("指令未注册: /%s", command)
            return ScriptResult(
                status=ScriptStatus.NOT_FOUND,
                command=command,
                reply=not_found_reply(command),
                duration_ms=self._elapsed_ms(started),
                reason="注册表中无此指令",
            )

        if not spec.enabled:
            return ScriptResult(
                status=ScriptStatus.DENIED,
                command=spec.name,
                reply=f"指令 /{spec.name} 已停用",
                duration_ms=self._elapsed_ms(started),
                reason="spec.enabled=False",
            )

        decision = self.checker.check(spec)
        if not decision.allowed:
            logger.warning(
                "权限预检未通过: /%s violations=%s", spec.name, decision.violations
            )
            return ScriptResult(
                status=ScriptStatus.DENIED,
                command=spec.name,
                reply=(
                    f"脚本 /{spec.name} 未通过权限校验，"
                    f"已阻止执行：{decision.reason}"
                ),
                duration_ms=self._elapsed_ms(started),
                reason=decision.reason,
            )

        try:
            if spec.is_builtin:
                result = await self._run_handler(spec, request, started)
            else:
                result = await self._run_script(spec, request, started)
        except Exception as exc:  # 兜底：执行器绝不把异常抛给路由层
            logger.exception("指令执行内部异常: /%s", spec.name)
            return ScriptResult(
                status=ScriptStatus.ERROR,
                command=spec.name,
                reply=f"指令 /{spec.name} 执行异常，请稍后重试或联系管理者",
                duration_ms=self._elapsed_ms(started),
                reason=str(exc),
            )

        logger.info(
            "指令执行完成: /%s status=%s duration=%dms truncated=%s",
            spec.name,
            result.status.value,
            result.duration_ms,
            result.truncated,
        )
        return result

    async def execute_command(
        self,
        command: str,
        args: list[str] | None = None,
        context: ExecutionContext | None = None,
    ) -> ScriptResult:
        """便捷入口：直接给指令名与参数。"""
        return await self.execute(
            ExecutionRequest(
                command=command,
                args=list(args or []),
                context=context or ExecutionContext(),
            )
        )

    # -- 两种执行路径 ----------------------------------------------------------

    async def _run_handler(
        self, spec: ScriptSpec, request: ExecutionRequest, started: float
    ) -> ScriptResult:
        """执行内置处理器（进程内调用，同步 / 异步均可）。"""
        assert spec.handler is not None  # is_builtin 保证
        async with self._semaphore:
            outcome = spec.handler(request.args, request.context)
            if inspect.isawaitable(outcome):
                outcome = await outcome
        return ScriptResult(
            status=ScriptStatus.SUCCESS,
            command=spec.name,
            reply=self._format_reply(str(outcome)),
            duration_ms=self._elapsed_ms(started),
        )

    async def _run_script(
        self, spec: ScriptSpec, request: ExecutionRequest, started: float
    ) -> ScriptResult:
        """执行落盘脚本（子进程 + 沙箱）。"""
        assert spec.entry is not None  # 非 builtin 保证
        argv = self._build_argv(spec, request.args)
        limits = self._resolve_limits(spec)
        stdin_data = json.dumps(
            request.context.model_dump(mode="json"), ensure_ascii=False
        ).encode("utf-8")

        async with self._semaphore:
            outcome = await self.sandbox.run(
                argv, limits=limits, stdin_data=stdin_data, tag=spec.name
            )

        base = ScriptResult(
            command=spec.name,
            duration_ms=self._elapsed_ms(started),
            truncated=outcome.truncated,
        )
        return self._map_outcome(spec, outcome, base, limits)

    # -- 内部 -----------------------------------------------------------------

    def _build_argv(self, spec: ScriptSpec, args: list[str]) -> list[str]:
        """构造完整 argv：解释器 + 隔离参数 + 脚本 + 内置参数 + 用户参数。

        脚本路径解析为绝对路径：子进程的 cwd 是沙箱临时目录，相对路径
        会找不到文件。
        """
        assert spec.entry is not None  # 仅落盘脚本走此路径（is_builtin 已分流）
        argv = [self.config.python_executable]
        if self.config.isolated_python:
            # -I：隔离模式（忽略环境变量 / 用户 site 目录，且不带脚本所在
            #     目录进 sys.path）；-u：无缓冲，保证输出实时可捕获；
            #     -X utf8：强制 stdio 使用 UTF-8（-I 隐含 -E 会忽略
            #     PYTHONIOENCODING，Windows 中文区域设置下子进程默认输出
            #     GBK，与沙箱的 UTF-8 解码不匹配）
            argv += ["-I", "-u", "-X", "utf8"]
        argv.append(str(Path(spec.entry).resolve()))
        argv += list(spec.fixed_args)
        argv += list(args)
        return argv

    def _resolve_limits(self, spec: ScriptSpec) -> ResourceLimits:
        """合并单脚本覆盖项与模块默认值，得到本次执行的资源上限。"""
        return ResourceLimits(
            timeout_seconds=spec.timeout_seconds
            or self.options.timeout_seconds
            or self.config.timeout_seconds,
            max_output_bytes=spec.max_output_bytes
            or self.options.max_output_bytes
            or self.config.max_output_bytes,
            max_memory_mb=spec.max_memory_mb,
            max_cpu_seconds=spec.max_cpu_seconds,
        )

    def _map_outcome(
        self,
        spec: ScriptSpec,
        outcome: SandboxOutcome,
        base: ScriptResult,
        limits: ResourceLimits,
    ) -> ScriptResult:
        """把沙箱结果映射为业务结果（状态 + 回复文本）。"""
        if outcome.error:
            return base.model_copy(
                update={
                    "status": ScriptStatus.ERROR,
                    "reply": f"脚本 /{spec.name} 启动失败：{outcome.error}",
                    "reason": outcome.error,
                    "stdout": outcome.stdout,
                    "stderr": outcome.stderr,
                    "exit_code": outcome.exit_code,
                }
            )

        if outcome.timed_out:
            return base.model_copy(
                update={
                    "status": ScriptStatus.TIMEOUT,
                    "reply": (
                        f"脚本 /{spec.name} 执行超时"
                        f"（上限 {limits.timeout_seconds:g} 秒），已强制终止"
                    ),
                    "reason": "timeout",
                    "stdout": outcome.stdout,
                    "stderr": outcome.stderr,
                    "exit_code": outcome.exit_code,
                }
            )

        if outcome.exit_code != 0:
            snippet = self._stderr_snippet(outcome.stderr)
            reply = f"脚本 /{spec.name} 执行失败（退出码 {outcome.exit_code}）"
            if snippet:
                reply += f"：{snippet}"
            return base.model_copy(
                update={
                    "status": ScriptStatus.FAILED,
                    "reply": reply,
                    "reason": snippet or f"exit_code={outcome.exit_code}",
                    "stdout": outcome.stdout,
                    "stderr": outcome.stderr,
                    "exit_code": outcome.exit_code,
                }
            )

        return base.model_copy(
            update={
                "status": ScriptStatus.SUCCESS,
                "reply": self._format_reply(outcome.stdout),
                "stdout": outcome.stdout,
                "stderr": outcome.stderr,
                "exit_code": outcome.exit_code,
            }
        )

    def _format_reply(self, stdout: str) -> str:
        """把脚本 stdout 收敛为回复文本（去首尾空白 + 长度截断）。"""
        text = stdout.strip()
        if not text:
            return "（脚本执行成功，无输出）"
        limit = self.config.max_reply_length
        if len(text) > limit:
            return (
                text[:limit]
                + f"\n…（输出过长已截断，原长度 {len(text)} 字符）"
            )
        return text

    @staticmethod
    def _stderr_snippet(stderr: str) -> str:
        """取 stderr 末尾片段（报错通常出现在最后）。"""
        text = stderr.strip()
        if not text:
            return ""
        if len(text) > _STDERR_SNIPPET_LENGTH:
            return "…" + text[-_STDERR_SNIPPET_LENGTH:]
        return text

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        """从 ``started`` 到现在的毫秒数。"""
        return int((time.monotonic() - started) * 1000)
