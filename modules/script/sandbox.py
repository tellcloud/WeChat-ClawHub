"""Script 模块沙箱：隔离脚本运行环境并限制资源。

对应 README 目录结构中的 ``modules/script/sandbox.py``。

沙箱提供的能力（进程级 containment）：

- **独立工作目录**：每次执行在 ``work_root`` 下创建全新临时目录作为
  cwd，结束后默认整树删除，脚本写文件不会污染项目目录；
- **环境净化**：只透传白名单环境变量（防 API Key 等泄漏给脚本），
  并强制 UTF-8 / 无缓冲 / 不落字节码；
- **资源限制**：超时强杀（整棵进程树）、stdout / stderr 捕获上限
  （超限即丢弃后续输出并标记截断，防内存耗尽）、POSIX 下可用
  ``setrlimit`` 限制内存与 CPU 时间；
- **进程树击杀**：超时 / 输出异常时终止脚本及其派生的子进程，
  避免孤儿进程（Windows 用 ``taskkill /T /F``，POSIX 用进程组信号）。

已知边界（如实说明）：

- Windows 无 ``setrlimit``，内存 / CPU 上限仅记录不生效；
- 网络与子进程的**禁止**依赖 :mod:`modules.script.permission` 的预检，
  本层不做协议级封锁（需要 OS 级沙箱，见 permission 模块说明）；
- 防的是「误操作与失控脚本」，不是「恶意代码」——恶意脚本应在容器中运行。

用法::

    sandbox = Sandbox(config)
    outcome = await sandbox.run(
        [config.python_executable, "-I", "-u", "hello.py", "世界"],
        limits=ResourceLimits(timeout_seconds=10),
        stdin_data=json.dumps(ctx).encode(),
    )
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable

from pydantic import BaseModel, Field

from .models import ScriptConfig

logger = logging.getLogger(__name__)

#: 单次读取管道的数据块大小
_CHUNK_SIZE = 65536

#: 进程退出后等待读取器收敛的宽限时间（秒）。
#: 正常情况管道立即 EOF；只有被击杀进程的孙进程仍持有管道时才需要等待。
_DRAIN_GRACE_SECONDS = 5.0

#: 工作目录名安全化（保留字母数字与少量符号）
_TAG_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class ResourceLimits(BaseModel):
    """单次执行的资源上限（由 spec 覆盖项与模块默认值合并而来）。"""

    timeout_seconds: float = Field(default=30.0, gt=0, description="超时（秒）")
    max_output_bytes: int = Field(
        default=64 * 1024, gt=0, description="输出捕获上限（字节）"
    )
    max_memory_mb: int | None = Field(
        default=None, gt=0, description="地址空间上限（MB，仅 POSIX 生效）"
    )
    max_cpu_seconds: int | None = Field(
        default=None, gt=0, description="CPU 时间上限（秒，仅 POSIX 生效）"
    )


class SandboxOutcome(BaseModel):
    """沙箱执行结果（底层，不含业务语义）。"""

    exit_code: int | None = Field(default=None, description="退出码（未结束为 None）")
    stdout: str = Field(default="", description="标准输出（UTF-8，已截断）")
    stderr: str = Field(default="", description="标准错误（UTF-8，已截断）")
    duration_ms: int = Field(default=0, description="耗时（毫秒）")
    timed_out: bool = Field(default=False, description="是否超时被杀")
    truncated: bool = Field(default=False, description="输出是否触顶截断")
    error: str = Field(default="", description="启动失败等系统级错误描述")
    workdir: str = Field(default="", description="本次执行的工作目录（调试用）")


async def _pump(stream: asyncio.StreamReader, budget: int) -> tuple[bytes, bool]:
    """读取一个输出管道直到 EOF，最多保留 ``budget`` 字节。

    超出预算后继续读取并丢弃：防止脚本疯狂输出时写满管道缓冲区造成
    双向死锁（脚本阻塞在 write，我们阻塞在 read）。内存占用恒不超过
    ``budget + _CHUNK_SIZE``。
    """
    chunks: list[bytes] = []
    total = 0
    overflow = False
    while True:
        chunk = await stream.read(_CHUNK_SIZE)
        if not chunk:
            break
        if not overflow:
            room = budget - total
            if len(chunk) > room:
                chunks.append(chunk[:room])
                overflow = True
            else:
                chunks.append(chunk)
                total += len(chunk)
        # overflow 之后：读到的数据直接丢弃
    return b"".join(chunks), overflow


class Sandbox:
    """脚本进程沙箱：以子进程方式隔离运行任意 argv。

    :param config: 模块配置（读取 ``work_root`` / ``env_allowlist`` /
        ``keep_workdir`` 等字段）。
    """

    def __init__(self, config: ScriptConfig) -> None:
        self.config = config

    # -- 主入口 ---------------------------------------------------------------

    async def run(
        self,
        argv: list[str],
        *,
        limits: ResourceLimits | None = None,
        stdin_data: bytes | None = None,
        env_extra: dict[str, str] | None = None,
        tag: str = "run",
    ) -> SandboxOutcome:
        """在沙箱中执行一条完整 argv（含解释器），返回执行结果。

        :param argv: 完整命令行，如 ``[python, "-I", "-u", script, *args]``；
        :param limits: 资源上限，默认 :class:`ResourceLimits` 缺省值；
        :param stdin_data: 写入脚本 stdin 的字节（如 JSON 上下文），
            脚本不消费时静默忽略（可能触发 BrokenPipe，已兜底）；
        :param env_extra: 追加到沙箱环境的环境变量（仍受白名单约束的是
            继承部分，这里是显式注入）；
        :param tag: 工作目录名前缀（取指令名，便于排查）。
        """
        limits = limits or ResourceLimits()
        started = time.monotonic()
        workdir = self._make_workdir(tag)
        env = self._build_env(env_extra)
        logger.debug(
            "沙箱执行: argv=%s cwd=%s timeout=%ss",
            argv,
            workdir,
            limits.timeout_seconds,
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=workdir,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=self._preexec_fn(limits),
                **self._platform_kwargs(),
            )
        except (OSError, ValueError) as exc:
            # 解释器不存在、脚本路径非法等启动级失败
            self._cleanup(workdir)
            return SandboxOutcome(
                error=f"脚本启动失败: {exc}",
                duration_ms=self._elapsed_ms(started),
                workdir=workdir if self.config.keep_workdir else "",
            )

        try:
            await self._write_stdin(proc, stdin_data)
            out_task = asyncio.create_task(
                _pump(proc.stdout, limits.max_output_bytes)  # type: ignore[arg-type]
            )
            err_task = asyncio.create_task(
                _pump(proc.stderr, limits.max_output_bytes)  # type: ignore[arg-type]
            )

            timed_out = False
            try:
                exit_code = await asyncio.wait_for(
                    proc.wait(), timeout=limits.timeout_seconds
                )
            except TimeoutError:
                timed_out = True
                logger.warning(
                    "脚本执行超时（%ss），终止进程树: argv=%s",
                    limits.timeout_seconds,
                    argv,
                )
                await self._terminate_tree(proc)
                try:
                    exit_code = await asyncio.wait_for(proc.wait(), timeout=5.0)
                except TimeoutError:
                    exit_code = None

            # 进程已退出（或被杀），管道随即 EOF；孙进程仍持有管道时等待宽限
            try:
                (stdout_bytes, out_overflow), (stderr_bytes, err_overflow) = await (
                    asyncio.wait_for(
                        asyncio.gather(out_task, err_task),
                        timeout=_DRAIN_GRACE_SECONDS,
                    )
                )
            except TimeoutError:
                for task in (out_task, err_task):
                    task.cancel()
                stdout_bytes, stderr_bytes = b"", b""
                out_overflow = err_overflow = True
        finally:
            self._cleanup(workdir)

        return SandboxOutcome(
            exit_code=exit_code,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            duration_ms=self._elapsed_ms(started),
            timed_out=timed_out,
            truncated=out_overflow or err_overflow,
            workdir=workdir if self.config.keep_workdir else "",
        )

    # -- 环境与工作目录 -------------------------------------------------------

    def _make_workdir(self, tag: str) -> str:
        """创建工作目录：``work_root/sbx-<tag>-<随机>/``。"""
        safe_tag = _TAG_SANITIZE_RE.sub("_", tag)[:32] or "run"
        try:
            self.config.work_root.mkdir(parents=True, exist_ok=True)
            return tempfile.mkdtemp(
                prefix=f"sbx-{safe_tag}-", dir=str(self.config.work_root)
            )
        except OSError as exc:
            # work_root 不可写（磁盘满 / 权限）时退回到系统临时目录
            logger.warning("沙箱工作目录创建失败，回退系统临时目录: %s", exc)
            return tempfile.mkdtemp(prefix=f"sbx-{safe_tag}-")

    def _build_env(self, extra: dict[str, str] | None) -> dict[str, str]:
        """构建净化后的环境变量：白名单继承 + 显式注入 + 编码加固。"""
        env: dict[str, str] = {}
        for key in self.config.env_allowlist:
            value = os.environ.get(key)
            if value is not None:
                env[key] = value
        if extra:
            env.update(extra)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return env

    def _cleanup(self, workdir: str) -> None:
        """删除工作目录（``keep_workdir=True`` 时保留供调试）。"""
        if self.config.keep_workdir:
            logger.info("保留沙箱工作目录: %s", workdir)
            return
        shutil.rmtree(workdir, ignore_errors=True)

    # -- 进程管理 -------------------------------------------------------------

    def _platform_kwargs(self) -> dict:
        """平台相关的子进程启动参数。

        POSIX：``start_new_session`` 使脚本成为独立进程组，便于整组击杀；
        Windows：``CREATE_NEW_PROCESS_GROUP`` 隔离 Ctrl+C，配合
        ``taskkill /T`` 实现进程树击杀。
        """
        if sys.platform == "win32":
            return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        return {"start_new_session": True}

    def _preexec_fn(self, limits: ResourceLimits) -> Callable[[], None] | None:
        """POSIX 资源限制（``setrlimit``），无限制项时返回 None。

        注意：``preexec_fn`` 在 fork 后的子进程中运行，官方明确其非线程
        安全；本模块仅设置 rlimit，逻辑保持最小。
        """

        def apply() -> None:
            import resource  # POSIX only：Windows 无此模块（调用方已分流）

            # 本函数只在 POSIX 分支返回（见下方），断言即运行期契约
            assert sys.platform != "win32"
            # resource 的 setrlimit / RLIMIT_* 在 Windows 类型桩中不存在
            # （Unix-only），逐行忽略（POSIX 运行期均可用）
            if limits.max_memory_mb:
                cap = limits.max_memory_mb * 1024 * 1024
                resource.setrlimit(  # type: ignore[attr-defined]
                    resource.RLIMIT_AS, (cap, cap)  # type: ignore[attr-defined]
                )
            if limits.max_cpu_seconds:
                resource.setrlimit(  # type: ignore[attr-defined]
                    resource.RLIMIT_CPU,  # type: ignore[attr-defined]
                    (limits.max_cpu_seconds, limits.max_cpu_seconds),
                )

        if limits.max_memory_mb or limits.max_cpu_seconds:
            if sys.platform == "win32":
                logger.warning(
                    "Windows 不支持 setrlimit，内存/CPU 上限未生效（仅记录）"
                )
                return None
            return apply
        return None

    async def _write_stdin(
        self, proc: asyncio.subprocess.Process, data: bytes | None
    ) -> None:
        """写入 stdin 后关闭；脚本不消费时容忍 BrokenPipe。"""
        if proc.stdin is None:
            return
        try:
            if data:
                proc.stdin.write(data)
                await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            # 脚本提前退出 / 不读 stdin 是常见情况，不影响执行结果
            logger.debug("stdin 写入被忽略（脚本可能未消费）: %s", exc)
        finally:
            try:
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass

    async def _terminate_tree(self, proc: asyncio.subprocess.Process) -> None:
        """终止脚本进程树（含派生的子进程）。"""
        if proc.returncode is not None:
            return
        if sys.platform == "win32":
            try:
                killer = await asyncio.create_subprocess_exec(
                    "taskkill",
                    "/PID",
                    str(proc.pid),
                    "/T",
                    "/F",
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                await killer.wait()
                if killer.returncode == 0:
                    return
            except (OSError, ValueError) as exc:
                logger.debug("taskkill 失败，回退单进程击杀: %s", exc)
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                return
            except (ProcessLookupError, PermissionError, OSError) as exc:
                logger.debug("进程组击杀失败，回退单进程击杀: %s", exc)
        self._kill(proc)

    @staticmethod
    def _kill(proc: asyncio.subprocess.Process) -> None:
        """单进程击杀兜底。"""
        try:
            proc.kill()
        except (ProcessLookupError, OSError) as exc:
            logger.debug("进程击杀失败（可能已退出）: %s", exc)

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        """从 ``started`` 到现在的毫秒数。"""
        return int((time.monotonic() - started) * 1000)
