"""modules/script 离线单元测试。

覆盖五个子模块 + 门面：models / registry / permission / sandbox /
executor / ScriptService。全部离线执行：子进程沙箱测试使用临时目录中
的真实 Python 脚本（``sys.executable``），不起任何网络请求。

运行方式（任选）::

    python -m unittest discover -s tests/unit -v
    python -m pytest tests/unit/test_script_module.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

# 保证可以导入 src/ 下的 messaging 包与根目录的 modules 包
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    sys.path.insert(0, str(_path))

from modules.script import (
    NOT_FOUND_REPLY_TEMPLATE,
    ExecutionContext,
    PermissionChecker,
    PermissionLevel,
    PermissionPolicy,
    ResourceLimits,
    Sandbox,
    ScriptConfig,
    ScriptExecutor,
    ScriptRegistry,
    ScriptResult,
    ScriptService,
    ScriptSpec,
    ScriptStatus,
    ScriptType,
    not_found_reply,
)

# ---------------------------------------------------------------------------
# 测试工具
# ---------------------------------------------------------------------------

#: 打印单行文本的极简脚本（验证 stdout 捕获）
ECHO_SCRIPT = 'print("echo-output")\n'

#: 把 stdin 原样写回 stdout 的脚本（验证 stdin 协议）
ECHO_STDIN_SCRIPT = "import sys; sys.stdout.write(sys.stdin.read())\n"

#: 打印环境变量键列表的脚本（验证环境净化）
PRINT_ENV_SCRIPT = (
    "import os, json; print(json.dumps(sorted(os.environ.keys())))\n"
)

#: 打印 cwd 并在 cwd 写一个文件的脚本（验证工作目录隔离）
TOUCH_SCRIPT = (
    "import os; print(os.getcwd()); open('marker.txt', 'w').write('x')\n"
)

#: 睡 N 秒的脚本（验证超时强杀）
SLEEP_SCRIPT = "import time; time.sleep({seconds}); print('done')\n"

#: 打印大量数据的脚本（验证输出截断）
FLOOD_SCRIPT = "print('x' * 1_000_000)\n"

#: 写 stderr 并非零退出的脚本（验证失败路径）
FAIL_SCRIPT = (
    "import sys; print('stdout-line', flush=True); "
    "print('stderr-detail', file=sys.stderr); sys.exit(3)\n"
)

#: 无输出的成功脚本（验证空输出回复）
SILENT_SCRIPT = "pass\n"


def write_script(directory: Path, name: str, source: str) -> Path:
    """在临时目录写一个脚本文件，返回路径。"""
    path = directory / name
    path.write_text(source, encoding="utf-8")
    return path


def make_sandbox(work_root: Path, **config_kwargs) -> Sandbox:
    """构造指向临时 work_root 的沙箱。"""
    return Sandbox(ScriptConfig(work_root=work_root, **config_kwargs))


def make_context(**kwargs) -> ExecutionContext:
    """测试用执行上下文（带可区分默认值）。"""
    defaults = dict(
        session_key="user:tester@im.wechat",
        user_id="tester@im.wechat",
        command="echo",
        args=["a"],
        message_id="m-1",
    )
    defaults.update(kwargs)
    return ExecutionContext(**defaults)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


class ScriptSpecModelTest(unittest.TestCase):
    """ScriptSpec 模型校验。"""

    def test_entry_only(self) -> None:
        spec = ScriptSpec(name="weather", entry=Path("scripts/weather.py"))
        self.assertFalse(spec.is_builtin)
        self.assertEqual(spec.script_type, ScriptType.PYTHON)
        self.assertEqual(spec.permission, PermissionLevel.STANDARD)

    def test_handler_only(self) -> None:
        spec = ScriptSpec(name="help", handler=lambda args, ctx: "menu")
        self.assertTrue(spec.is_builtin)

    def test_entry_and_handler_both_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ScriptSpec(name="bad", entry=Path("a.py"), handler=lambda a, c: "")

    def test_neither_entry_nor_handler_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ScriptSpec(name="bad")

    def test_empty_entry_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ScriptSpec(name="bad", entry=Path(""))

    def test_empty_name_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ScriptSpec(name="", entry=Path("a.py"))

    def test_non_positive_timeout_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ScriptSpec(name="x", entry=Path("a.py"), timeout_seconds=0)


class ScriptResultModelTest(unittest.TestCase):
    """ScriptResult 模型。"""

    def test_ok_property(self) -> None:
        ok = ScriptResult(status=ScriptStatus.SUCCESS, reply="hi")
        failed = ScriptResult(status=ScriptStatus.FAILED, reply="no")
        self.assertTrue(ok.ok)
        self.assertFalse(failed.ok)

    def test_default_status_is_error(self) -> None:
        # 中间结果不允许「默认成功」
        self.assertFalse(ScriptResult(command="x").ok)

    def test_not_found_reply_template(self) -> None:
        self.assertEqual(
            not_found_reply("weather"),
            NOT_FOUND_REPLY_TEMPLATE.format(command="weather"),
        )


class ScriptConfigModelTest(unittest.TestCase):
    """ScriptConfig 默认值。"""

    def test_defaults(self) -> None:
        config = ScriptConfig()
        self.assertEqual(config.timeout_seconds, 30.0)
        self.assertEqual(config.max_output_bytes, 64 * 1024)
        self.assertEqual(config.max_reply_length, 2000)
        self.assertEqual(config.max_concurrent, 4)
        self.assertTrue(config.isolated_python)
        self.assertTrue(config.static_scan)
        self.assertFalse(config.keep_workdir)
        self.assertIn("PATH", config.env_allowlist)

    def test_invalid_concurrency_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ScriptConfig(max_concurrent=0)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


class ScriptRegistryTest(unittest.TestCase):
    """注册表：注册 / 解析 / 渲染。"""

    def setUp(self) -> None:
        self.registry = ScriptRegistry()
        self.registry.register(
            ScriptSpec(
                name="Hello",
                entry=Path("scripts/hello.py"),
                keywords=("hello", "问候"),
                help_text="/hello <名字> — 问候",
            )
        )
        self.registry.register(
            ScriptSpec(
                name="weather",
                entry=Path("scripts/weather.py"),
                keywords=("weather", "天气", "查天气"),
                help_text="/weather <城市> — 查询天气",
            )
        )

    def test_register_normalizes_name(self) -> None:
        spec = self.registry.resolve_exact("HELLO")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.name, "hello")

    def test_register_overwrites_same_name(self) -> None:
        self.registry.register(
            ScriptSpec(name="hello", entry=Path("other.py"), help_text="新版")
        )
        self.assertEqual(self.registry.resolve_exact("hello").help_text, "新版")
        self.assertEqual(len(self.registry), 2)

    def test_register_rejects_invalid_names(self) -> None:
        for bad in ("", "  ", "my cmd", "a/b"):
            with self.assertRaises(ValueError, msg=bad):
                self.registry.register(ScriptSpec(name=bad, entry=Path("x.py")))

    def test_unregister(self) -> None:
        self.registry.unregister("weather")
        self.assertIsNone(self.registry.resolve_exact("weather"))
        self.registry.unregister("weather")  # 重复注销静默

    def test_resolve_exact_whitespace_and_case(self) -> None:
        self.assertIsNotNone(self.registry.resolve_exact("  WeAtHeR "))

    def test_resolve_prefix_unique(self) -> None:
        spec = self.registry.resolve_prefix("we")
        self.assertEqual(spec.name, "weather")

    def test_resolve_prefix_ambiguous_returns_none(self) -> None:
        # he / hel 同时是 hello 与 help 的前缀 → 不唯一，不命中
        self.registry.register(ScriptSpec(name="help", entry=Path("h.py")))
        self.assertIsNone(self.registry.resolve_prefix("he"))
        self.assertIsNone(self.registry.resolve_prefix("hel"))
        self.assertEqual(self.registry.resolve_prefix("hell").name, "hello")
        self.assertEqual(self.registry.resolve_prefix("help").name, "help")

    def test_resolve_prefix_empty(self) -> None:
        self.assertIsNone(self.registry.resolve_prefix("  "))

    def test_resolve_keyword_longest_wins(self) -> None:
        # 「查天气」比「天气」更长，更具体者优先
        spec = self.registry.resolve_keyword("查天气 北京")
        self.assertEqual(spec.name, "weather")
        spec = self.registry.resolve_keyword("今天天气怎么样")
        self.assertEqual(spec.name, "weather")

    def test_resolve_keyword_no_match(self) -> None:
        self.assertIsNone(self.registry.resolve_keyword("讲个笑话"))

    def test_commands_sorted(self) -> None:
        names = [spec.name for spec in self.registry.commands()]
        self.assertEqual(names, sorted(names))

    def test_render_help_lists_enabled_only(self) -> None:
        self.registry.register(
            ScriptSpec(name="off", entry=Path("o.py"), help_text="/off — 已停用",
                       enabled=False)
        )
        text = self.registry.render_help(extra_lines=["/role list — 角色"])
        self.assertIn("/hello <名字> — 问候", text)
        self.assertIn("/role list — 角色", text)
        self.assertNotIn("/off", text)
        self.assertTrue(text.endswith("发送 /指令 [参数]，或直接输入关键词即可触发"))

    def test_render_help_fallback_text(self) -> None:
        registry = ScriptRegistry()
        registry.register(ScriptSpec(name="bare", entry=Path("b.py")))
        self.assertIn("/bare — 自定义脚本", registry.render_help())


class ScriptRegistryDirectoryTest(unittest.TestCase):
    """注册表：目录扫描与元信息解析。"""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="sbx-reg-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, True))

    def test_load_directory_with_meta(self) -> None:
        write_script(
            self.tmp,
            "hello.py",
            "# script: name=hi; keywords=你好,hey; help=/hi — 嗨; "
            "permission=full; timeout=5\n"
            "print('hi')\n",
        )
        registry = ScriptRegistry()
        specs = registry.load_directory(self.tmp)
        self.assertEqual([spec.name for spec in specs], ["hi"])
        spec = registry.resolve_exact("hi")
        self.assertEqual(spec.keywords, ("你好", "hey"))
        self.assertEqual(spec.help_text, "/hi — 嗨")
        self.assertEqual(spec.permission, PermissionLevel.FULL)
        self.assertEqual(spec.timeout_seconds, 5.0)
        self.assertTrue(spec.entry.is_absolute())
        self.assertEqual(spec.script_type, ScriptType.PYTHON)

    def test_load_directory_defaults(self) -> None:
        write_script(self.tmp, "bare.py", "print('x')\n")
        registry = ScriptRegistry()
        registry.load_directory(self.tmp)
        spec = registry.resolve_exact("bare")
        self.assertEqual(spec.permission, PermissionLevel.STANDARD)  # 默认档位
        self.assertEqual(spec.keywords, ())
        self.assertIn("/bare", spec.help_text)

    def test_load_directory_default_permission_override(self) -> None:
        write_script(self.tmp, "bare.py", "print('x')\n")
        registry = ScriptRegistry()
        registry.load_directory(self.tmp, PermissionLevel.READONLY)
        self.assertEqual(
            registry.resolve_exact("bare").permission, PermissionLevel.READONLY
        )

    def test_load_directory_skips_underscore_files(self) -> None:
        write_script(self.tmp, "_helper.py", "print('x')\n")
        write_script(self.tmp, "__init__.py", "")
        write_script(self.tmp, "real.py", "print('x')\n")
        registry = ScriptRegistry()
        specs = registry.load_directory(self.tmp)
        self.assertEqual([spec.name for spec in specs], ["real"])

    def test_load_directory_missing_dir_returns_empty(self) -> None:
        registry = ScriptRegistry()
        self.assertEqual(registry.load_directory(self.tmp / "nope"), [])

    def test_load_directory_invalid_permission_falls_back(self) -> None:
        write_script(
            self.tmp,
            "weird.py",
            "# script: permission=god-mode\nprint('x')\n",
        )
        registry = ScriptRegistry()
        registry.load_directory(self.tmp, PermissionLevel.STANDARD)
        self.assertEqual(
            registry.resolve_exact("weird").permission, PermissionLevel.STANDARD
        )

    def test_load_directory_invalid_timeout_falls_back(self) -> None:
        write_script(
            self.tmp,
            "slow.py",
            "# script: timeout=abc\nprint('x')\n",
        )
        registry = ScriptRegistry()
        registry.load_directory(self.tmp)
        self.assertIsNone(registry.resolve_exact("slow").timeout_seconds)

    def test_load_directory_invalid_name_skipped(self) -> None:
        # 元信息声明了带空格的指令名 → 跳过该文件（不中断其他注册）
        write_script(self.tmp, "a.py", "# script: name=my cmd\nprint('x')\n")
        write_script(self.tmp, "b.py", "print('x')\n")
        registry = ScriptRegistry()
        specs = registry.load_directory(self.tmp)
        self.assertEqual([spec.name for spec in specs], ["b"])

    def test_load_directory_disabled_meta(self) -> None:
        write_script(self.tmp, "off.py", "# script: enabled=false\nprint('x')\n")
        registry = ScriptRegistry()
        registry.load_directory(self.tmp)
        self.assertFalse(registry.resolve_exact("off").enabled)


# ---------------------------------------------------------------------------
# permission
# ---------------------------------------------------------------------------


class PermissionLevelTest(unittest.TestCase):
    """档位展开。"""

    def test_readonly(self) -> None:
        perm = PermissionLevel.READONLY
        from modules.script.permission import PermissionSet

        resolved = PermissionSet.from_level(perm)
        self.assertFalse(resolved.allow_network)
        self.assertFalse(resolved.allow_subprocess)
        self.assertFalse(resolved.allow_write)

    def test_standard(self) -> None:
        from modules.script.permission import PermissionSet

        resolved = PermissionSet.from_level(PermissionLevel.STANDARD)
        self.assertTrue(resolved.allow_write)
        self.assertEqual(resolved.write_scope, "sandbox")
        self.assertFalse(resolved.allow_network)

    def test_full(self) -> None:
        from modules.script.permission import PermissionSet

        resolved = PermissionSet.from_level(PermissionLevel.FULL)
        self.assertTrue(resolved.allow_network)
        self.assertTrue(resolved.allow_subprocess)
        self.assertEqual(resolved.write_scope, "anywhere")


class PermissionCheckerTest(unittest.TestCase):
    """权限预检。"""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="sbx-perm-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, True))
        self.checker = PermissionChecker()

    def _spec(self, source: str, **kwargs) -> ScriptSpec:
        entry = write_script(self.tmp, "target.py", source)
        return ScriptSpec(name="target", entry=entry, **kwargs)

    def test_clean_script_passes(self) -> None:
        decision = self.checker.check(
            self._spec("import json\nprint(json.dumps({}))\n")
        )
        self.assertTrue(decision.allowed)

    def test_network_import_denied_at_standard(self) -> None:
        decision = self.checker.check(self._spec("import socket\n"))
        self.assertFalse(decision.allowed)
        self.assertTrue(any("网络" in v for v in decision.violations))

    def test_requests_import_denied_at_standard(self) -> None:
        decision = self.checker.check(self._spec("import requests\n"))
        self.assertFalse(decision.allowed)

    def test_subprocess_denied_at_standard(self) -> None:
        decision = self.checker.check(self._spec("import subprocess\n"))
        self.assertFalse(decision.allowed)
        self.assertTrue(any("子进程" in v for v in decision.violations))

    def test_ctypes_denied_at_standard(self) -> None:
        decision = self.checker.check(self._spec("import ctypes\n"))
        self.assertFalse(decision.allowed)

    def test_full_level_allows_network(self) -> None:
        checker = PermissionChecker(PermissionPolicy(max_level=PermissionLevel.FULL))
        decision = checker.check(
            self._spec("import socket\n", permission=PermissionLevel.FULL)
        )
        self.assertTrue(decision.allowed)

    def test_level_above_policy_denied(self) -> None:
        decision = self.checker.check(
            self._spec("print('x')\n", permission=PermissionLevel.FULL)
        )
        self.assertFalse(decision.allowed)
        self.assertTrue(any("策略上限" in v for v in decision.violations))

    def test_builtin_handler_skips_check(self) -> None:
        spec = ScriptSpec(
            name="help", handler=lambda a, c: "", permission=PermissionLevel.FULL
        )
        decision = self.checker.check(spec)
        self.assertTrue(decision.allowed)
        self.assertIn("内置", decision.reason)

    def test_missing_entry_denied(self) -> None:
        spec = ScriptSpec(name="ghost", entry=self.tmp / "ghost.py")
        decision = self.checker.check(spec)
        self.assertFalse(decision.allowed)
        self.assertTrue(any("不存在" in v for v in decision.violations))

    def test_static_scan_disabled(self) -> None:
        checker = PermissionChecker(PermissionPolicy(static_scan=False))
        decision = checker.check(self._spec("import socket\n"))
        self.assertTrue(decision.allowed)


# ---------------------------------------------------------------------------
# sandbox
# ---------------------------------------------------------------------------


class SandboxTest(unittest.IsolatedAsyncioTestCase):
    """沙箱：隔离、stdin、环境净化、超时、截断、清理。"""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="sbx-test-"))
        self.work_root = self.tmp / "runs"
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, True))

    def _sandbox(self, **kwargs) -> Sandbox:
        return make_sandbox(self.work_root, **kwargs)

    def _argv(self, script: Path) -> list[str]:
        return [sys.executable, "-I", "-u", "-X", "utf8", str(script)]

    async def test_echo_stdout(self) -> None:
        script = write_script(self.tmp, "echo.py", ECHO_SCRIPT)
        outcome = await self._sandbox().run(self._argv(script))
        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(outcome.stdout.strip(), "echo-output")
        self.assertEqual(outcome.error, "")
        self.assertFalse(outcome.timed_out)

    async def test_stdin_json_reaches_script(self) -> None:
        script = write_script(self.tmp, "echo_stdin.py", ECHO_STDIN_SCRIPT)
        payload = '{"command": "hello", "args": ["世界"]}'
        outcome = await self._sandbox().run(
            self._argv(script), stdin_data=payload.encode("utf-8")
        )
        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(outcome.stdout, payload)

    async def test_script_not_reading_stdin_is_tolerated(self) -> None:
        # 不读 stdin 的脚本 + 有 stdin 数据 →  BrokenPipe 被兜底，正常结束
        script = write_script(self.tmp, "ignore_stdin.py", ECHO_SCRIPT)
        outcome = await self._sandbox().run(
            self._argv(script), stdin_data=b'{"x": 1}'
        )
        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(outcome.stdout.strip(), "echo-output")

    async def test_env_scrubbing(self) -> None:
        script = write_script(self.tmp, "env.py", PRINT_ENV_SCRIPT)
        os.environ["SBX_SECRET_LEAK_TEST"] = "leak-me"
        self.addCleanup(lambda: os.environ.pop("SBX_SECRET_LEAK_TEST", None))
        outcome = await self._sandbox().run(self._argv(script))
        import json

        keys = set(json.loads(outcome.stdout))
        self.assertNotIn("SBX_SECRET_LEAK_TEST", keys)  # 白名单外变量被剥除
        self.assertIn("PYTHONIOENCODING", keys)  # 编码加固始终注入
        self.assertIn("PYTHONDONTWRITEBYTECODE", keys)
        if "PATH" in os.environ:
            self.assertIn("PATH", keys)

    async def test_workdir_isolation_and_cleanup(self) -> None:
        script = write_script(self.tmp, "touch.py", TOUCH_SCRIPT)
        outcome = await self._sandbox().run(self._argv(script))
        cwd = outcome.stdout.strip()
        self.assertIn(str(self.work_root), cwd)  # cwd 是沙箱临时目录
        self.assertNotEqual(Path(cwd), Path.cwd())
        # 脚本写的文件落在沙箱目录，不污染项目根
        self.assertFalse((Path.cwd() / "marker.txt").exists())
        # 执行结束后工作目录被删除
        self.assertEqual(list(self.work_root.glob("sbx-*")), [])

    async def test_keep_workdir(self) -> None:
        script = write_script(self.tmp, "keep.py", ECHO_SCRIPT)
        outcome = await self._sandbox(keep_workdir=True).run(
            self._argv(script), tag="keep"
        )
        self.assertTrue(Path(outcome.workdir).is_dir())

    async def test_timeout_kills_process(self) -> None:
        script = write_script(self.tmp, "sleep.py", SLEEP_SCRIPT.format(seconds=30))
        outcome = await self._sandbox().run(
            self._argv(script), limits=ResourceLimits(timeout_seconds=0.5)
        )
        self.assertTrue(outcome.timed_out)
        self.assertNotEqual(outcome.exit_code, 0)
        self.assertLess(outcome.duration_ms, 10_000)

    async def test_output_truncation(self) -> None:
        script = write_script(self.tmp, "flood.py", FLOOD_SCRIPT)
        outcome = await self._sandbox().run(
            self._argv(script), limits=ResourceLimits(max_output_bytes=4096)
        )
        self.assertTrue(outcome.truncated)
        self.assertLessEqual(len(outcome.stdout), 4096)
        self.assertEqual(outcome.exit_code, 0)

    async def test_launch_failure_returns_error(self) -> None:
        outcome = await self._sandbox().run(["sbx-no-such-interpreter-xyz", "a.py"])
        self.assertTrue(outcome.error)
        self.assertIsNone(outcome.exit_code)
        self.assertEqual(list(self.work_root.glob("sbx-*")), [])  # 无残留

    async def test_stderr_captured(self) -> None:
        script = write_script(self.tmp, "fail.py", FAIL_SCRIPT)
        outcome = await self._sandbox().run(self._argv(script))
        self.assertEqual(outcome.exit_code, 3)
        self.assertIn("stdout-line", outcome.stdout)
        self.assertIn("stderr-detail", outcome.stderr)

    @unittest.skipUnless(sys.platform != "win32", "setrlimit 仅 POSIX 生效")
    async def test_cpu_limit_posix(self) -> None:
        script = write_script(self.tmp, "spin.py", "while True:\n    pass\n")
        outcome = await self._sandbox().run(
            self._argv(script),
            limits=ResourceLimits(
                timeout_seconds=15, max_cpu_seconds=1
            ),
        )
        self.assertFalse(outcome.timed_out)
        self.assertIsNotNone(outcome.exit_code)
        self.assertNotEqual(outcome.exit_code, 0)


# ---------------------------------------------------------------------------
# executor
# ---------------------------------------------------------------------------


class ScriptExecutorTest(unittest.IsolatedAsyncioTestCase):
    """执行器：解析 / 预检 / 执行 / 收敛。"""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="sbx-exec-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, True))
        self.registry = ScriptRegistry()
        self.config = ScriptConfig(work_root=self.tmp / "runs")
        self.executor = ScriptExecutor(self.registry, config=self.config)

    def _register_script(self, name: str, source: str, **kwargs) -> ScriptSpec:
        entry = write_script(self.tmp, f"{name}.py", source)
        return self.registry.register(ScriptSpec(name=name, entry=entry, **kwargs))

    async def test_not_found(self) -> None:
        result = await self.executor.execute_command("ghost", ["x"])
        self.assertEqual(result.status, ScriptStatus.NOT_FOUND)
        self.assertEqual(result.reply, not_found_reply("ghost"))

    async def test_prefix_resolution(self) -> None:
        self._register_script("hello", ECHO_SCRIPT)
        result = await self.executor.execute_command("hel", [])
        self.assertEqual(result.status, ScriptStatus.SUCCESS)
        self.assertEqual(result.command, "hello")

    async def test_disabled_spec_denied(self) -> None:
        self._register_script("off", ECHO_SCRIPT, enabled=False)
        result = await self.executor.execute_command("off", [])
        self.assertEqual(result.status, ScriptStatus.DENIED)
        self.assertIn("已停用", result.reply)

    async def test_permission_denied_never_launches(self) -> None:
        self._register_script("net", "import socket\n", permission=PermissionLevel.FULL)
        result = await self.executor.execute_command("net", [])
        self.assertEqual(result.status, ScriptStatus.DENIED)
        self.assertIn("权限", result.reply)
        # 预检拒绝 → 没有进程被启动（工作目录无残留）
        self.assertEqual(list((self.tmp / "runs").glob("sbx-*")), [])

    async def test_sync_handler(self) -> None:
        self.registry.register_handler(
            "echo",
            lambda args, ctx: f"echo:{','.join(args)}",
            help_text="/echo — 复读",
        )
        result = await self.executor.execute_command("echo", ["a", "b"])
        self.assertEqual(result.status, ScriptStatus.SUCCESS)
        self.assertEqual(result.reply, "echo:a,b")

    async def test_async_handler(self) -> None:
        async def slow_handler(args: list[str], ctx: ExecutionContext) -> str:
            await asyncio.sleep(0.01)
            return "async-ok"

        self.registry.register_handler("slow", slow_handler)
        result = await self.executor.execute_command("slow", [])
        self.assertEqual(result.status, ScriptStatus.SUCCESS)
        self.assertEqual(result.reply, "async-ok")

    async def test_handler_exception_becomes_error_result(self) -> None:
        def boom(args: list[str], ctx: ExecutionContext) -> str:
            raise RuntimeError("handler-detail")

        self.registry.register_handler("boom", boom)
        result = await self.executor.execute_command("boom", [])
        self.assertEqual(result.status, ScriptStatus.ERROR)
        self.assertIn("执行异常", result.reply)
        self.assertEqual(result.reason, "handler-detail")

    async def test_script_success(self) -> None:
        self._register_script("hello", 'print("Hello!")\n')
        result = await self.executor.execute_command("hello", [])
        self.assertEqual(result.status, ScriptStatus.SUCCESS)
        self.assertEqual(result.reply, "Hello!")
        self.assertEqual(result.exit_code, 0)

    async def test_script_silent_output(self) -> None:
        self._register_script("silent", SILENT_SCRIPT)
        result = await self.executor.execute_command("silent", [])
        self.assertEqual(result.status, ScriptStatus.SUCCESS)
        self.assertEqual(result.reply, "（脚本执行成功，无输出）")

    async def test_script_failure_with_stderr_snippet(self) -> None:
        self._register_script("fail", FAIL_SCRIPT)
        result = await self.executor.execute_command("fail", [])
        self.assertEqual(result.status, ScriptStatus.FAILED)
        self.assertEqual(result.exit_code, 3)
        self.assertIn("退出码 3", result.reply)
        self.assertIn("stderr-detail", result.reply)

    async def test_script_timeout(self) -> None:
        self._register_script("sleep", SLEEP_SCRIPT.format(seconds=30))
        result = await self.executor.execute_command("sleep", [])
        self.assertEqual(result.status, ScriptStatus.TIMEOUT)

    async def test_stdin_context_reaches_script(self) -> None:
        self._register_script("ctx", ECHO_STDIN_SCRIPT)
        ctx = make_context(session_key="user:u1@im.wechat", command="ctx", args=["x"])
        result = await self.executor.execute_command("ctx", ["x"], ctx)
        self.assertEqual(result.status, ScriptStatus.SUCCESS)
        self.assertIn("user:u1@im.wechat", result.reply)

    async def test_fixed_args_prepended(self) -> None:
        # 脚本把 argv 原样打印：内置参数应前置在用户参数之前
        source = "import sys; print(' '.join(sys.argv[1:]))\n"
        self._register_script("fixed", source, fixed_args=("--mode", "fast"))
        result = await self.executor.execute_command("fixed", ["user-arg"])
        self.assertEqual(result.reply, "--mode fast user-arg")

    async def test_reply_truncated_at_max_length(self) -> None:
        source = "print('y' * 5000)\n"
        self._register_script("long", source)
        executor = ScriptExecutor(
            self.registry,
            config=ScriptConfig(work_root=self.tmp / "runs", max_reply_length=100),
        )
        result = await executor.execute_command("long", [])
        self.assertEqual(result.status, ScriptStatus.SUCCESS)
        self.assertIn("输出过长已截断", result.reply)
        self.assertLessEqual(len(result.reply), 100 + 64)

    async def test_spec_timeout_override(self) -> None:
        self._register_script(
            "quick", SLEEP_SCRIPT.format(seconds=30), timeout_seconds=0.5
        )
        result = await self.executor.execute_command("quick", [])
        self.assertEqual(result.status, ScriptStatus.TIMEOUT)

    async def test_concurrency_limit(self) -> None:
        # 并发上限 1：两个慢任务必须串行（总耗时 ≈ 2 倍单任务）
        self._register_script("slow", SLEEP_SCRIPT.format(seconds=0.4))
        executor = ScriptExecutor(
            self.registry,
            config=ScriptConfig(work_root=self.tmp / "runs", max_concurrent=1),
        )
        loop = asyncio.get_event_loop()
        started = loop.time()
        results = await asyncio.gather(
            executor.execute_command("slow", []),
            executor.execute_command("slow", []),
        )
        elapsed = loop.time() - started
        self.assertTrue(all(r.ok for r in results))
        self.assertGreaterEqual(elapsed, 0.8)


# ---------------------------------------------------------------------------
# ScriptService（门面）
# ---------------------------------------------------------------------------


class ScriptServiceTest(unittest.IsolatedAsyncioTestCase):
    """门面：目录加载 / handle 分流 / 路由同步 / 帮助菜单。"""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="sbx-svc-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, True))
        self.scripts_dir = self.tmp / "scripts"

    def _service(self, **kwargs) -> ScriptService:
        return ScriptService.from_directory(
            self.scripts_dir,
            config=ScriptConfig(
                scripts_dir=self.scripts_dir, work_root=self.tmp / "runs"
            ),
            **kwargs,
        )

    async def test_from_directory_loads_scripts(self) -> None:
        self.scripts_dir.mkdir(parents=True)
        write_script(self.scripts_dir, "greet.py", 'print("hi")\n')
        service = self._service()
        self.assertIsNotNone(service.registry.resolve_exact("greet"))
        # 内置 help 始终存在
        self.assertIsNotNone(service.registry.resolve_exact("help"))

    async def test_handle_registered_script(self) -> None:
        self.scripts_dir.mkdir(parents=True)
        write_script(self.scripts_dir, "greet.py", 'print("hi there")\n')
        service = self._service()
        reply = await service.handle("greet", [])
        self.assertEqual(reply, "hi there")

    async def test_handle_builtin_help(self) -> None:
        service = self._service()
        reply = await service.handle("help", [])
        self.assertIn("可用指令", reply)
        self.assertIn("/help", reply)

    async def test_handle_unimplemented_command(self) -> None:
        service = self._service(
            unimplemented={
                "role": "暂未完成AI模块的开发工作",
                "agent": "暂未完成Agent模块的开发工作",
            }
        )
        self.assertEqual(
            await service.handle("role", ["set", "程序员"]),
            "暂未完成AI模块的开发工作",
        )
        self.assertEqual(
            await service.handle("AGENT", ["list"]),  # 大小写不敏感
            "暂未完成Agent模块的开发工作",
        )

    async def test_handle_unknown_command(self) -> None:
        service = self._service()
        reply = await service.handle("nosuch", ["x"])
        self.assertEqual(reply, not_found_reply("nosuch"))

    async def test_handle_empty_command_returns_help(self) -> None:
        service = self._service()
        self.assertIn("可用指令", await service.handle("", []))

    async def test_handle_never_raises_on_bad_input(self) -> None:
        service = self._service()
        # 乱码 / 超长 / None 参数都不应抛异常
        self.assertIsInstance(await service.handle("\x00\xff", []), str)
        self.assertIsInstance(await service.handle("x" * 500, None), str)

    async def test_register_into_messaging_registry(self) -> None:
        from messaging.router import CommandRegistry, RouteTarget

        self.scripts_dir.mkdir(parents=True)
        write_script(
            self.scripts_dir,
            "greet.py",
            '# script: keywords=打招呼; help=/greet — 打招呼\nprint("hi")\n',
        )
        service = self._service()
        messaging_registry = CommandRegistry()
        synced = service.register_into(messaging_registry)

        names = {spec.name for spec in synced}
        self.assertIn("greet", names)
        self.assertIn("help", names)
        # 同步后的指令可被路由层三级匹配认识
        spec = messaging_registry.resolve_exact("greet")
        self.assertEqual(spec.target, RouteTarget.SCRIPT)
        self.assertEqual(spec.keywords, ("打招呼",))
        self.assertIsNotNone(messaging_registry.resolve_keyword("来来 打招呼"))

    async def test_register_into_skips_disabled(self) -> None:
        from messaging.router import CommandRegistry

        self.scripts_dir.mkdir(parents=True)
        write_script(
            self.scripts_dir, "off.py", "# script: enabled=false\nprint('x')\n"
        )
        service = self._service()
        messaging_registry = CommandRegistry()
        service.register_into(messaging_registry)
        self.assertIsNone(messaging_registry.resolve_exact("off"))

    async def test_extra_help_lines_in_menu(self) -> None:
        service = self._service(extra_help_lines=["/role list — 列出角色"])
        menu = service.render_help()
        self.assertIn("/role list — 列出角色", menu)

    async def test_load_scripts_rescan(self) -> None:
        service = self._service()
        self.assertIsNone(service.registry.resolve_exact("late"))
        self.scripts_dir.mkdir(parents=True)
        write_script(self.scripts_dir, "late.py", 'print("late")\n')
        service.load_scripts()
        self.assertIsNotNone(service.registry.resolve_exact("late"))

    async def test_end_to_end_keyword_flow(self) -> None:
        """模拟完整链路：目录脚本 → 路由匹配 → 执行回复。"""
        from messaging.models import ContentKind, Message
        from messaging.router import CommandRegistry, MessageRouter

        self.scripts_dir.mkdir(parents=True)
        write_script(
            self.scripts_dir,
            "greet.py",
            '# script: keywords=打招呼\nprint("你好呀！")\n',
        )
        service = self._service()
        router = MessageRouter(CommandRegistry())
        service.register_into(router.registry)

        message = Message(
            message_id="m1",
            sender_id="u@im.wechat",
            sender_key="u",
            session_key="user:u@im.wechat",
            kind=ContentKind.TEXT,
            text="跟你打招呼",
        )
        result = router.route(message)
        self.assertEqual(result.command, "greet")
        reply = await service.handle(result.command, result.args)
        self.assertEqual(reply, "你好呀！")
