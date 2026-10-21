"""Script 模块注册表：注册和管理可用脚本命令。

对应 README 目录结构中的 ``modules/script/registry.py``。

职责：

- 维护 :class:`~modules.script.models.ScriptSpec` 集合（新增 / 覆盖 /
  注销 / 查询），覆盖时静默替换（热更新语义）；
- 三级解析：精确 → 唯一前缀 → 最长关键词，与 messaging 层
  :class:`~messaging.router.CommandRegistry` 行为对齐；
- **目录扫描**：``load_directory`` 把 ``scripts/*.py`` 自动注册为指令，
  脚本文件头部的 ``# script:`` 元信息注释可声明名称 / 关键词 / 帮助文本 /
  权限档位 / 超时（零代码侵入，见 ``scripts/hello.py`` 示例）；
- 渲染 ``/help`` 帮助菜单（按指令名排序，仅列启用中的指令）。

与 messaging 层的关系：本注册表是 script 模块的权威来源；
:class:`~modules.script.ScriptService.register_into` 会把这里的指令
同步成 messaging 的 ``CommandSpec``，让路由层认识这些命令。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from .models import ScriptSpec, ScriptType
from .permission import PermissionLevel

logger = logging.getLogger(__name__)

#: 目录脚本元信息注释：``# script: name=weather; keywords=天气,查天气; help=...``
_META_LINE_RE = re.compile(r"^#\s*script:\s*(?P<meta>.+?)\s*$")

#: 指令名合法性：不允许空白与路径分隔符（否则永远无法被 ``/指令`` 命中）
_NAME_INVALID_RE = re.compile(r"[\s/\\]")

#: 目录加载时允许声明的元信息键
_META_KEYS = frozenset(
    {"name", "keywords", "help", "description", "permission", "timeout", "enabled"}
)


def default_script_registry() -> ScriptRegistry:
    """内置脚本注册表（当前仅 ``/help`` 由 :class:`ScriptService` 注入）。

    返回空注册表；内置指令（help）由服务门面在初始化时注册，
    以便帮助菜单能合并 messaging 层补充的命令行。
    """
    return ScriptRegistry()


class ScriptRegistry:
    """脚本指令注册表。

    用法::

        registry = ScriptRegistry()
        registry.register(ScriptSpec(name="weather", entry=Path("scripts/weather.py"),
                                     keywords=("天气", "查天气"),
                                     help_text="/weather <城市> — 查询天气"))
        registry.load_directory(Path("scripts"))     # 目录扫描（元信息驱动）

        spec = registry.resolve_exact("weather")     # 精确
        spec = registry.resolve_prefix("we")         # 唯一前缀
        spec = registry.resolve_keyword("查天气 北京")  # 最长关键词
    """

    def __init__(self, specs: Iterable[ScriptSpec] = ()) -> None:
        self._specs: dict[str, ScriptSpec] = {}
        for spec in specs:
            self.register(spec)

    # -- 注册 ----------------------------------------------------------------

    def register(self, spec: ScriptSpec) -> ScriptSpec:
        """注册一条脚本指令（同名覆盖，用于热更新）。

        :returns: 归一化后的 spec（指令名已转小写并去空白）。
        :raises ValueError: 指令名为空或包含空白 / 路径分隔符。
        """
        name = spec.name.strip().lower()
        if not name:
            raise ValueError("指令名不能为空")
        if _NAME_INVALID_RE.search(name):
            raise ValueError(f"指令名包含非法字符（空白或路径分隔符）: {spec.name!r}")
        normalized = spec.model_copy(update={"name": name})
        if name in self._specs:
            logger.info("指令 %r 已存在，覆盖注册（origin=%s）", name, spec.origin)
        self._specs[name] = normalized
        return normalized

    def register_handler(
        self,
        name: str,
        handler: object,
        *,
        keywords: Sequence[str] = (),
        help_text: str = "",
        description: str = "",
        permission: PermissionLevel = PermissionLevel.FULL,
        enabled: bool = True,
        origin: str = "builtin",
    ) -> ScriptSpec:
        """注册进程内内置指令（不落盘，如 ``/help``）。

        内置处理器是运营者代码，运行在宿主解释器内，不过子进程沙箱，
        因此权限档位无实际约束（仅作声明）。
        """
        spec = ScriptSpec(
            name=name,
            keywords=tuple(keywords),
            help_text=help_text,
            description=description,
            handler=handler,  # type: ignore[arg-type]
            permission=permission,
            enabled=enabled,
            origin=origin,
        )
        return self.register(spec)

    def unregister(self, name: str) -> None:
        """注销一条指令（不存在时静默忽略）。"""
        self._specs.pop(name.strip().lower(), None)

    # -- 解析 ----------------------------------------------------------------

    def resolve_exact(self, name: str) -> ScriptSpec | None:
        """按指令名精确匹配（大小写不敏感）。"""
        return self._specs.get(name.strip().lower())

    def resolve_prefix(self, name: str) -> ScriptSpec | None:
        """指令名唯一前缀匹配（``/he`` → ``hello``，仅一个候选时命中）。"""
        prefix = name.strip().lower()
        if not prefix:
            return None
        candidates = [
            spec for cmd, spec in self._specs.items() if cmd.startswith(prefix)
        ]
        if len(candidates) == 1:
            return candidates[0]
        return None

    def resolve_keyword(self, text: str) -> ScriptSpec | None:
        """关键词模糊匹配：文本包含某指令的关键词即命中。

        多个候选时取最长关键词（更具体者优先），与 messaging 层一致。
        """
        lowered = text.strip().lower()
        if not lowered:
            return None
        best: tuple[int, ScriptSpec] | None = None
        for spec in self._specs.values():
            for keyword in spec.keywords:
                kw = keyword.strip().lower()
                if kw and kw in lowered:
                    if best is None or len(kw) > best[0]:
                        best = (len(kw), spec)
        return best[1] if best else None

    # -- 查询与展示 ----------------------------------------------------------

    def commands(self) -> list[ScriptSpec]:
        """全部指令（按名称排序）。"""
        return [self._specs[name] for name in sorted(self._specs)]

    def __len__(self) -> int:
        return len(self._specs)

    def render_help(self, extra_lines: Sequence[str] = ()) -> str:
        """渲染帮助菜单文本。

        :param extra_lines: 外部补充行（如 messaging 层尚未就绪的
            role / agent 命令），原样追加在脚本指令之后。
        """
        lines = ["可用指令："]
        for spec in self.commands():
            if not spec.enabled:
                continue
            text = spec.help_text or (
                f"/{spec.name} — {spec.description or '自定义脚本'}"
            )
            lines.append(text)
        lines.extend(extra_lines)
        lines.append("发送 /指令 [参数]，或直接输入关键词即可触发")
        return "\n".join(lines)

    # -- 目录扫描 --------------------------------------------------------------

    def load_directory(
        self, path: Path, default_permission: PermissionLevel = PermissionLevel.STANDARD
    ) -> list[ScriptSpec]:
        """扫描目录下的 ``*.py`` 并注册为指令。

        规则：

        - 跳过 ``_`` 开头与 ``__init__.py`` 的文件；
        - 文件头部 ``# script:`` 注释可声明 ``name`` / ``keywords`` /
          ``help`` / ``description`` / ``permission`` / ``timeout`` /
          ``enabled``，未声明则用文件名（小写）与默认帮助文本；
        - 目录不存在时静默返回空列表（首次运行无 ``scripts/`` 属正常）。

        :param path: 脚本目录。
        :param default_permission: 未声明档位的脚本使用的默认档位。
        :returns: 本次注册的 spec 列表（按文件名排序）。
        """
        directory = Path(path)
        if not directory.is_dir():
            logger.debug("脚本目录不存在，跳过扫描: %s", directory)
            return []

        registered: list[ScriptSpec] = []
        for entry in sorted(directory.glob("*.py")):
            if entry.name.startswith("_") or entry.stem.startswith("__"):
                continue
            try:
                spec = self._spec_from_file(entry, default_permission)
            except ValueError as exc:
                logger.warning("跳过无法注册的脚本 %s: %s", entry, exc)
                continue
            self.register(spec)
            registered.append(spec)
        if registered:
            logger.info(
                "从目录加载 %d 个脚本: %s",
                len(registered),
                ", ".join(spec.name for spec in registered),
            )
        return registered

    def _spec_from_file(
        self, entry: Path, default_permission: PermissionLevel
    ) -> ScriptSpec:
        """从单个脚本文件构造 spec（解析头部元信息注释）。"""
        meta = self._parse_meta(entry)
        name = meta.get("name") or entry.stem.lower()
        if _NAME_INVALID_RE.search(name.strip().lower()):
            raise ValueError(f"指令名包含非法字符: {name!r}")

        keywords = tuple(
            kw.strip()
            for kw in re.split(r"[,，]", meta.get("keywords", ""))
            if kw.strip()
        )
        permission = default_permission
        if "permission" in meta:
            try:
                permission = PermissionLevel(meta["permission"].strip().lower())
            except ValueError:
                logger.warning(
                    "脚本 %s 的 permission=%r 无法识别，使用默认档位 %s",
                    entry,
                    meta["permission"],
                    default_permission.value,
                )
        timeout: float | None = None
        if "timeout" in meta:
            try:
                timeout = float(meta["timeout"])
                if timeout <= 0:
                    raise ValueError
            except ValueError:
                logger.warning(
                    "脚本 %s 的 timeout=%r 无法识别，使用模块默认值",
                    entry,
                    meta["timeout"],
                )
                timeout = None
        enabled = True
        if "enabled" in meta:
            enabled = meta["enabled"].strip().lower() in {"1", "true", "yes", "on"}

        description = meta.get("description", "")
        help_text = meta.get("help") or f"/{name} — {description or '自定义脚本'}"
        return ScriptSpec(
            name=name,
            keywords=keywords,
            help_text=help_text,
            description=description,
            script_type=ScriptType.PYTHON,
            entry=entry.resolve(),  # 绝对路径：子进程 cwd 是沙箱目录
            timeout_seconds=timeout,
            permission=permission,
            enabled=enabled,
            origin=str(entry),
        )

    @staticmethod
    def _parse_meta(entry: Path) -> dict[str, str]:
        """解析脚本头部第一个 ``# script:`` 注释行。

        格式：``# script: key=value; key=value``，分号分隔、取第一个 ``=``
        切分；无法读取文件时返回空字典（用默认值注册）。
        """
        try:
            head = entry.read_text(encoding="utf-8", errors="replace")[:4096]
        except OSError as exc:
            logger.warning("读取脚本头部失败，使用默认元信息: %s: %s", entry, exc)
            return {}

        for line in head.splitlines():
            match = _META_LINE_RE.match(line.strip())
            if match is None:
                continue
            meta: dict[str, str] = {}
            for part in match.group("meta").split(";"):
                key, sep, value = part.partition("=")
                key = key.strip().lower()
                if not sep or not key:
                    continue
                if key not in _META_KEYS:
                    logger.warning("脚本 %s 的元信息键 %r 暂不支持，已忽略", entry, key)
                    continue
                meta[key] = value.strip()
            return meta
        return {}
