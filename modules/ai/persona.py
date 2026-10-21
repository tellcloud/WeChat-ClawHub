"""角色管理：加载、切换和管理 AI Persona。

对应 README 目录结构中的 ``modules/ai/persona.py``。

角色配置为 ``personas/<角色名>.json``（见 README「角色配置文件」）::

    {
      "name": "程序员",
      "systemPrompt": "你是一个资深全栈工程师……",
      "temperature": 0.5,
      "model": "glm-4-flash",
      "greeting": "你好，我是你的编程助手。"
    }

能力：

- :class:`PersonaManager` 扫描角色目录，按文件 stem 或 JSON ``name``
  匹配（大小写不敏感），支持热重载（文件增删 / 修改后自动重新加载）；
- 运行时可创建 / 保存角色（写回 JSON 文件）；
- 目录缺失或为空时提供内置默认角色兜底，保证服务始终可用。

解析是纯本地的：不发起网络请求。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

#: 默认角色名（``personas/default.json`` 或内置兜底）
DEFAULT_PERSONA_NAME = "default"

#: 角色配置文件名匹配（只认 .json）
_PERSONA_GLOB = "*.json"


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


class Persona(BaseModel):
    """AI 角色（人设）。

    字段兼容 README 的 camelCase JSON（``systemPrompt`` / ``maxTokens``），
    同时支持下划线命名（``populate_by_name=True``）。
    """

    name: str = Field(description="角色显示名（如「程序员」）")
    system_prompt: str = Field(
        default="",
        alias="systemPrompt",
        description="系统提示词（角色人格的核心）",
    )
    description: str = Field(default="", description="角色简介（列表展示用）")
    temperature: Optional[float] = Field(
        default=None, description="角色温度；None 表示用全局默认"
    )
    model: str = Field(default="", description="角色指定模型；空表示用全局默认")
    max_tokens: Optional[int] = Field(
        default=None, alias="maxTokens", description="角色最大生成 token 数"
    )
    greeting: str = Field(default="", description="切换到该角色时的问候语")
    tags: list[str] = Field(default_factory=list, description="标签（检索 / 展示）")

    model_config = {"populate_by_name": True}

    def summary_line(self, *, current: bool = False) -> str:
        """角色列表展示的一行文本。"""
        marker = " ← 当前" if current else ""
        description = self.description or self.system_prompt[:30] or "（无简介）"
        return f"· {self.name}{marker} — {description}"


#: 内置默认角色（personas 目录缺失时的兜底）
BUILTIN_DEFAULT_PERSONA = Persona(
    name=DEFAULT_PERSONA_NAME,
    system_prompt=(
        "你是 ClawHub，一个运行在微信里的智能助手。"
        "回答简洁、友好、直接；不确定的事情如实说明，不要编造。"
    ),
    description="默认助手",
    greeting="你好，我是 ClawHub，有什么可以帮你？",
)


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------


class PersonaError(Exception):
    """角色管理统一异常。"""


class PersonaNotFoundError(PersonaError):
    """请求的角色不存在。"""

    def __init__(self, name: str, available: list[str]) -> None:
        self.name = name
        self.available = available
        listed = "、".join(available) if available else "（无）"
        super().__init__(f"角色不存在: {name!r}（可用角色: {listed}）")


class PersonaValidationError(PersonaError):
    """角色配置文件内容非法。"""


# ---------------------------------------------------------------------------
# 管理器
# ---------------------------------------------------------------------------


class PersonaManager:
    """角色加载与管理。

    用法::

        manager = PersonaManager(Path("personas"))
        manager.reload()                 # 首次加载
        manager.list()                   # 全部角色
        persona = manager.get("程序员")   # 按名称 / 文件名查找
        manager.create(Persona(name="翻译", system_prompt="..."))

    :param personas_dir: 角色配置目录（不存在时使用内置默认角色）；
    :param default_name: 默认角色名，缺省 ``"default"``；
    :param auto_reload: 是否在每次访问前检查目录变更并热重载。
    """

    def __init__(
        self,
        personas_dir: Path = Path("personas"),
        *,
        default_name: str = DEFAULT_PERSONA_NAME,
        auto_reload: bool = True,
    ) -> None:
        self.personas_dir = Path(personas_dir)
        self.default_name = default_name.strip().lower() or DEFAULT_PERSONA_NAME
        self.auto_reload = auto_reload
        self._personas: dict[str, Persona] = {}
        self._file_index: dict[str, str] = {}  # 文件 stem（小写）→ 角色名
        self._mtimes: dict[Path, float] = {}
        self._loaded = False

    # -- 加载 ---------------------------------------------------------------

    def reload(self) -> list[Persona]:
        """（重新）扫描角色目录，返回加载到的角色列表。"""
        personas: dict[str, Persona] = {}
        file_index: dict[str, str] = {}
        mtimes: dict[Path, float] = {}

        if self.personas_dir.is_dir():
            for path in sorted(self.personas_dir.glob(_PERSONA_GLOB)):
                try:
                    persona = _load_persona_file(path)
                except (PersonaValidationError, OSError) as exc:
                    logger.warning("跳过无效角色文件 %s: %s", path, exc)
                    continue
                key = persona.name.lower()
                personas[key] = persona
                file_index[path.stem.lower()] = key
                mtimes[path] = path.stat().st_mtime
        else:
            logger.info("角色目录不存在: %s（使用内置默认角色）", self.personas_dir)

        # 默认角色兜底：目录里没有默认角色时补一个内置的
        if self.default_name not in personas:
            personas[self.default_name] = BUILTIN_DEFAULT_PERSONA

        self._personas = personas
        self._file_index = file_index
        self._mtimes = mtimes
        self._loaded = True
        logger.info("已加载 %d 个角色: %s", len(personas), list(personas))
        return self.list()

    def _ensure_loaded(self) -> None:
        """懒加载 + 热重载检查。"""
        if not self._loaded:
            self.reload()
            return
        if self.auto_reload and self._directory_changed():
            logger.info("检测到角色目录变更，热重载")
            self.reload()

    def _directory_changed(self) -> bool:
        """目录内文件集合或 mtime 是否变化。"""
        if not self.personas_dir.is_dir():
            # 目录被删除：仅当此前加载过文件时才需要重载
            return bool(self._mtimes)
        current = {
            path: path.stat().st_mtime
            for path in self.personas_dir.glob(_PERSONA_GLOB)
        }
        if set(current) != set(self._mtimes):
            return True
        return any(
            mtime != self._mtimes.get(path) for path, mtime in current.items()
        )

    # -- 查询 ---------------------------------------------------------------

    def list(self) -> list[Persona]:
        """全部角色（按名称排序）。"""
        self._ensure_loaded()
        return [self._personas[key] for key in sorted(self._personas)]

    def exists(self, name: str) -> bool:
        """角色是否存在（名称 / 文件名，大小写不敏感）。"""
        self._ensure_loaded()
        return self._resolve_key(name) is not None

    def get(self, name: str) -> Persona:
        """按名称获取角色。

        匹配顺序：角色名（JSON ``name``）→ 文件 stem；均大小写不敏感。

        :raises PersonaNotFoundError: 角色不存在。
        """
        self._ensure_loaded()
        key = self._resolve_key(name)
        if key is None:
            raise PersonaNotFoundError(name, [p.name for p in self.list()])
        return self._personas[key]

    def resolve(self, name: Optional[str]) -> Persona:
        """解析角色名，为空 / 不存在时回退默认角色。"""
        if not name or not name.strip():
            return self.get_default()
        try:
            return self.get(name)
        except PersonaNotFoundError:
            logger.warning("角色 %r 不存在，回退默认角色 %r", name, self.default_name)
            return self.get_default()

    def get_default(self) -> Persona:
        """默认角色（始终可用）。"""
        self._ensure_loaded()
        return self._personas[self.default_name]

    # -- 修改 ---------------------------------------------------------------

    def create(self, persona: Persona) -> Path:
        """创建新角色并写入 ``personas/<name>.json``。

        :returns: 角色文件路径。
        :raises PersonaError: 同名角色已存在。
        """
        self._ensure_loaded()
        if persona.name.lower() in self._personas:
            raise PersonaError(f"角色已存在: {persona.name!r}")
        return self.save(persona)

    def save(self, persona: Persona) -> Path:
        """保存角色（新建或覆盖），立即生效（无需等待热重载）。"""
        path = self.personas_dir / f"{_safe_filename(persona.name)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = persona.model_dump(by_alias=True, exclude_none=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info("角色已保存: %s → %s", persona.name, path)
        # 直接更新内存索引，避免依赖下次目录巡检
        key = persona.name.lower()
        self._personas[key] = persona
        self._file_index[path.stem.lower()] = key
        self._mtimes[path] = path.stat().st_mtime
        return path

    def unregister(self, name: str) -> bool:
        """从内存中移除角色（不删除文件；返回是否移除成功）。"""
        self._ensure_loaded()
        key = self._resolve_key(name)
        if key is None:
            return False
        del self._personas[key]
        self._file_index = {
            stem: target for stem, target in self._file_index.items() if target != key
        }
        return True

    # -- 内部 ---------------------------------------------------------------

    def _resolve_key(self, name: str) -> Optional[str]:
        """名称 → 内存键（角色名优先，文件 stem 次之）。"""
        normalized = name.strip().lower()
        if not normalized:
            return None
        if normalized in self._personas:
            return normalized
        return self._file_index.get(normalized)


# ---------------------------------------------------------------------------
# 文件解析工具
# ---------------------------------------------------------------------------


def _load_persona_file(path: Path) -> Persona:
    """读取并校验单个角色 JSON 文件。"""
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PersonaValidationError(f"JSON 解析失败: {exc}") from exc
    if not isinstance(data, dict):
        raise PersonaValidationError(f"角色文件必须是 JSON 对象: {path}")

    # name 缺失时用文件名兜底
    data.setdefault("name", path.stem)
    try:
        return Persona.model_validate(data)
    except Exception as exc:  # pydantic ValidationError 等
        raise PersonaValidationError(f"字段校验失败: {exc}") from exc


def _safe_filename(name: str) -> str:
    """把角色名转为安全文件名（保留中文，替换路径敏感字符）。"""
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name).strip()
    return cleaned or "unnamed"
