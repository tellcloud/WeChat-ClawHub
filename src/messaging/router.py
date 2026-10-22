"""消息路由：判断消息应该交给 Script、AI 还是 Agent。

职责（对应 README 目录结构中的 ``router.py``）：

- 输入：统一消息 :class:`~messaging.models.Message`；
- 输出：:class:`RoutingResult`——目标模块 + 命令 + 参数 + 匹配信息；
- 三级匹配（README 架构设计）：精确匹配 → 关键词模糊匹配 → 兜底帮助菜单；
- 粘性模式：会话当前停留在某个智能体时，非指令消息持续交给 Agent
  （由 ``mode_provider`` 提供，未来由 session 层实现）；
- 群聊策略：未 @ 到 Bot 的非指令消息可配置为忽略（``IGNORE``）。

路由规则不依赖具体业务模块：命令通过 :class:`CommandRegistry` 注册，
未来 ``modules/script`` 的注册表只需适配同一接口即可接入。
"""

from __future__ import annotations

import logging
import re
import shlex
from enum import StrEnum
from typing import Callable, Iterable, Optional

from pydantic import BaseModel, Field

from .models import ChatType, ContentKind, Message

logger = logging.getLogger(__name__)


class RouteTarget(StrEnum):
    """路由目标：消息应该交给哪个模块处理。"""

    SCRIPT = "script"  #: 脚本指令模块
    AI = "ai"  #: AI 对话模块
    AGENT = "agent"  #: 智能体模块
    IGNORE = "ignore"  #: 忽略（如群聊中未 @ Bot 的消息）


class MatchType(StrEnum):
    """匹配方式（对应三级匹配 + 粘性 / 默认）。"""

    EXACT = "exact"  #: 指令精确匹配
    KEYWORD = "keyword"  #: 关键词模糊匹配
    FALLBACK = "fallback"  #: 兜底（未知指令 → 帮助菜单）
    STICKY = "sticky"  #: 粘性模式（当前智能体）
    DEFAULT = "default"  #: 默认（AI 对话）


class CommandSpec(BaseModel):
    """命令注册项。

    ``name`` 为指令主名（``/`` 后第一个 token）；``keywords`` 为触发
    同一条命令的关键词（中文别名等），用于无前缀的关键词模糊匹配。
    """

    name: str = Field(description="指令主名，如 help / role / weather")
    keywords: tuple[str, ...] = Field(default=(), description="关键词别名（小写比较）")
    help_text: str = Field(default="", description="帮助菜单展示文本")
    target: RouteTarget = Field(default=RouteTarget.SCRIPT, description="命令归属模块")


class CommandRegistry:
    """命令注册表：精确匹配 + 关键词模糊匹配。

    用法::

        registry = CommandRegistry()
        registry.register(CommandSpec(name="weather", keywords=("天气", "查天气")))
        registry.register(CommandSpec(name="help", keywords=("帮助",)))

    未来 ``modules/script`` 可继承或持有一个本注册表实例，
    把脚本指令统一登记到这里。
    """

    def __init__(self, specs: Iterable[CommandSpec] = ()) -> None:
        self._specs: dict[str, CommandSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: CommandSpec) -> CommandSpec:
        """注册一条命令（同名覆盖，用于热更新）。"""
        name = spec.name.strip().lower()
        if not name:
            raise ValueError("命令名不能为空")
        # pydantic 不可变字段重建，保证 name 归一化
        normalized = spec.model_copy(update={"name": name})
        self._specs[name] = normalized
        return normalized

    def unregister(self, name: str) -> None:
        """注销一条命令（不存在时静默忽略）。"""
        self._specs.pop(name.strip().lower(), None)

    def resolve_exact(self, name: str) -> Optional[CommandSpec]:
        """按指令名精确匹配（大小写不敏感）。"""
        return self._specs.get(name.strip().lower())

    def resolve_prefix(self, name: str) -> Optional[CommandSpec]:
        """指令名唯一前缀匹配（``/we`` → ``weather``，仅一个候选时命中）。"""
        prefix = name.strip().lower()
        if not prefix:
            return None
        candidates = [
            spec for cmd, spec in self._specs.items() if cmd.startswith(prefix)
        ]
        if len(candidates) == 1:
            return candidates[0]
        return None

    def resolve_keyword(self, text: str) -> Optional[CommandSpec]:
        """关键词模糊匹配：文本包含某命令的关键词即命中。

        多个候选时取最长关键词（更具体者优先），与 README 的
        “精确 → 模糊” 三级匹配对应。
        """
        lowered = text.strip().lower()
        if not lowered:
            return None
        best: Optional[tuple[int, CommandSpec]] = None
        for spec in self._specs.values():
            for keyword in spec.keywords:
                kw = keyword.strip().lower()
                if kw and kw in lowered:
                    if best is None or len(kw) > best[0]:
                        best = (len(kw), spec)
        return best[1] if best else None

    def commands(self) -> list[CommandSpec]:
        """全部命令（按名称排序，供帮助菜单使用）。"""
        return [self._specs[name] for name in sorted(self._specs)]


#: 粘性模式提供者：入参 ``session_key``，返回当前会话应持续走的目标
ModeProvider = Callable[[str], Optional[RouteTarget]]

#: 文本开头的 @ 提及 token（normalizer 可能已剥离，router 再兜底一次）
_LEADING_MENTION_RE = re.compile(r"^(?:@[^\s@\u2005]+[\s\u2005]*)+")


class RouterConfig(BaseModel):
    """路由行为配置。"""

    #: 指令前缀（微信场景习惯用 ``/``）
    command_prefix: str = "/"
    #: 是否启用无前缀的关键词模糊匹配
    enable_keyword_match: bool = True
    #: 群聊中未 @ Bot 的非指令消息是否忽略
    group_require_mention: bool = False
    #: Bot 的群昵称列表（大小写不敏感）；为空时任意 @ 都视为指向 Bot
    bot_names: tuple[str, ...] = ()
    #: 未知指令兜底命令（帮助菜单）
    fallback_command: str = "help"
    #: 非文本消息的默认目标（媒体消息交给 AI 处理）
    non_text_target: RouteTarget = RouteTarget.AI


class RoutingResult(BaseModel):
    """路由决策结果。

    ``message`` 透传标准化后的消息，dispatcher 可直接使用；
    ``command`` / ``args`` 仅在 ``target=SCRIPT`` 时有值。
    """

    target: RouteTarget
    match_type: MatchType
    message: Message
    command: Optional[str] = None
    args: list[str] = Field(default_factory=list)
    keyword: Optional[str] = Field(default=None, description="命中的关键词（模糊匹配）")
    reason: str = Field(default="", description="人类可读的决策理由（日志用）")


def default_registry() -> CommandRegistry:
    """内置命令注册表（README「内置指令」）。

    具体执行由未来的 script / session / agent 模块实现，
    这里只负责让路由层认识这些指令。
    """

    return CommandRegistry(
        [
            CommandSpec(
                name="help",
                keywords=("帮助", "menu", "菜单"),
                help_text="/help — 显示所有可用指令",
            ),
            CommandSpec(
                name="role",
                keywords=("角色",),
                help_text="/role list|set <名称> — 列出 / 切换 AI 角色",
            ),
            CommandSpec(
                name="llm",
                keywords=("模型", "llm"),
                help_text="/llm list|set <名称> — 列出 / 切换 LLM 模型",
            ),
            CommandSpec(
                name="agent",
                keywords=("智能体",),
                help_text="/agent list|use <名称> — 列出 / 切换智能体",
            ),
        ]
    )


class MessageRouter:
    """消息路由器：决定每条消息的去向。

    用法::

        router = MessageRouter()                      # 使用内置命令表
        result = router.route(message)
        if result.target is RouteTarget.SCRIPT:
            await run_command(result.command, result.args)
        elif result.target is RouteTarget.AGENT:
            await agent.handle(result.message)

    :param registry: 命令注册表，默认 :func:`default_registry`；
    :param mode_provider: 粘性模式提供者（``session_key → RouteTarget``），
        例如用户 ``/agent use codex`` 后，返回 ``RouteTarget.AGENT``；
    :param config: 路由行为配置。
    """

    def __init__(
        self,
        registry: Optional[CommandRegistry] = None,
        *,
        mode_provider: Optional[ModeProvider] = None,
        config: Optional[RouterConfig] = None,
    ) -> None:
        self.registry = registry or default_registry()
        self.mode_provider = mode_provider
        self.config = config or RouterConfig()

    # -- 主入口 ---------------------------------------------------------------

    def route(self, message: Message) -> RoutingResult:
        """对一条统一消息作出路由决策（纯函数，无副作用）。"""
        # 1. 非文本消息（图片 / 语音 / 文件 / 视频 / 空消息）
        if message.kind is not ContentKind.TEXT or not message.text.strip():
            return self._route_non_text(message)

        text = message.text.strip()
        prefix = self.config.command_prefix

        # 2. 指令消息：/命令 [参数...]
        body = self._strip_leading_mentions(text)
        if prefix and body.startswith(prefix):
            return self._route_command(message, body[len(prefix) :].strip())

        # 3. 群聊策略：未 @ Bot 的非指令消息
        if (
            message.chat_type is ChatType.GROUP
            and self.config.group_require_mention
            and not self._is_addressed_to_bot(message)
        ):
            return RoutingResult(
                target=RouteTarget.IGNORE,
                match_type=MatchType.DEFAULT,
                message=message,
                reason="群聊消息未 @ Bot，忽略",
            )

        # 4. 粘性模式：当前会话停留在某个智能体
        sticky = self._sticky_target(message.session_key)
        if sticky is not None and sticky is not RouteTarget.IGNORE:
            return RoutingResult(
                target=sticky,
                match_type=MatchType.STICKY,
                message=message,
                reason=f"会话粘性模式: {sticky.value}",
            )

        # 5. 关键词模糊匹配（无前缀，如「查天气 北京」）
        if self.config.enable_keyword_match:
            spec = self.registry.resolve_keyword(text)
            if spec is not None:
                keyword = self._matched_keyword(spec, text)
                name, args = self._split_command(text, keyword)
                return RoutingResult(
                    target=spec.target,
                    match_type=MatchType.KEYWORD,
                    message=message,
                    command=spec.name,
                    args=args,
                    keyword=keyword,
                    reason=f"关键词命中: {keyword}",
                )

        # 6. 默认：AI 对话
        return RoutingResult(
            target=RouteTarget.AI,
            match_type=MatchType.DEFAULT,
            message=message,
            reason="默认进入 AI 对话",
        )

    # -- 内部 ----------------------------------------------------------------

    def _route_non_text(self, message: Message) -> RoutingResult:
        """非文本消息：走配置的默认目标（粘性优先）。"""
        sticky = self._sticky_target(message.session_key)
        if sticky is not None and sticky is not RouteTarget.IGNORE:
            return RoutingResult(
                target=sticky,
                match_type=MatchType.STICKY,
                message=message,
                reason=f"非文本消息，会话粘性模式: {sticky.value}",
            )
        return RoutingResult(
            target=self.config.non_text_target,
            match_type=MatchType.DEFAULT,
            message=message,
            reason=(
                f"非文本消息（{message.kind.value}），"
                f"默认目标: {self.config.non_text_target.value}"
            ),
        )

    def _route_command(self, message: Message, body: str) -> RoutingResult:
        """指令消息：精确 → 前缀 → 兜底帮助菜单。"""
        name, args = self._split_command(body, None)

        # 精确匹配
        spec = self.registry.resolve_exact(name)
        if spec is not None:
            return RoutingResult(
                target=spec.target,
                match_type=MatchType.EXACT,
                message=message,
                command=spec.name,
                args=args,
                reason=f"指令精确匹配: /{spec.name}",
            )

        # 指令名唯一前缀模糊匹配（/we → weather）
        spec = self.registry.resolve_prefix(name)
        if spec is not None:
            return RoutingResult(
                target=spec.target,
                match_type=MatchType.KEYWORD,
                message=message,
                command=spec.name,
                args=args,
                reason=f"指令前缀匹配: /{name} → /{spec.name}",
            )

        # 兜底：未知指令 → 帮助菜单
        return RoutingResult(
            target=RouteTarget.SCRIPT,
            match_type=MatchType.FALLBACK,
            message=message,
            command=self.config.fallback_command,
            args=[name, *args],
            reason=f"未知指令 /{name}，兜底帮助菜单",
        )

    def _sticky_target(self, session_key: str) -> Optional[RouteTarget]:
        """查询会话粘性模式（未配置提供者时返回 None）。"""
        if self.mode_provider is None:
            return None
        try:
            return self.mode_provider(session_key)
        except Exception:
            logger.exception("mode_provider 查询失败: session=%s", session_key)
            return None

    def _is_addressed_to_bot(self, message: Message) -> bool:
        """判断群聊消息是否 @ 到 Bot。

        未配置 ``bot_names`` 时，任意 @ 都视为指向 Bot（宽松策略）。
        """
        if not message.mentions:
            return False
        if not self.config.bot_names:
            return True
        bot_names = {name.lower() for name in self.config.bot_names}
        return any(mention.lower() in bot_names for mention in message.mentions)

    @staticmethod
    def _strip_leading_mentions(text: str) -> str:
        """剥离文本开头的连续 @ 提及（``@Bot /help`` → ``/help``）。"""
        return _LEADING_MENTION_RE.sub("", text).strip()

    @staticmethod
    def _split_command(body: str, keyword: Optional[str]) -> tuple[str, list[str]]:
        """把指令体拆成 (命令名, 参数列表)。

        支持引号参数（``/say "hello world"``）；引号不配对时退化为
        普通空白切分。关键词匹配场景下用关键词后的剩余部分作为参数。
        """
        if keyword:
            index = body.lower().find(keyword.lower())
            if index >= 0:
                rest = body[index + len(keyword) :].strip()
                return keyword, _split_args(rest)
        name, _, rest = body.partition(" ")
        return name.strip(), _split_args(rest.strip())

    @staticmethod
    def _matched_keyword(spec: CommandSpec, text: str) -> str:
        """返回实际命中的关键词（用于参数切分与日志）。"""
        lowered = text.lower()
        best = ""
        for keyword in spec.keywords:
            if keyword.lower() in lowered and len(keyword) > len(best):
                best = keyword
        return best or spec.name


def _split_args(text: str) -> list[str]:
    """参数切分：优先 shlex（支持引号），失败退化为空白切分。"""
    if not text:
        return []
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()
