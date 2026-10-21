"""AI 服务：负责组织完整的 AI 对话流程。

对应 README 目录结构中的 ``modules/ai/service.py``。

:class:`AIService` 是 AI 对话模块的唯一入口，把各子系统编排为完整闭环：

.. code-block:: text

    用户消息（messaging.Message，源自 transport 长轮询）
        │
        ▼
    ① 角色解析 ──► PersonaManager（/role set 切换，按会话记忆）
        │
        ▼
    ② 历史压缩 ──► ConversationMemory.compress（超阈值 → LLM 摘要）
        │
        ▼
    ③ 长期记忆 ──► 召回相关事实 + 从本轮消息抽取新事实
        │
        ▼
    ④ 上下文组装 ──► ContextBuilder（system + 历史 + 当前消息）
        │
        ▼
    ⑤ LLM 调用 ──► LLMProvider.chat（GLM，带 typing 状态与重试）
        │
        ▼
    ⑥ 记录与回复 ──► 写回短期记忆，经 transport MessageSender 发出

消息入口与回复出口都走 transport：入参是 messaging 层标准化后的
:class:`~messaging.models.Message`（由 transport 长轮询的
``WeixinMessage`` 经 parse → normalize 得到），回复通过满足
:class:`ReplySink` 协议的发送器（生产环境即
:class:`transport.wechat.sender.MessageSender`）发出。
"""

from __future__ import annotations

import logging
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any, Optional, Protocol

from pydantic import BaseModel, Field

from messaging.models import Message

from .context import ContextBuilder, ContextConfig
from .memory import (
    DEFAULT_HISTORY_LIMIT,
    DEFAULT_MEMORY_PATH,
    ConversationMemory,
    FactExtractor,
    HeuristicFactExtractor,
    LLMSummarizer,
    LongTermMemory,
    MemoryEntry,
    MemorySummarizer,
    NullSummarizer,
)
from .persona import DEFAULT_PERSONA_NAME, Persona, PersonaManager, PersonaNotFoundError
from .providers.base import ChatMessage, ChatRole, LLMProvider, ProviderError
from .providers.glm import GLMConfig

logger = logging.getLogger(__name__)

#: 短期记忆超过该条数时触发摘要压缩（保留 keep_recent 条最近的原文）
DEFAULT_COMPRESS_THRESHOLD = 24

#: 压缩时保留的最近消息条数
DEFAULT_KEEP_RECENT = 8


# ---------------------------------------------------------------------------
# 回复出口协议（与 transport MessageSender 结构兼容）
# ---------------------------------------------------------------------------


class ReplySink(Protocol):
    """消息回复出口协议。

    :class:`transport.wechat.sender.MessageSender`（生产）与测试替身
    都满足该协议；:class:`AIService` 只依赖这两个方法。
    """

    async def send_text(
        self,
        to_user_id: str,
        text: str,
        *,
        context_token: str = "",
        ilink_user_id: Optional[str] = None,
    ) -> None: ...

    def typing(
        self, user_id: str, context_token: str = ""
    ) -> AbstractAsyncContextManager[None]: ...


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


class AIServiceConfig(BaseModel):
    """AI 服务配置。"""

    # -- 角色 --
    personas_dir: Path = Field(default=Path("personas"), description="角色配置目录")
    default_persona: str = Field(
        default=DEFAULT_PERSONA_NAME, description="默认角色名"
    )

    # -- 短期记忆 --
    history_limit: int = Field(
        default=DEFAULT_HISTORY_LIMIT, description="滑动窗口长度（消息条数）"
    )
    compress_threshold: int = Field(
        default=DEFAULT_COMPRESS_THRESHOLD,
        description="历史超过该条数时摘要压缩更早的消息",
    )
    keep_recent: int = Field(
        default=DEFAULT_KEEP_RECENT, description="压缩时保留的最近消息条数"
    )

    # -- 长期记忆 --
    enable_long_term_memory: bool = Field(
        default=True, description="是否启用长期记忆（持久化）"
    )
    enable_auto_facts: bool = Field(
        default=True, description="是否自动从用户消息抽取事实存入长期记忆"
    )
    memory_path: Optional[Path] = Field(
        default=DEFAULT_MEMORY_PATH, description="长期记忆存储文件（None 仅内存）"
    )
    memory_recall_limit: int = Field(
        default=3, description="每轮对话注入的长期记忆条数"
    )

    # -- LLM --
    glm: GLMConfig = Field(
        default_factory=GLMConfig.from_env, description="GLM Provider 配置"
    )
    context: ContextConfig = Field(
        default_factory=ContextConfig, description="上下文组装配置"
    )

    # -- 回复文案 --
    error_reply: str = Field(
        default="AI 服务开小差了，请稍后再试。（如反复出现请检查 GLM_API_KEY 配置）",
        description="LLM 调用失败时的回复",
    )
    unconfigured_reply: str = Field(
        default="AI 对话尚未配置：请设置环境变量 GLM_API_KEY"
        "（智谱开放平台）后重启生效。",
        description="未配置 API Key 时的回复",
    )
    non_text_reply: str = Field(
        default="我目前只能理解文本消息～图片、语音、文件还看不来。",
        description="收到非文本消息时的回复",
    )

    @classmethod
    def from_env(cls, **overrides: Any) -> AIServiceConfig:
        """从环境变量构造配置（``GLM_*`` 经 :meth:`GLMConfig.from_env` 读取）。"""
        values: dict[str, Any] = {"glm": GLMConfig.from_env()}
        values.update(overrides)
        return cls(**values)


# ---------------------------------------------------------------------------
# AI 服务
# ---------------------------------------------------------------------------


class AIService:
    """AI 对话服务：角色 + 记忆 + 上下文 + LLM 的完整编排。

    用法::

        service = AIService()                     # 读环境变量装配
        reply = await service.handle_message(message, sender)

    或（测试注入假 Provider / 假记忆）::

        service = AIService(provider=FakeProvider(), config=AIServiceConfig(...))

    :param provider: LLM Provider；``None`` 表示未配置（回复引导配置文案）；
    :param config: 服务配置；
    :param persona_manager: 角色管理器（缺省按 ``config.personas_dir`` 创建）；
    :param summarizer: 摘要器（缺省：有 Provider 用 :class:`LLMSummarizer`，
        否则 :class:`NullSummarizer`）；
    :param fact_extractor: 事实抽取器（缺省 :class:`HeuristicFactExtractor`）。
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        *,
        config: Optional[AIServiceConfig] = None,
        persona_manager: Optional[PersonaManager] = None,
        summarizer: Optional[MemorySummarizer] = None,
        fact_extractor: Optional[FactExtractor] = None,
    ) -> None:
        self.config = config or AIServiceConfig()
        self.provider = provider
        self.personas = persona_manager or PersonaManager(
            self.config.personas_dir, default_name=self.config.default_persona
        )
        self.context_builder = ContextBuilder(self.config.context)
        self.conversation = ConversationMemory(max_messages=self.config.history_limit)
        self.long_term = LongTermMemory(
            self.config.memory_path if self.config.enable_long_term_memory else None
        )
        if summarizer is not None:
            self.summarizer = summarizer
        elif provider is not None:
            self.summarizer = LLMSummarizer(provider)
        else:
            self.summarizer = NullSummarizer()
        self.fact_extractor = fact_extractor or HeuristicFactExtractor()

        #: session_key → 当前角色名（内存态，重启回到默认角色）
        self._active_personas: dict[str, str] = {}

    # -- 角色管理 ------------------------------------------------------------

    def current_persona_name(self, session_key: str) -> str:
        """会话当前角色名。"""
        return self._active_personas.get(session_key, self.config.default_persona)

    def current_persona(self, session_key: str) -> Persona:
        """会话当前角色（不存在时回退默认角色）。"""
        return self.personas.resolve(self.current_persona_name(session_key))

    def set_persona(self, session_key: str, name: str) -> Persona:
        """切换会话角色。

        :raises PersonaNotFoundError: 角色不存在。
        """
        persona = self.personas.get(name)  # 不存在则抛错
        self._active_personas[session_key] = persona.name
        logger.info(
            "角色切换: session=%s → %s（各角色上下文独立保留）",
            session_key,
            persona.name,
        )
        return persona

    def list_personas(self, session_key: str = "") -> list[Persona]:
        """全部角色（按名称排序）。"""
        return self.personas.list()

    # -- 主流程 --------------------------------------------------------------

    async def handle_message(
        self, message: Message, sink: Optional[ReplySink] = None
    ) -> str:
        """处理一条 AI 对话消息，返回回复文本。

        :param message: messaging 层标准化后的消息（路由目标为 AI）；
        :param sink: 回复出口（``None`` 时只返回文本不发送，供测试）。
        """
        session_key = message.session_key
        text = message.text.strip()

        if not text:
            # 图片 / 语音 / 文件等非文本消息：礼貌说明能力边界
            logger.info(
                "AI 收到非文本消息: session=%s kind=%s", session_key, message.kind.value
            )
            return await self._reply(
                sink, message, self.config.non_text_reply
            )

        persona = self.current_persona(session_key)

        # ② 历史压缩：超阈值时把更早的消息摘要化
        await self._maybe_compress(session_key, persona.name)

        # ③ 长期记忆：先召回相关事实，再从本轮消息抽取新事实
        memories = self._recall(session_key, text)
        self._extract_facts(session_key, text, source=message.message_id)

        # ④ 组装上下文
        assembled = self.context_builder.build(
            persona=persona,
            history=self.conversation.get(session_key, persona.name),
            user_text=text,
            summary=self.conversation.summary(session_key, persona.name),
            memories=memories,
            quoted_text=message.quoted.text if message.quoted else "",
        )
        request = assembled.to_request(
            persona=persona,
            default_model=self.config.glm.model,
            default_temperature=self.config.glm.temperature,
            default_max_tokens=self.config.glm.max_tokens,
        )

        # ⑤ LLM 调用（未配置时引导配置；失败时友好降级）
        if self.provider is None:
            logger.warning("AI 对话未配置 Provider（缺少 GLM_API_KEY）")
            return await self._reply(sink, message, self.config.unconfigured_reply)

        try:
            if sink is not None:
                async with sink.typing(message.sender_id, message.context_token):
                    response = await self.provider.chat(request)
            else:
                response = await self.provider.chat(request)
        except ProviderError:
            logger.exception(
                "LLM 调用失败: session=%s persona=%s provider=%s",
                session_key,
                persona.name,
                self.provider.name,
            )
            return await self._reply(sink, message, self.config.error_reply)

        reply = response.content.strip()
        if not reply:
            logger.warning("LLM 返回空内容: session=%s", session_key)
            reply = self.config.error_reply

        # ⑥ 记录与回复
        self._record_turn(session_key, persona.name, text, reply)
        return await self._reply(sink, message, reply)

    async def handle_command(
        self,
        command: str,
        args: list[str],
        message: Message,
        sink: Optional[ReplySink] = None,
    ) -> str:
        """处理 AI 模块指令（当前为 ``/role``），返回回复文本。

        指令由 messaging 路由层识别为 SCRIPT 目标后交回本模块执行
        （README「AI 角色对话」中的 ``/role list|set <名称>``）。

        :param command: 指令名（``role``）；
        :param args: 指令参数；
        :param message: 原始消息（取 session / sender / context_token）；
        :param sink: 回复出口。
        """
        session_key = message.session_key
        sub = args[0].lower() if args else "current"
        rest = " ".join(args[1:]).strip() if len(args) > 1 else ""

        try:
            if sub == "list":
                reply = self._format_persona_list(session_key)
            elif sub == "set":
                if not rest:
                    reply = "用法：/role set <角色名>（可用 /role list 查看角色）"
                else:
                    persona = self.set_persona(session_key, rest)
                    reply = f"已切换到角色「{persona.name}」。"
                    if persona.greeting:
                        reply += persona.greeting
            elif sub == "reload":
                personas = self.personas.reload()
                reply = f"已重载角色配置，共 {len(personas)} 个角色。"
            elif sub in ("current", ""):
                persona = self.current_persona(session_key)
                suffix = f" — {persona.description}" if persona.description else ""
                reply = f"当前角色：「{persona.name}」{suffix}"
            elif sub == "info":
                name = rest or self.current_persona_name(session_key)
                persona = self.personas.get(name)
                reply = self._format_persona_info(persona)
            else:
                reply = (
                    "用法：/role list（列出角色）| /role set <名称>（切换）| "
                    "/role current（当前角色）| /role info <名称>（详情）"
                )
        except PersonaNotFoundError as exc:
            listed = "、".join(exc.available)
            reply = f"角色「{exc.name}」不存在。可用角色：{listed or '（无）'}"

        return await self._reply(sink, message, reply)

    # -- 统计 / 生命周期 -------------------------------------------------------

    def stats(self) -> dict[str, object]:
        """运行状态（调试 / 测试用）。"""
        return {
            "provider": self.provider.name if self.provider else None,
            "active_personas": dict(self._active_personas),
            "personas": [p.name for p in self.personas.list()],
            "conversation": self.conversation.stats(),
            "long_term_sessions": self.long_term.session_count(),
        }

    async def close(self) -> None:
        """释放 Provider 资源。"""
        if self.provider is not None:
            await self.provider.close()

    async def __aenter__(self) -> AIService:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.close()

    # -- 内部 ---------------------------------------------------------------

    async def _maybe_compress(self, session_key: str, persona_name: str) -> None:
        """历史超过阈值时压缩更早的消息。"""
        history = self.conversation.get(session_key, persona_name)
        if len(history) > self.config.compress_threshold:
            await self.conversation.compress(
                session_key,
                persona_name,
                keep_recent=self.config.keep_recent,
                summarizer=self.summarizer,
            )

    def _recall(self, session_key: str, text: str) -> list[MemoryEntry]:
        """召回长期记忆（未启用时返回空列表）。"""
        if not self.config.enable_long_term_memory:
            return []
        try:
            return self.long_term.recall(
                session_key, text, limit=self.config.memory_recall_limit
            )
        except Exception:
            logger.exception("长期记忆召回失败: session=%s", session_key)
            return []

    def _extract_facts(self, session_key: str, text: str, *, source: str) -> None:
        """从用户消息抽取事实并写入长期记忆。"""
        if not (self.config.enable_long_term_memory and self.config.enable_auto_facts):
            return
        try:
            facts = self.fact_extractor.extract(text)
        except Exception:
            logger.exception("事实抽取失败: session=%s", session_key)
            return
        for fact in facts:
            try:
                self.long_term.remember(session_key, fact, source=source)
                logger.info("长期记忆新增: session=%s fact=%r", session_key, fact)
            except ValueError:
                continue  # 空内容，忽略

    def _record_turn(
        self, session_key: str, persona_name: str, user_text: str, reply: str
    ) -> None:
        """把一轮对话写入短期记忆。"""
        self.conversation.add(
            session_key,
            persona_name,
            ChatMessage(role=ChatRole.USER, content=user_text),
        )
        self.conversation.add(
            session_key,
            persona_name,
            ChatMessage(role=ChatRole.ASSISTANT, content=reply),
        )

    async def _reply(
        self, sink: Optional[ReplySink], message: Message, text: str
    ) -> str:
        """经 transport 发送器回复（sink 为 None 时只返回文本）。"""
        if sink is not None:
            await sink.send_text(
                message.sender_id, text, context_token=message.context_token
            )
        logger.info(
            "AI 已回复: session=%s sender=%s len=%d",
            message.session_key,
            message.sender_id,
            len(text),
        )
        return text

    # -- 指令格式化 ----------------------------------------------------------

    def _format_persona_list(self, session_key: str) -> str:
        """``/role list`` 的角色清单。"""
        current = self.current_persona_name(session_key).lower()
        lines = ["可用角色："]
        lines.extend(
            persona.summary_line(current=persona.name.lower() == current)
            for persona in self.personas.list()
        )
        lines.append("切换：/role set <角色名>")
        return "\n".join(lines)

    def _format_persona_info(self, persona: Persona) -> str:
        """``/role info`` 的角色详情。"""
        temperature = persona.temperature
        lines = [
            f"角色：{persona.name}",
            f"简介：{persona.description or '（无）'}",
            f"系统提示：{persona.system_prompt or '（无）'}",
            f"温度：{temperature if temperature is not None else '默认'}",
            f"模型：{persona.model or '默认'}",
        ]
        if persona.greeting:
            lines.append(f"问候语：{persona.greeting}")
        return "\n".join(lines)
