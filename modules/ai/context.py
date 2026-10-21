"""AI 上下文：组装 Prompt、历史消息和记忆。

对应 README 目录结构中的 ``modules/ai/context.py``。

:class:`ContextBuilder` 把角色、长期记忆、对话摘要、历史消息与当前
用户消息组装为最终发给 Provider 的消息列表：

1. **system 消息**：角色 ``system_prompt`` + 可选的两个附加块——
   长期记忆（``[长期记忆]``，让 AI「记得」用户事实）与对话摘要
   （``[对话摘要]``，压缩后的早期上下文）；
2. **历史消息**：原样透传（滑动窗口已由记忆层控制）；
3. **当前用户消息**：引用消息以「（引用消息：…）」前缀拼接。

组装遵循字符预算 ``max_chars``：超预算时从最旧的历史消息开始裁剪，
保证 system 提示与当前消息永远完整（模型最重要的指令不丢）。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Optional

from pydantic import BaseModel, Field

from .memory import MemoryEntry
from .persona import Persona
from .providers.base import ChatMessage, ChatRequest, ChatRole

logger = logging.getLogger(__name__)


class ContextConfig(BaseModel):
    """上下文组装配置。"""

    #: 最终消息列表的总字符预算（超预算时从旧到新裁剪历史）
    max_chars: int = Field(default=6000, description="总字符预算")
    #: 是否注入长期记忆
    include_long_term: bool = True
    #: 是否注入对话摘要
    include_summary: bool = True


class AssembledContext(BaseModel):
    """组装结果（附诊断信息，便于日志与测试断言）。"""

    messages: list[ChatMessage] = Field(default_factory=list)
    #: 最终使用的 system 提示（含记忆 / 摘要附加块）
    system_prompt: str = ""
    #: 透传的历史消息条数（裁剪后）
    history_count: int = 0
    #: 注入的长期记忆条数
    memory_count: int = 0
    #: 注入的对话摘要字符数（0 表示无摘要）
    summary_chars: int = 0
    #: 是否发生过历史裁剪
    truncated: bool = False
    #: 组装后的总字符数
    total_chars: int = 0

    def to_request(
        self,
        *,
        persona: Persona,
        default_model: str = "",
        default_temperature: float = 0.7,
        default_max_tokens: int = 1024,
    ) -> ChatRequest:
        """转为 Provider 请求（角色覆盖优先于全局默认）。"""
        return ChatRequest(
            messages=list(self.messages),
            model=persona.model or default_model,
            temperature=(
                persona.temperature
                if persona.temperature is not None
                else default_temperature
            ),
            max_tokens=(
                persona.max_tokens
                if persona.max_tokens is not None
                else default_max_tokens
            ),
        )


class ContextBuilder:
    """上下文组装器（无状态，可复用）。"""

    def __init__(self, config: Optional[ContextConfig] = None) -> None:
        self.config = config or ContextConfig()

    def build(
        self,
        *,
        persona: Persona,
        history: Sequence[ChatMessage] = (),
        user_text: str,
        summary: str = "",
        memories: Sequence[MemoryEntry] = (),
        quoted_text: str = "",
    ) -> AssembledContext:
        """组装一次对话的完整上下文。

        :param persona: 当前角色；
        :param history: 短期记忆中的历史消息（不含当前用户消息）；
        :param user_text: 当前用户消息文本；
        :param summary: 对话摘要（早期上下文的压缩）；
        :param memories: 召回到的长期记忆条目；
        :param quoted_text: 被引用的消息文本（拼接到用户消息前）。
        """
        system_prompt = self._build_system_prompt(persona, summary, memories)
        messages: list[ChatMessage] = []
        if system_prompt.strip():
            messages.append(
                ChatMessage(role=ChatRole.SYSTEM, content=system_prompt)
            )

        # 历史消息按字符预算裁剪（从最旧开始丢）
        budget = self.config.max_chars - len(system_prompt) - len(user_text)
        kept, truncated = self._fit_history(history, budget)
        messages.extend(kept)
        if truncated:
            logger.debug(
                "历史消息超预算已裁剪: 保留 %d/%d 条", len(kept), len(history)
            )

        user_content = self._build_user_content(user_text, quoted_text)
        messages.append(ChatMessage(role=ChatRole.USER, content=user_content))

        return AssembledContext(
            messages=messages,
            system_prompt=system_prompt,
            history_count=len(kept),
            memory_count=len(memories) if self.config.include_long_term else 0,
            summary_chars=len(summary) if self.config.include_summary else 0,
            truncated=truncated,
            total_chars=sum(len(m.content) for m in messages),
        )

    # -- 内部 ---------------------------------------------------------------

    def _build_system_prompt(
        self,
        persona: Persona,
        summary: str,
        memories: Sequence[MemoryEntry],
    ) -> str:
        """角色提示 + 长期记忆 + 对话摘要。"""
        blocks: list[str] = []
        if persona.system_prompt.strip():
            blocks.append(persona.system_prompt.strip())

        if self.config.include_long_term and memories:
            lines = [f"- {entry.content}" for entry in memories]
            blocks.append("[长期记忆]\n" + "\n".join(lines))

        if self.config.include_summary and summary.strip():
            blocks.append(f"[对话摘要]\n{summary.strip()}")

        return "\n\n".join(blocks)

    def _fit_history(
        self, history: Sequence[ChatMessage], budget: int
    ) -> tuple[list[ChatMessage], bool]:
        """在字符预算内保留尽可能新的历史消息。"""
        if budget <= 0:
            return [], bool(history)

        kept_reversed: list[ChatMessage] = []
        used = 0
        truncated = False
        for message in reversed(list(history)):
            cost = len(message.content)
            if used + cost > budget:
                truncated = True
                continue  # 跳过更旧的消息，继续尝试更短的
            kept_reversed.append(message)
            used += cost
        kept = list(reversed(kept_reversed))
        return kept, truncated

    @staticmethod
    def _build_user_content(user_text: str, quoted_text: str) -> str:
        """当前用户消息（引用消息以前缀形式拼接）。"""
        text = user_text.strip()
        if quoted_text.strip():
            return f"（引用消息：{quoted_text.strip()}）\n{text}"
        return text
