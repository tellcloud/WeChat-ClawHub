"""WeChat-ClawHub AI 对话模块：LLM 对话和角色系统。

对应 README 目录结构中的 ``modules/ai/``：

- :mod:`~modules.ai.service`    — AI 服务：组织完整的 AI 对话流程；
- :mod:`~modules.ai.persona`    — 角色管理：加载、切换和管理 AI Persona；
- :mod:`~modules.ai.memory`     — 记忆系统：短期上下文 + 长期记忆；
- :mod:`~modules.ai.context`    — AI 上下文：组装 Prompt、历史消息和记忆；
- :mod:`~modules.ai.providers`  — LLM Provider 适配层（GLM / DeepSeek / Qwen）。

典型数据流::

    Message ──► AIService.handle_message ──► ReplySink.send_text

公开接口::

    from modules.ai import AIService, AIServiceConfig, PersonaManager, GLMProvider

    service = AIService()                       # 读环境变量装配（GLM_API_KEY）
    reply = await service.handle_message(message, sender)
"""

from __future__ import annotations

from .context import AssembledContext, ContextBuilder, ContextConfig
from .memory import (
    ConversationMemory,
    FactExtractor,
    HeuristicFactExtractor,
    LLMSummarizer,
    LongTermMemory,
    MemoryEntry,
    MemorySummarizer,
    NullSummarizer,
)
from .persona import (
    BUILTIN_DEFAULT_PERSONA,
    DEFAULT_PERSONA_NAME,
    Persona,
    PersonaError,
    PersonaManager,
    PersonaNotFoundError,
    PersonaValidationError,
)
from .providers import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatRole,
    DeepSeekConfig,
    DeepSeekProvider,
    GLMConfig,
    GLMProvider,
    LLMProvider,
    ProviderAPIError,
    ProviderAuthError,
    ProviderConfig,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    QwenConfig,
    QwenProvider,
    Usage,
    available_providers,
    create_provider,
)
from .service import AIService, AIServiceConfig, ReplySink

__all__ = [
    # 服务
    "AIService",
    "AIServiceConfig",
    "ReplySink",
    # 角色
    "Persona",
    "PersonaManager",
    "PersonaError",
    "PersonaNotFoundError",
    "PersonaValidationError",
    "DEFAULT_PERSONA_NAME",
    "BUILTIN_DEFAULT_PERSONA",
    # 记忆
    "ConversationMemory",
    "LongTermMemory",
    "MemoryEntry",
    "MemorySummarizer",
    "LLMSummarizer",
    "NullSummarizer",
    "FactExtractor",
    "HeuristicFactExtractor",
    # 上下文
    "ContextBuilder",
    "ContextConfig",
    "AssembledContext",
    # Provider
    "LLMProvider",
    "GLMProvider",
    "GLMConfig",
    "DeepSeekProvider",
    "DeepSeekConfig",
    "QwenProvider",
    "QwenConfig",
    "ProviderConfig",
    "ProviderError",
    "ProviderAuthError",
    "ProviderRateLimitError",
    "ProviderAPIError",
    "ProviderTimeoutError",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ChatRole",
    "Usage",
    "create_provider",
    "available_providers",
]
