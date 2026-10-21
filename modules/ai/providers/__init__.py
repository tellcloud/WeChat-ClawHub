"""LLM Provider 适配层。

公开接口::

    from modules.ai.providers import GLMProvider, GLMConfig, create_provider

    provider = create_provider("glm", GLMConfig.from_env())

- :mod:`~modules.ai.providers.base` — Provider 基类与统一数据结构；
- :mod:`~modules.ai.providers.glm`  — 智谱 GLM 接入（项目当前唯一后端）。
"""

from __future__ import annotations

from typing import Any, Optional

from .base import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatRole,
    LLMProvider,
    ProviderAPIError,
    ProviderAuthError,
    ProviderConfig,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    Usage,
)
from .glm import DEFAULT_GLM_BASE_URL, DEFAULT_GLM_MODEL, GLMConfig, GLMProvider

#: Provider 工厂注册表：名称 → (实现类, 默认配置类)
_PROVIDER_REGISTRY: dict[str, tuple[type[LLMProvider], type[ProviderConfig]]] = {
    "glm": (GLMProvider, GLMConfig),
}


def create_provider(
    name: str,
    config: Optional[Any] = None,
    **overrides: Any,
) -> LLMProvider:
    """按名称创建 Provider。

    :param name: Provider 名称（当前支持 ``glm``）；
    :param config: 配置对象或配置字段字典；缺省时用 ``from_env`` 读取环境变量；
    :param overrides: 直接覆盖的配置字段。
    :raises ValueError: 未知的 Provider 名称。
    """
    key = name.strip().lower()
    entry = _PROVIDER_REGISTRY.get(key)
    if entry is None:
        supported = ", ".join(sorted(_PROVIDER_REGISTRY))
        raise ValueError(f"未知的 LLM Provider: {name!r}（支持: {supported}）")

    provider_cls, config_cls = entry
    if config is None:
        resolved = config_cls.from_env(**overrides)  # type: ignore[attr-defined]
    elif isinstance(config, config_cls):
        resolved = config
    elif isinstance(config, dict):
        resolved = config_cls(**{**config, **overrides})
    else:
        raise TypeError(
            f"config 必须是 {config_cls.__name__}、dict 或 None，"
            f"实际 {type(config).__name__}"
        )
    return provider_cls(resolved)


def available_providers() -> list[str]:
    """已注册的 Provider 名称列表。"""
    return sorted(_PROVIDER_REGISTRY)


__all__ = [
    # 基类与数据结构
    "LLMProvider",
    "ProviderConfig",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ChatRole",
    "Usage",
    # 异常
    "ProviderError",
    "ProviderAuthError",
    "ProviderRateLimitError",
    "ProviderAPIError",
    "ProviderTimeoutError",
    # GLM
    "GLMProvider",
    "GLMConfig",
    "DEFAULT_GLM_BASE_URL",
    "DEFAULT_GLM_MODEL",
    # 工厂
    "create_provider",
    "available_providers",
]
