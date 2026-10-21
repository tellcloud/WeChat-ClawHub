"""LLM Provider 基类：定义统一的 LLM 接口。

对应 README 目录结构中的 ``modules/ai/providers/base.py``。

本模块只定义抽象契约，不绑定任何具体模型服务：

- :class:`ChatMessage` / :class:`ChatRequest` / :class:`ChatResponse`——
  跨 Provider 统一的对话数据结构（OpenAI 风格，业界事实标准）；
- :class:`ProviderConfig`——Provider 通用运行参数（密钥、基址、超时、重试）；
- :class:`LLMProvider`——抽象基类，``chat()`` 内置指数退避重试，
  子类只需实现 ``_chat_once()`` 完成一次真实调用；
- 异常体系：:class:`ProviderError` 及认证 / 限流 / API / 超时子类，
  通过 ``retryable`` 标记告诉基类哪些错误值得重试。

具体 Provider（GLM 等）在 :mod:`ai.providers.glm` 中实现，
通过 :func:`ai.providers.create_provider` 工厂按名称创建。
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 统一对话数据结构
# ---------------------------------------------------------------------------


class ChatRole(StrEnum):
    """对话角色（与 OpenAI 风格一致，GLM v4 同样兼容）。"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """一条对话消息。"""

    role: ChatRole
    content: str


class ChatRequest(BaseModel):
    """一次对话补全请求。

    ``model`` / ``temperature`` / ``max_tokens`` 允许为空或 ``None``，
    由 Provider 使用自身配置的默认值兜底（角色可覆盖默认值）。
    """

    messages: list[ChatMessage] = Field(default_factory=list)
    model: str = ""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: bool = False


class Usage(BaseModel):
    """Token 用量（Provider 不返回时全为 0）。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    """一次对话补全响应。"""

    content: str = ""
    model: str = ""
    finish_reason: str = ""
    usage: Usage = Field(default_factory=Usage)


# ---------------------------------------------------------------------------
# 配置与异常
# ---------------------------------------------------------------------------


class ProviderConfig(BaseModel):
    """Provider 通用配置。

    优先级：显式传参 > 环境变量（见各 Provider 的 ``from_env``）> 默认值。
    """

    api_key: str = Field(default="", description="API Key（Bearer 认证）")
    base_url: str = Field(default="", description="API 基址")
    model: str = Field(default="", description="默认模型名")
    temperature: float = Field(default=0.7, description="默认温度")
    max_tokens: int = Field(default=1024, description="默认最大生成 token 数")
    timeout: float = Field(default=30.0, description="单次请求超时（秒）")
    max_retries: int = Field(default=2, description="失败重试次数（不含首次）")
    retry_base_delay: float = Field(default=1.0, description="重试基础退避（秒）")
    retry_max_delay: float = Field(default=8.0, description="重试退避上限（秒）")


class ProviderError(Exception):
    """LLM Provider 统一异常基类。

    ``retryable`` 标记该错误是否值得由基类的重试逻辑自动重试；
    ``status`` 为 HTTP 状态码（非 HTTP 错误为 ``None``）。
    """

    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        status: Optional[int] = None,
        retryable: Optional[bool] = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        if retryable is not None:
            self.retryable = retryable


class ProviderAuthError(ProviderError):
    """认证失败（401 / 403）：Key 错误或过期，重试无意义。"""


class ProviderRateLimitError(ProviderError):
    """触发限流（429）：尊重 ``retry_after`` 退避后重试。"""

    retryable = True

    def __init__(
        self,
        message: str,
        *,
        status: Optional[int] = 429,
        retry_after: Optional[float] = None,
    ) -> None:
        super().__init__(message, status=status)
        self.retry_after = retry_after


class ProviderAPIError(ProviderError):
    """API 业务错误（4xx / 5xx 或响应体非法）。"""

    def __init__(
        self,
        message: str,
        *,
        status: Optional[int] = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message, status=status, retryable=retryable)


class ProviderTimeoutError(ProviderError):
    """请求超时或网络错误：可重试。"""

    retryable = True


# ---------------------------------------------------------------------------
# Provider 抽象基类
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """统一 LLM 接口。

    子类只需实现 :meth:`_chat_once`（单次调用，不做重试）；
    :meth:`chat` 由基类提供统一的重试外壳：仅对 ``retryable`` 的异常
    按指数退避重试，认证 / 参数错误直接抛出。

    用法::

        provider = GLMProvider(GLMConfig.from_env())
        async with provider:
            response = await provider.chat(ChatRequest(messages=[...]))
    """

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    # -- 子类实现 ------------------------------------------------------------

    @abstractmethod
    async def _chat_once(self, request: ChatRequest) -> ChatResponse:
        """执行一次对话补全（不处理重试）。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider 名称（如 ``glm``），用于日志与工厂。"""

    # -- 公共入口 ------------------------------------------------------------

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """带重试的对话补全。

        :raises ProviderError: 重试耗尽或遇到不可重试错误。
        """
        attempts = self.config.max_retries + 1
        for attempt in range(attempts):
            try:
                return await self._chat_once(request)
            except ProviderError as exc:
                if not exc.retryable or attempt == attempts - 1:
                    raise
                delay = self._backoff_delay(attempt, exc)
                logger.warning(
                    "%s 调用失败（第 %d/%d 次），%.1f 秒后重试: %s",
                    self.name,
                    attempt + 1,
                    attempts,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
        # 理论不可达（循环内必定 return 或 raise）
        raise ProviderError(f"{self.name} 调用失败：重试逻辑异常")

    def _backoff_delay(self, attempt: int, exc: ProviderError) -> float:
        """计算第 ``attempt`` 次重试前的等待秒数。"""
        if isinstance(exc, ProviderRateLimitError) and exc.retry_after:
            return min(exc.retry_after, self.config.retry_max_delay)
        delay = self.config.retry_base_delay * (2**attempt)
        return min(delay, self.config.retry_max_delay)

    # -- 生命周期 ------------------------------------------------------------

    async def close(self) -> None:  # noqa: B027 - 可选钩子，默认无操作
        """释放 Provider 持有的资源（默认无操作）。"""

    async def __aenter__(self) -> LLMProvider:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.close()
