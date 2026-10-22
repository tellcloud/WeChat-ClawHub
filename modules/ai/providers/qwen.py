"""Qwen Provider：接入阿里云百炼 DashScope 的 Qwen（通义千问）系列模型。

对应 README 目录结构中的 ``modules/ai/providers/qwen.py``
（GLM / DeepSeek 之后补充的第三个后端）。

DashScope **兼容模式**（OpenAI 兼容风格，区别于 DashScope 原生 API）：

- 端点：``POST {base_url}/chat/completions``，默认基址
  ``https://dashscope.aliyuncs.com/compatible-mode/v1``
  （海外实例可换 ``https://dashscope-intl.aliyuncs.com/compatible-mode/v1``）；
- 认证：``Authorization: Bearer <QWEN_API_KEY>``（DashScope API Key，
  形如 ``sk-...``）；
- 请求：OpenAI 风格 ``{"model": ..., "messages": [...],
  "temperature": ..., "max_tokens": ..., "stream": false}``；
  Qwen3 系列另支持 ``enable_thinking``（思考模式），本 Provider 以
  :attr:`QwenConfig.enable_thinking` 控制：``None``（默认）时不下发，
  避免老型号报参数错误；
- 响应：``choices[0].message.content`` + ``usage`` + ``finish_reason``；
  思考模式 / Qwen3 型号另返回 ``reasoning_content``（思维链，最终
  答案仍在 ``content``，仅在 ``content`` 为空时回退）；
- 系列模型：
  - ``qwen-plus``——通用对话型号（默认，能力与速度均衡）；
  - ``qwen-turbo``——速度优先、低成本；
  - ``qwen-max``——能力最强；
  - ``qwen-long``——超长上下文（百万级 token）；
  - ``qwen-flash``——轻量快速（免费额度档）。
- 错误：HTTP 非 2xx，响应体 ``{"error": {"code": ..., "message": ...}}``
  （OpenAI 风格；常见：401 认证失败 / 400 参数错误 / 429 限流
  Throttling / 5xx 服务端故障）。

依赖项目已有的 httpx（不引入 dashscope / OpenAI SDK 等额外依赖）。
测试可通过构造参数注入挂载 ``httpx.MockTransport`` 的客户端，完全离线。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import httpx
from pydantic import Field

from .base import (
    ChatRequest,
    ChatResponse,
    LLMProvider,
    ProviderAPIError,
    ProviderAuthError,
    ProviderConfig,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    Usage,
)

logger = logging.getLogger(__name__)

#: DashScope 兼容模式默认基址（/v1 后缀已含）
DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

#: 默认模型：qwen-plus（通用对话型号）
DEFAULT_QWEN_MODEL = "qwen-plus"

#: 环境变量 → 配置字段映射（QWEN_* 优先，LLM_* 作为通用回退）
#: 注意书写顺序：靠后的键覆盖同名前面的键，因此 QWEN_API_KEY
#: 天然优先于 LLM_API_KEY。
_ENV_MAP = {
    "LLM_API_KEY": "api_key",
    "QWEN_API_KEY": "api_key",
    "QWEN_BASE_URL": "base_url",
    "QWEN_MODEL": "model",
    "QWEN_TIMEOUT": "timeout",
    "QWEN_MAX_RETRIES": "max_retries",
}


class QwenConfig(ProviderConfig):
    """Qwen Provider 配置。"""

    base_url: str = Field(default=DEFAULT_QWEN_BASE_URL)
    model: str = Field(default=DEFAULT_QWEN_MODEL)
    #: Qwen3 系列思考模式开关（``None`` 时不下发该参数）
    enable_thinking: Optional[bool] = Field(
        default=None,
        description="Qwen3 系列思考模式（None 不下发；老型号可能不支持）",
    )

    @classmethod
    def from_env(cls, **overrides: Any) -> QwenConfig:
        """从环境变量构造配置，``overrides`` 优先级最高。"""
        values: dict[str, Any] = {}
        for env_key, field in _ENV_MAP.items():
            value = os.environ.get(env_key)
            if value:
                values[field] = value
        values.update(overrides)
        return cls(**values)


class QwenProvider(LLMProvider):
    """DashScope 兼容模式 chat/completions 接入。

    用法::

        config = QwenConfig.from_env()          # 读取 QWEN_API_KEY 等
        provider = QwenProvider(config)
        async with provider:
            response = await provider.chat(ChatRequest(messages=[...]))
            print(response.content)

    :param config: Qwen 配置；
    :param client: 可选的已构造 ``httpx.AsyncClient``（测试注入
        ``httpx.MockTransport`` 用）；缺省时内部懒创建并在 ``close()`` 释放。
    """

    def __init__(
        self,
        config: Optional[QwenConfig] = None,
        *,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        super().__init__(config or QwenConfig())
        self._client = client
        self._owns_client = client is None

    @property
    def name(self) -> str:
        return "qwen"

    # -- 单次调用 ------------------------------------------------------------

    async def _chat_once(self, request: ChatRequest) -> ChatResponse:
        """执行一次 Qwen chat/completions 调用（不含重试）。"""
        config = self.config
        if not isinstance(config, QwenConfig):  # 防御：基类签名宽泛
            config = QwenConfig.model_validate(config.model_dump())
        if not config.api_key:
            raise ProviderAuthError("未配置 Qwen API Key（QWEN_API_KEY）")

        payload = self._build_payload(request, config)
        client = await self._get_client()
        try:
            response = await client.post(
                "chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {config.api_key}"},
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"Qwen 请求超时: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderTimeoutError(f"Qwen 网络错误: {exc}") from exc

        if response.status_code >= 400:
            raise self._map_http_error(response)

        return self._parse_response(response)

    # -- 请求 / 响应 ----------------------------------------------------------

    @staticmethod
    def _build_payload(request: ChatRequest, config: QwenConfig) -> dict[str, Any]:
        """组装 Qwen 请求体（None / 空值的可选字段不下发）。"""
        payload: dict[str, Any] = {
            "model": request.model or config.model,
            "messages": [
                {"role": msg.role.value, "content": msg.content}
                for msg in request.messages
            ],
            "stream": False,
        }
        temperature = request.temperature
        if temperature is None:
            temperature = config.temperature
        if temperature is not None:
            payload["temperature"] = temperature
        max_tokens = request.max_tokens
        if max_tokens is None:
            max_tokens = config.max_tokens
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        # Qwen3 思考模式：仅显式配置时下发（老型号不支持该参数）
        if config.enable_thinking is not None:
            payload["enable_thinking"] = config.enable_thinking
        return payload

    @staticmethod
    def _map_http_error(response: httpx.Response) -> ProviderError:
        """把 HTTP 错误响应映射为 Provider 异常体系。

        429（DashScope 限流 Throttling）走限流退避；5xx 可重试；
        其余 4xx（参数错误、认证失败等）重试无意义。
        """
        code, message = _parse_error_body(response)
        status = response.status_code
        detail = f"Qwen API 错误: HTTP {status} code={code} message={message}"

        if status in (401, 403):
            return ProviderAuthError(detail, status=status)
        if status == 429:
            retry_after = _parse_retry_after(response)
            return ProviderRateLimitError(
                detail, status=status, retry_after=retry_after
            )
        if status >= 500:
            return ProviderAPIError(detail, status=status, retryable=True)
        # 4xx（参数错误等）：重试无意义
        return ProviderAPIError(detail, status=status)

    @staticmethod
    def _parse_response(response: httpx.Response) -> ChatResponse:
        """解析 Qwen 成功响应。"""
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProviderAPIError(
                f"Qwen 响应不是合法 JSON: {response.text[:200]!r}"
            ) from exc

        choices = data.get("choices") or []
        if not choices:
            raise ProviderAPIError(f"Qwen 响应缺少 choices: {str(data)[:200]}")
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        if not content:
            # 思考模式（Qwen3）可能只返回 reasoning_content
            content = message.get("reasoning_content") or ""

        usage_data = data.get("usage") or {}
        return ChatResponse(
            content=content,
            model=data.get("model", ""),
            finish_reason=choice.get("finish_reason", ""),
            usage=Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
        )

    # -- 客户端管理 ----------------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        """懒创建内部 HTTP 客户端（测试注入时直接返回）。"""
        if self._client is None:
            config = self.config
            timeout = httpx.Timeout(
                config.timeout, connect=min(config.timeout, 10.0)
            )
            self._client = httpx.AsyncClient(
                base_url=config.base_url, timeout=timeout
            )
        return self._client

    async def close(self) -> None:
        """释放内部创建的客户端（外部注入的不接管其生命周期）。"""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None


# ---------------------------------------------------------------------------
# 响应解析工具
# ---------------------------------------------------------------------------


def _parse_error_body(response: httpx.Response) -> tuple[str, str]:
    """从错误响应体提取 ``(code, message)``，兼容新旧两种错误格式。"""
    try:
        data = response.json()
    except (json.JSONDecodeError, ValueError):
        return "", response.text[:200]
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            return str(error.get("code", "")), str(error.get("message", ""))
        # 旧版格式：{"code": ..., "msg": ...}
        return str(data.get("code", "")), str(data.get("msg", ""))
    return "", str(data)[:200]


def _parse_retry_after(response: httpx.Response) -> Optional[float]:
    """解析 ``Retry-After`` 头（秒），缺失或非法时返回 ``None``。"""
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return None
