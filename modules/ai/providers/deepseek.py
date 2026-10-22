"""DeepSeek Provider：接入 DeepSeek 系列模型。

对应 README 目录结构中的 ``modules/ai/providers/deepseek.py``
（GLM 之后按需补充的第二个后端，OpenAI / Anthropic 后续再按需补充）。

DeepSeek API（OpenAI 兼容风格）：

- 端点：``POST {base_url}/chat/completions``，默认基址
  ``https://api.deepseek.com``（带 ``/v1`` 后缀同样可用，方便直接
  套用 OpenAI SDK / 客户端习惯）；
- 认证：``Authorization: Bearer <DEEPSEEK_API_KEY>``（形如 ``sk-...``）；
- 请求：``{"model": ..., "messages": [{"role": ..., "content": ...}],
  "temperature": ..., "max_tokens": ..., "stream": false}``；
- 响应：OpenAI 风格 ``choices[0].message.content`` + ``usage`` +
  ``finish_reason``；推理型号另返回 ``reasoning_content``（思维链，
  最终答案仍在 ``content``，仅在 ``content`` 为空时回退）；
- 系列模型：
  - ``deepseek-chat``——通用对话型号（DeepSeek-V3.x 非思考模式，
    默认型号）；
  - ``deepseek-reasoner``——推理型号（思考模式，输出链式推理，
    部分采样参数由服务端忽略）。
- 错误：HTTP 非 2xx，响应体 ``{"error": {"code": ..., "message": ...}}``
  （OpenAI 风格；常见状态码：401 认证失败 / 402 余额不足 /
  422 参数错误 / 429 触发限流 / 500 服务器故障 / 503 服务器繁忙）。

依赖项目已有的 httpx（不引入 OpenAI SDK 等额外依赖）。
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

#: DeepSeek 开放平台 API 默认基址（/v1 后缀可选）
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

#: 默认模型：deepseek-flash（最便宜的deepseek）
DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"

#: 环境变量 → 配置字段映射（DEEPSEEK_* 优先，LLM_* 作为通用回退）
#: 注意书写顺序：靠后的键覆盖同名前面的键，因此 DEEPSEEK_API_KEY
#: 天然优先于 LLM_API_KEY。
_ENV_MAP = {
    "LLM_API_KEY": "api_key",
    "DEEPSEEK_API_KEY": "api_key",
    "DEEPSEEK_BASE_URL": "base_url",
    "DEEPSEEK_MODEL": "model",
    "DEEPSEEK_TIMEOUT": "timeout",
    "DEEPSEEK_MAX_RETRIES": "max_retries",
}


class DeepSeekConfig(ProviderConfig):
    """DeepSeek Provider 配置。"""

    base_url: str = Field(default=DEFAULT_DEEPSEEK_BASE_URL)
    model: str = Field(default=DEFAULT_DEEPSEEK_MODEL)

    @classmethod
    def from_env(cls, **overrides: Any) -> DeepSeekConfig:
        """从环境变量构造配置，``overrides`` 优先级最高。"""
        values: dict[str, Any] = {}
        for env_key, field in _ENV_MAP.items():
            value = os.environ.get(env_key)
            if value:
                values[field] = value
        values.update(overrides)
        return cls(**values)


class DeepSeekProvider(LLMProvider):
    """DeepSeek chat/completions 接入。

    用法::

        config = DeepSeekConfig.from_env()      # 读取 DEEPSEEK_API_KEY 等
        provider = DeepSeekProvider(config)
        async with provider:
            response = await provider.chat(ChatRequest(messages=[...]))
            print(response.content)

    :param config: DeepSeek 配置；
    :param client: 可选的已构造 ``httpx.AsyncClient``（测试注入
        ``httpx.MockTransport`` 用）；缺省时内部懒创建并在 ``close()`` 释放。
    """

    def __init__(
        self,
        config: Optional[DeepSeekConfig] = None,
        *,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        super().__init__(config or DeepSeekConfig())
        self._client = client
        self._owns_client = client is None

    @property
    def name(self) -> str:
        return "deepseek"

    # -- 单次调用 ------------------------------------------------------------

    async def _chat_once(self, request: ChatRequest) -> ChatResponse:
        """执行一次 DeepSeek chat/completions 调用（不含重试）。"""
        config = self.config
        if not isinstance(config, DeepSeekConfig):  # 防御：基类签名宽泛
            config = DeepSeekConfig.model_validate(config.model_dump())
        if not config.api_key:
            raise ProviderAuthError("未配置 DeepSeek API Key（DEEPSEEK_API_KEY）")

        payload = self._build_payload(request, config)
        client = await self._get_client()
        try:
            response = await client.post(
                "chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {config.api_key}"},
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"DeepSeek 请求超时: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderTimeoutError(f"DeepSeek 网络错误: {exc}") from exc

        if response.status_code >= 400:
            raise self._map_http_error(response)

        return self._parse_response(response)

    # -- 请求 / 响应 ----------------------------------------------------------

    @staticmethod
    def _build_payload(request: ChatRequest, config: DeepSeekConfig) -> dict[str, Any]:
        """组装 DeepSeek 请求体（None / 空值的可选字段不下发）。"""
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
        return payload

    @staticmethod
    def _map_http_error(response: httpx.Response) -> ProviderError:
        """把 HTTP 错误响应映射为 Provider 异常体系。

        402（余额不足）按不可重试的 API 错误处理——充钱才能解决，
        重试无意义；429 走限流退避；5xx 可重试。
        """
        code, message = _parse_error_body(response)
        status = response.status_code
        detail = f"DeepSeek API 错误: HTTP {status} code={code} message={message}"

        if status in (401, 403):
            return ProviderAuthError(detail, status=status)
        if status == 429:
            retry_after = _parse_retry_after(response)
            return ProviderRateLimitError(
                detail, status=status, retry_after=retry_after
            )
        if status >= 500:
            return ProviderAPIError(detail, status=status, retryable=True)
        # 4xx（参数错误、余额不足等）：重试无意义
        return ProviderAPIError(detail, status=status)

    @staticmethod
    def _parse_response(response: httpx.Response) -> ChatResponse:
        """解析 DeepSeek 成功响应。"""
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProviderAPIError(
                f"DeepSeek 响应不是合法 JSON: {response.text[:200]!r}"
            ) from exc

        choices = data.get("choices") or []
        if not choices:
            raise ProviderAPIError(f"DeepSeek 响应缺少 choices: {str(data)[:200]}")
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        if not content:
            # 推理型号（deepseek-reasoner）可能只返回 reasoning_content
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
