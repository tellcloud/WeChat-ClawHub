"""GLM Provider：接入智谱 AI（GLM 系列模型）。

对应 README 目录结构中的 ``modules/ai/providers/glm.py``
（项目实际选用 GLM 作为首个后端，DeepSeek / Qwen 为后续补充的后端，
OpenAI / Anthropic 再按需补充）。

智谱 GLM v4 API（OpenAI 兼容风格）：

- 端点：``POST {base_url}/chat/completions``，默认基址
  ``https://open.bigmodel.cn/api/paas/v4``；
- 认证：``Authorization: Bearer <GLM_API_KEY>``（形如 ``id.secret``）；
- 请求：``{"model": ..., "messages": [{"role": ..., "content": ...}],
  "temperature": ..., "max_tokens": ..., "stream": false}``；
- 响应：``choices[0].message.content`` + ``usage`` + ``finish_reason``；
- 错误：HTTP 非 2xx，响应体 ``{"error": {"code": ..., "message": ...}}``
  （旧版为 ``{"code": ..., "msg": ...}``，两种格式都兼容）。

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

#: 智谱开放平台 v4 API 默认基址
DEFAULT_GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

#: 默认模型：glm-4-flash（智谱免费档，适合个人助理场景）
DEFAULT_GLM_MODEL = "glm-4-flash"

#: 环境变量 → 配置字段映射（GLM_* 优先，LLM_* 作为通用回退）
_ENV_MAP = {
    "GLM_API_KEY": "api_key",
    "LLM_API_KEY": "api_key",
    "GLM_BASE_URL": "base_url",
    "GLM_MODEL": "model",
    "GLM_TIMEOUT": "timeout",
    "GLM_MAX_RETRIES": "max_retries",
}


class GLMConfig(ProviderConfig):
    """GLM Provider 配置。"""

    base_url: str = Field(default=DEFAULT_GLM_BASE_URL)
    model: str = Field(default=DEFAULT_GLM_MODEL)

    @classmethod
    def from_env(cls, **overrides: Any) -> GLMConfig:
        """从环境变量构造配置，``overrides`` 优先级最高。"""
        values: dict[str, Any] = {}
        for env_key, field in _ENV_MAP.items():
            value = os.environ.get(env_key)
            if value:
                values[field] = value
        # GLM_API_KEY 优先于 LLM_API_KEY：先写入的被后者覆盖时换序处理
        if os.environ.get("GLM_API_KEY") and os.environ.get("LLM_API_KEY"):
            values["api_key"] = os.environ["GLM_API_KEY"]
        values.update(overrides)
        return cls(**values)


class GLMProvider(LLMProvider):
    """智谱 GLM v4 chat/completions 接入。

    用法::

        config = GLMConfig.from_env()          # 读取 GLM_API_KEY 等
        provider = GLMProvider(config)
        async with provider:
            response = await provider.chat(ChatRequest(messages=[...]))
            print(response.content)

    :param config: GLM 配置；
    :param client: 可选的已构造 ``httpx.AsyncClient``（测试注入
        ``httpx.MockTransport`` 用）；缺省时内部懒创建并在 ``close()`` 释放。
    """

    def __init__(
        self,
        config: Optional[GLMConfig] = None,
        *,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        super().__init__(config or GLMConfig())
        self._client = client
        self._owns_client = client is None

    @property
    def name(self) -> str:
        return "glm"

    # -- 单次调用 ------------------------------------------------------------

    async def _chat_once(self, request: ChatRequest) -> ChatResponse:
        """执行一次 GLM chat/completions 调用（不含重试）。"""
        config = self.config
        if not isinstance(config, GLMConfig):  # 防御：基类签名宽泛
            config = GLMConfig.model_validate(config.model_dump())
        if not config.api_key:
            raise ProviderAuthError("未配置 GLM API Key（GLM_API_KEY）")

        payload = self._build_payload(request, config)
        client = await self._get_client()
        try:
            response = await client.post(
                "chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {config.api_key}"},
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"GLM 请求超时: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderTimeoutError(f"GLM 网络错误: {exc}") from exc

        if response.status_code >= 400:
            raise self._map_http_error(response)

        return self._parse_response(response)

    # -- 请求 / 响应 ----------------------------------------------------------

    @staticmethod
    def _build_payload(request: ChatRequest, config: GLMConfig) -> dict[str, Any]:
        """组装 GLM 请求体（None / 空值的可选字段不下发）。"""
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
        """把 HTTP 错误响应映射为 Provider 异常体系。"""
        code, message = _parse_error_body(response)
        status = response.status_code
        detail = f"GLM API 错误: HTTP {status} code={code} message={message}"

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
        """解析 GLM 成功响应。"""
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProviderAPIError(
                f"GLM 响应不是合法 JSON: {response.text[:200]!r}"
            ) from exc

        choices = data.get("choices") or []
        if not choices:
            raise ProviderAPIError(f"GLM 响应缺少 choices: {str(data)[:200]}")
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        if not content:
            # 某些模型（如带 reasoning 的型号）可能只返回 reasoning_content
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
