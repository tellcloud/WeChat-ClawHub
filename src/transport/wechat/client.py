"""微信客户端：封装 iLink Bot API 基础调用。

职责（对应 README 目录结构中的 ``client.py``）：

- 统一管理公共请求头（``iLink-App-Id`` / ``iLink-App-ClientVersion`` /
  ``X-WECHAT-UIN`` / ``Authorization`` / ``AuthorizationType``）；
- 封装全部 7 个端点的 HTTP 调用；
- 支持登录响应 ``baseurl`` 覆盖与跨机房 ``redirect_host`` 切换；
- 错误码映射：``ret=-14`` 会话过期、``ret=-2`` 参数错误、HTTP 4xx/5xx。

高层用法（扫码登录 / 长轮询 / 发消息）见 :meth:`WeChatClient.login`、
:meth:`WeChatClient.poll`、:meth:`WeChatClient.send_message`。
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Optional, Union
from uuid import uuid4

import httpx
from pydantic import BaseModel, ValidationError

from .models import (
    API_PREFIX,
    ERR_INVALID_PARAM,
    ERR_SESSION_EXPIRED,
    ILINK_APP_ID,
    ApiResponse,
    BaseInfo,
    Credentials,
    GetConfigRequest,
    GetConfigResponse,
    GetUpdatesRequest,
    GetUpdatesResponse,
    GetUploadUrlRequest,
    GetUploadUrlResponse,
    QrcodeResponse,
    QrcodeStatusResponse,
    SendMessageRequest,
    SendTypingRequest,
    WeChatConfig,
    encode_client_version,
    generate_uin,
)

logger = logging.getLogger(__name__)

#: 二维码就绪回调：(qrcode_id, qrcode_img_content)
QrcodeCallback = Callable[[str, str], Awaitable[None]]


# ---------------------------------------------------------------------------
# 异常体系
# ---------------------------------------------------------------------------


class WeChatTransportError(Exception):
    """transport/wechat 层异常基类。"""


class WeChatAPIError(WeChatTransportError):
    """iLink 业务错误（响应体带 ``ret != 0``）。"""

    def __init__(self, ret: int, errcode: Optional[int] = None, errmsg: Optional[str] = None) -> None:
        self.ret = ret
        self.errcode = errcode
        self.errmsg = errmsg
        super().__init__(f"iLink API 错误: ret={ret}, errcode={errcode}, errmsg={errmsg}")


class SessionExpiredError(WeChatAPIError):
    """会话过期（``errcode=-14``）。

    协议要求：暂停该账户所有 API 调用约 1 小时，或引导用户重新扫码登录。
    """

    def __init__(self) -> None:
        super().__init__(ret=ERR_SESSION_EXPIRED, errcode=ERR_SESSION_EXPIRED, errmsg="会话过期")


class InvalidParamError(WeChatAPIError):
    """参数错误（``ret=-2``）。"""


class WeChatHTTPError(WeChatTransportError):
    """HTTP 层错误（连接失败、4xx/5xx 等）。"""


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------


class WeChatClient:
    """iLink Bot API 底层 HTTP 客户端。

    用法::

        async with WeChatClient() as client:
            credentials = await client.login()          # 扫码登录
            async for msg in ...                        # 见 MessagePoller
            await client.send_message(user_id, "你好")  # 发送文本

    未登录时调用业务端点会抛出 :class:`WeChatTransportError`。
    """

    def __init__(
        self,
        config: Optional[WeChatConfig] = None,
        credentials: Optional[Credentials] = None,
        *,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.config = config or WeChatConfig.from_env()
        self.credentials = credentials
        self._client = http_client
        self._owns_client = http_client is None

    # -- 生命周期 ----------------------------------------------------------

    @property
    def http(self) -> httpx.AsyncClient:
        """懒加载内部 ``httpx.AsyncClient``。"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    self.config.api_timeout,
                    connect=self.config.connect_timeout,
                ),
                follow_redirects=True,
            )
        return self._client

    async def aclose(self) -> None:
        """关闭内部 HTTP 客户端（外部注入的不关闭）。"""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> WeChatClient:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    # -- 请求头 ------------------------------------------------------------

    def _common_headers(self) -> dict[str, str]:
        """所有请求携带的公共头。"""
        headers = {
            "iLink-App-Id": ILINK_APP_ID,
            "iLink-App-ClientVersion": str(encode_client_version(self.config.channel_version)),
        }
        if self.config.route_tag:
            headers["SKRouteTag"] = self.config.route_tag
        return headers

    def _authed_headers(self) -> dict[str, str]:
        """POST 业务请求头：追加 Bearer Token 与防重放 UIN。"""
        if self.credentials is None:
            raise WeChatTransportError("尚未登录：请先调用 login() 完成扫码认证")
        headers = self._common_headers()
        headers.update(
            {
                "Content-Type": "application/json",
                "AuthorizationType": "ilink_bot_token",
                "Authorization": f"Bearer {self.credentials.bearer_token}",
                "X-WECHAT-UIN": generate_uin(),
            }
        )
        return headers

    # -- 基址管理 ----------------------------------------------------------

    @property
    def base_url(self) -> str:
        """当前 API 基址：优先登录响应返回的 ``baseurl``。"""
        if self.credentials is not None and self.credentials.baseurl:
            return self.credentials.baseurl.rstrip("/")
        return self.config.base_url.rstrip("/")

    def redirect_to(self, host: str) -> None:
        """跨机房调度：切换到 ``https://{host}``（``scaned_but_redirect`` 场景）。"""
        new_base = f"https://{host}"
        logger.info("iLink 跨机房调度，切换基址到 %s", new_base)
        self.config.base_url = new_base
        if self.credentials is not None:
            self.credentials.baseurl = new_base

    # -- 底层请求 ----------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
        authed: bool = False,
        timeout: Optional[float] = None,
    ) -> httpx.Response:
        """统一 HTTP 调用，负责超时、错误映射与重试无关的原始返回。"""
        url = f"{self.base_url}{API_PREFIX}{path}"
        headers = self._authed_headers() if authed else self._common_headers()
        try:
            response = await self.http.request(
                method,
                url,
                params=params,
                json=json_body,
                headers=headers,
                timeout=timeout or self.config.api_timeout,
            )
        except httpx.TimeoutException as exc:
            raise WeChatHTTPError(f"请求超时: {method} {path}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise WeChatHTTPError(f"请求失败: {method} {path}: {exc}") from exc
        if response.status_code >= 400:
            raise WeChatHTTPError(
                f"HTTP {response.status_code}: {method} {path}: {response.text[:200]}"
            )
        return response

    @staticmethod
    def _check_api_ret(payload: dict[str, Any]) -> None:
        """检查响应体 ``ret`` / ``errcode``，映射为异常。"""
        ret = payload.get("ret", 0)
        errcode = payload.get("errcode")
        if ret == 0 and errcode in (None, 0):
            return
        if errcode == ERR_SESSION_EXPIRED:
            raise SessionExpiredError()
        if ret == ERR_INVALID_PARAM or errcode == ERR_INVALID_PARAM:
            raise InvalidParamError(ret=ret, errcode=errcode, errmsg=payload.get("errmsg"))
        raise WeChatAPIError(ret=ret, errcode=errcode, errmsg=payload.get("errmsg"))

    async def _get_json(
        self,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        authed: bool = False,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        response = await self._request(
            "GET", path, params=params, authed=authed, timeout=timeout
        )
        try:
            return response.json()
        except ValueError as exc:
            raise WeChatHTTPError(f"响应不是合法 JSON: {path}: {response.text[:200]}") from exc

    async def _post_json(
        self,
        path: str,
        body: Union[BaseModel, dict[str, Any]],
        *,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        payload = body.model_dump(mode="json", exclude_none=True) if isinstance(body, BaseModel) else dict(body)
        response = await self._request(
            "POST", path, json_body=payload, authed=True, timeout=timeout
        )
        # sendmessage 成功时返回 HTTP 200 无响应体
        if not response.content:
            return {"ret": 0}
        try:
            return response.json()
        except ValueError:
            return {"ret": 0, "raw": response.text[:200]}

    # -- 端点：扫码登录 -----------------------------------------------------

    async def get_bot_qrcode(self) -> QrcodeResponse:
        """``GET /ilink/bot/get_bot_qrcode?bot_type=3`` 获取登录二维码。"""
        payload = await self._get_json(
            "get_bot_qrcode",
            params={"bot_type": self.config.bot_type},
            timeout=self.config.api_timeout,
        )
        return QrcodeResponse.model_validate(payload)

    async def get_qrcode_status(self, qrcode: str) -> QrcodeStatusResponse:
        """``GET /ilink/bot/get_qrcode_status`` 轮询扫码状态。"""
        payload = await self._get_json(
            "get_qrcode_status",
            params={"qrcode": qrcode},
            timeout=self.config.longpoll_timeout,
        )
        return QrcodeStatusResponse.model_validate(payload)

    # -- 端点：消息 ---------------------------------------------------------

    async def get_updates(self, get_updates_buf: str = "") -> GetUpdatesResponse:
        """``POST /ilink/bot/getupdates`` 长轮询接收消息（服务端挂起约 35 秒）。"""
        request = GetUpdatesRequest(get_updates_buf=get_updates_buf)
        payload = await self._post_json("getupdates", request, timeout=self.config.longpoll_timeout)
        self._check_api_ret(payload)
        try:
            return GetUpdatesResponse.model_validate(payload)
        except ValidationError as exc:
            raise WeChatTransportError(f"getupdates 响应解析失败: {exc}") from exc

    async def send_raw_message(self, request: SendMessageRequest) -> None:
        """``POST /ilink/bot/sendmessage`` 发送原始消息载荷。"""
        payload = await self._post_json("sendmessage", request)
        self._check_api_ret(payload)

    # -- 端点：输入状态 -----------------------------------------------------

    async def get_config(
        self, ilink_user_id: str, context_token: str = ""
    ) -> GetConfigResponse:
        """``POST /ilink/bot/getconfig`` 获取 typing_ticket。"""
        request = GetConfigRequest(ilink_user_id=ilink_user_id, context_token=context_token)
        payload = await self._post_json("getconfig", request)
        self._check_api_ret(payload)
        return GetConfigResponse.model_validate(payload)

    async def send_typing(
        self,
        ilink_user_id: str,
        typing_ticket: str,
        status: int,
    ) -> None:
        """``POST /ilink/bot/sendtyping`` 发送“正在输入”状态（1=开始, 2=取消）。"""
        request = SendTypingRequest(
            ilink_user_id=ilink_user_id, typing_ticket=typing_ticket, status=status  # type: ignore[arg-type]
        )
        payload = await self._post_json("sendtyping", request)
        self._check_api_ret(payload)

    # -- 端点：媒体上传（预留） ---------------------------------------------

    async def get_upload_url(self, request: GetUploadUrlRequest) -> GetUploadUrlResponse:
        """``POST /ilink/bot/getuploadurl`` 获取 CDN 预签名上传地址。"""
        payload = await self._post_json("getuploadurl", request)
        self._check_api_ret(payload)
        return GetUploadUrlResponse.model_validate(payload)

    # -- 高层封装（技术方案中的核心接口） -----------------------------------

    async def login(
        self,
        on_qrcode: Optional[QrcodeCallback] = None,
        *,
        poll_interval: Optional[float] = None,
        max_refresh: Optional[int] = None,
    ) -> Credentials:
        """扫码登录完整流程（委托 :class:`~transport.wechat.auth.AuthManager`）。

        :param on_qrcode: 二维码就绪回调 ``(qrcode_id, qrcode_img_content)``，
            用于把二维码展示给用户；缺省时仅记录日志。
        :returns: 登录成功后的 :class:`Credentials`。
        """
        from .auth import AuthManager  # 延迟导入避免循环依赖

        manager = AuthManager(self)
        return await manager.login(
            on_qrcode=on_qrcode,
            poll_interval=poll_interval,
            max_refresh=max_refresh,
        )

    async def poll(self, handler: Callable[[Any], Awaitable[None]], *, once: bool = False) -> None:
        """启动长轮询消息循环（委托 :class:`~transport.wechat.poller.MessagePoller`）。

        :param handler: 收到消息后的异步回调，签名为 ``handler(WeixinMessage)``。
        :param once: 为 ``True`` 时只轮询一次后返回（调试用）。
        """
        from .poller import MessagePoller  # 延迟导入避免循环依赖

        poller = MessagePoller(self)
        await poller.run(handler, once=once)

    async def send_message(
        self,
        conversation_id: str,
        text: str,
        *,
        context_token: str = "",
        ilink_user_id: Optional[str] = None,
    ) -> None:
        """发送文本消息（委托 :class:`~transport.wechat.sender.MessageSender`）。

        :param conversation_id: 接收方用户 ID（``to_user_id``）。
        :param text: 文本内容。
        :param context_token: 入站消息的会话令牌；强烈建议回传，
            否则可能无法精确匹配对话窗口。
        :param ilink_user_id: typing 等场景使用的用户标识，默认取 conversation_id。
        """
        from .sender import MessageSender  # 延迟导入避免循环依赖

        sender = MessageSender(self)
        await sender.send_text(
            conversation_id,
            text,
            context_token=context_token,
            ilink_user_id=ilink_user_id or conversation_id,
        )


def new_client_id() -> str:
    """生成 ``sendmessage`` 用的 client_id（uuid4）。"""
    return str(uuid4())
