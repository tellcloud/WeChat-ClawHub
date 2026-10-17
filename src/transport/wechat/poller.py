"""消息轮询：通过长轮询持续接收微信消息。

职责（对应 README 目录结构中的 ``poller.py``）：

- 循环调用 ``POST /ilink/bot/getupdates``（服务端挂起约 35 秒）；
- ``get_updates_buf`` 游标持久化：首次为空串，之后必须回传上次响应的值，
  否则会重复收到消息；持久化到 TokenStore，重启后恢复；
- 失败退避：连续失败 1-2 次等待 2 秒，3 次以上等待 30 秒；
- 会话过期（``errcode=-14``）：触发 ``on_session_expired`` 回调
  （通常由上层执行重新登录），或按配置暂停约 1 小时；
- 支持优雅停止（``stop()`` / ``asyncio.CancelledError``）。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional

from .auth import TokenStore
from .client import SessionExpiredError, WeChatAPIError, WeChatClient, WeChatHTTPError
from .models import WeixinMessage

logger = logging.getLogger(__name__)

#: 消息处理器：``async def handler(message: WeixinMessage) -> None``
MessageHandler = Callable[[WeixinMessage], Awaitable[None]]

#: 会话过期处理器：``async def on_expired() -> None``（可重新登录后恢复轮询）
SessionExpiredHandler = Callable[[], Awaitable[None]]


class MessagePoller:
    """iLink 长轮询消息循环。

    用法::

        poller = MessagePoller(client)

        async def handle(msg: WeixinMessage) -> None:
            print(msg.from_user_id, "->", msg.text)

        await poller.run(handle)          # 阻塞运行，Ctrl-C / stop() 退出
        # 或：await poller.run(handle, once=True)   # 只轮询一次（调试）
    """

    def __init__(
        self,
        client: WeChatClient,
        store: Optional[TokenStore] = None,
        *,
        on_session_expired: Optional[SessionExpiredHandler] = None,
    ) -> None:
        self.client = client
        self.store = store or TokenStore(client.config.token_store_path)
        self.on_session_expired = on_session_expired
        self._cursor: str = ""
        self._stop_event = asyncio.Event()
        self._failure_count = 0

    # -- 生命周期 ------------------------------------------------------------

    def stop(self) -> None:
        """请求停止轮询循环（下一次循环迭代退出）。"""
        self._stop_event.set()

    @property
    def running(self) -> bool:
        """是否正在运行。"""
        return not self._stop_event.is_set()

    # -- 主循环 --------------------------------------------------------------

    async def run(self, handler: MessageHandler, *, once: bool = False) -> None:
        """启动长轮询循环。

        :param handler: 收到消息后的异步回调。
        :param once: 为 ``True`` 时只执行一次长轮询即返回（调试 / 测试用）。
        """
        # 恢复持久化游标，避免重启后重复收消息
        self._cursor = self.store.load_cursor()
        if self._cursor:
            logger.info("已恢复长轮询游标（长度 %d）", len(self._cursor))
        self._stop_event.clear()
        self._failure_count = 0

        while not self._stop_event.is_set():
            try:
                messages = await self.poll_once(handler)
                self._failure_count = 0
                if once and messages is not None:
                    return
            except SessionExpiredError:
                await self._handle_session_expired()
                if once:
                    raise
            except (WeChatAPIError, WeChatHTTPError) as exc:
                await self._handle_failure(exc)
                if once:
                    raise
            except asyncio.CancelledError:
                logger.info("长轮询已取消")
                raise

    async def poll_once(self, handler: MessageHandler) -> Optional[list[WeixinMessage]]:
        """执行一次长轮询：拉取消息 → 持久化游标 → 分发消息。

        :returns: 本次收到的消息列表（可能为空）；``once`` 场景用于判断是否有数据。
        """
        response = await self.client.get_updates(self._cursor)

        # 先持久化游标，再执行业务处理：即使处理中崩溃，重启后也不会漏消息
        if response.get_updates_buf:
            self._cursor = response.get_updates_buf
            self.store.save_cursor(self._cursor)

        for message in response.msgs:
            if message.message_type.value == 2:  # BOT 自己发的回声，跳过
                continue
            try:
                await handler(message)
            except Exception:
                logger.exception("消息处理失败: message_id=%s", message.message_id)

        return response.msgs

    # -- 异常处理 ------------------------------------------------------------

    async def _handle_failure(self, exc: Exception) -> None:
        """网络 / API 失败：按协议退避后继续。"""
        self._failure_count += 1
        if self._failure_count >= 3:
            delay = self.client.config.retry_max_delay
        else:
            delay = self.client.config.retry_base_delay
        logger.warning(
            "长轮询失败（第 %d 次），%.0f 秒后重试: %s",
            self._failure_count,
            delay,
            exc,
        )
        await self._sleep_or_stop(delay)

    async def _handle_session_expired(self) -> None:
        """会话过期：优先委托上层重新登录；否则按协议暂停约 1 小时。"""
        logger.error("会话已过期（errcode=-14），token 失效")
        # 清理失效凭据，防止后续请求继续使用
        from .auth import AuthManager

        AuthManager(self.client, self.store).logout()

        if self.on_session_expired is not None:
            logger.info("调用 on_session_expired 回调，等待重新登录")
            try:
                await self.on_session_expired()
            except Exception:
                logger.exception("on_session_expired 回调执行失败")
            return

        cooldown = self.client.config.session_expired_cooldown
        logger.warning("未配置重新登录回调，暂停轮询 %.0f 秒", cooldown)
        await self._sleep_or_stop(cooldown)

    async def _sleep_or_stop(self, delay: float) -> None:
        """可中断的睡眠：``stop()`` 被调用时立即返回。"""
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass
