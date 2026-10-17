"""消息发送：向微信发送文本和状态消息。

职责（对应 README 目录结构中的 ``sender.py``）：

- 文本发送：``POST /ilink/bot/sendmessage``，载荷
  ``message_type=2``（BOT）、``message_state=2``（FINISH）、
  ``item_list=[{"type": 1, "text_item": {"text": ...}}]``；
- **必须回传**入站消息的 ``context_token``，否则回复无法关联到正确的对话窗口；
- “正在输入”状态：``getconfig`` 获取 ``typing_ticket``（按用户缓存，
  TTL 约 24 小时）→ ``sendtyping``（``status=1`` 开始 / ``2`` 取消）；
- 长任务期间每 5 秒发送一次 keepalive（``status=1``），
  通过 :meth:`MessageSender.typing` 上下文管理器自动完成。
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from .client import WeChatClient, new_client_id
from .media import UploadedMedia
from .models import CDNMedia, MessageItemType, SendMessageRequest, SendMsg, TypingStatus

logger = logging.getLogger(__name__)


class MessageSender:
    """文本消息与输入状态发送器。

    用法::

        sender = MessageSender(client)

        # 简单发送
        await sender.send_text(user_id, "你好", context_token=msg.context_token)

        # 长任务期间显示“正在输入”
        async with sender.typing(user_id, context_token=msg.context_token):
            result = await long_running_task()
            await sender.send_text(user_id, result, context_token=msg.context_token)
    """

    def __init__(self, client: WeChatClient) -> None:
        self.client = client
        #: user_id -> (typing_ticket, 过期时间戳)
        self._ticket_cache: dict[str, tuple[str, float]] = {}

    # -- 文本发送 ------------------------------------------------------------

    async def send_text(
        self,
        to_user_id: str,
        text: str,
        *,
        context_token: str = "",
        ilink_user_id: Optional[str] = None,
    ) -> None:
        """发送一条文本消息。

        :param to_user_id: 接收方用户 ID（``xxx@im.wechat``）。
        :param text: 文本内容。
        :param context_token: 入站消息的 ``context_token``，回复时必须回传。
        :param ilink_user_id: 仅用于日志区分，默认同 ``to_user_id``。
        """
        if not text:
            logger.warning("忽略空文本消息: to=%s", to_user_id)
            return

        msg = SendMsg(
            to_user_id=to_user_id,
            client_id=new_client_id(),
            message_type=2,  # BOT
            message_state=2,  # FINISH
            context_token=context_token,
            item_list=[{"type": MessageItemType.TEXT, "text_item": {"text": text}}],
        )
        await self.client.send_raw_message(SendMessageRequest(msg=msg))
        logger.info(
            "消息已发送: to=%s(context=%s) len=%d",
            ilink_user_id or to_user_id,
            bool(context_token),
            len(text),
        )

    # -- 图片 / 媒体发送 ------------------------------------------------------

    async def send_image(
        self,
        to_user_id: str,
        uploaded: UploadedMedia,
        *,
        context_token: str = "",
    ) -> None:
        """发送一张图片。

        :param to_user_id: 接收方用户 ID。
        :param uploaded: 上传结果，由
            :meth:`transport.wechat.media.MediaUploader.upload_image` 返回。
            其中的 ``cipher_size`` 会写入 ``image_item.mid_size``——
            微信客户端依赖该字段拉取和渲染图片，缺失会导致图片打不开。
        :param context_token: 入站消息的 ``context_token``，回复时必须回传。
        """
        image_item: dict = {
            "media": uploaded.media.model_dump(exclude_defaults=True),
            "mid_size": uploaded.cipher_size,
        }
        await self._send_media_item(
            to_user_id,
            item={"type": MessageItemType.IMAGE, "image_item": image_item},
            context_token=context_token,
        )

    async def send_video(
        self,
        to_user_id: str,
        uploaded: UploadedMedia,
        *,
        context_token: str = "",
    ) -> None:
        """发送一段视频。

        :param to_user_id: 接收方用户 ID。
        :param uploaded: 上传结果，由
            :meth:`transport.wechat.media.MediaUploader.upload_video` 返回。
            其中的 ``cipher_size`` 会写入 ``video_item.video_size``。
        :param context_token: 入站消息的 ``context_token``，回复时必须回传。
        """
        video_item: dict = {
            "media": uploaded.media.model_dump(exclude_defaults=True),
            "video_size": uploaded.cipher_size,
        }
        await self._send_media_item(
            to_user_id,
            item={"type": MessageItemType.VIDEO, "video_item": video_item},
            context_token=context_token,
        )

    async def _send_media_item(
        self,
        to_user_id: str,
        *,
        item: dict,
        context_token: str = "",
    ) -> None:
        """发送单内容项消息（图片 / 语音 / 文件 / 视频通用）。"""
        msg = SendMsg(
            to_user_id=to_user_id,
            client_id=new_client_id(),
            message_type=2,  # BOT
            message_state=2,  # FINISH
            context_token=context_token,
            item_list=[item],
        )
        await self.client.send_raw_message(SendMessageRequest(msg=msg))
        logger.info(
            "媒体消息已发送: to=%s item_type=%s context=%s",
            to_user_id,
            item.get("type"),
            bool(context_token),
        )

    # -- “正在输入”状态 ------------------------------------------------------

    async def _get_typing_ticket(self, user_id: str, context_token: str = "") -> str:
        """获取（并缓存）用户的 ``typing_ticket``，TTL 约 24 小时。"""
        cached = self._ticket_cache.get(user_id)
        if cached is not None and cached[1] > time.monotonic():
            return cached[0]

        response = await self.client.get_config(user_id, context_token)
        if not response.typing_ticket:
            raise ValueError(f"getconfig 未返回 typing_ticket: user={user_id}")
        self._ticket_cache[user_id] = (
            response.typing_ticket,
            time.monotonic() + self.client.config.typing_ticket_ttl,
        )
        return response.typing_ticket

    async def start_typing(self, user_id: str, context_token: str = "") -> None:
        """开始显示“对方正在输入中”（``status=1``）。"""
        try:
            ticket = await self._get_typing_ticket(user_id, context_token)
            await self.client.send_typing(user_id, ticket, TypingStatus.TYPING)
        except Exception:
            # typing 是体验优化，失败不应影响主流程
            logger.exception("发送 typing 状态失败: user=%s", user_id)

    async def stop_typing(self, user_id: str, context_token: str = "") -> None:
        """取消“正在输入”状态（``status=2``）。"""
        try:
            ticket = await self._get_typing_ticket(user_id, context_token)
            await self.client.send_typing(user_id, ticket, TypingStatus.CANCEL)
        except Exception:
            logger.exception("取消 typing 状态失败: user=%s", user_id)

    @asynccontextmanager
    async def typing(
        self, user_id: str, context_token: str = ""
    ) -> AsyncIterator[None]:
        """长任务期间自动维持“正在输入”状态的上下文管理器。

        进入时发送 ``status=1``，期间每 5 秒 keepalive 一次，
        退出时发送 ``status=2`` 取消。用法::

            async with sender.typing(user_id, msg.context_token):
                await do_heavy_work()
        """
        await self.start_typing(user_id, context_token)
        keepalive: Optional[asyncio.Task] = None
        try:
            keepalive = asyncio.create_task(
                self._keepalive(user_id, context_token)
            )
            yield
        finally:
            if keepalive is not None:
                keepalive.cancel()
                try:
                    await keepalive
                except asyncio.CancelledError:
                    pass
            await self.stop_typing(user_id, context_token)

    async def _keepalive(self, user_id: str, context_token: str) -> None:
        """每 ``typing_keepalive_interval`` 秒重复发送 ``status=1``。"""
        while True:
            await asyncio.sleep(self.client.config.typing_keepalive_interval)
            try:
                ticket = await self._get_typing_ticket(user_id, context_token)
                await self.client.send_typing(user_id, ticket, TypingStatus.TYPING)
            except Exception:
                # keepalive 失败不致命，下一轮继续尝试
                logger.debug("typing keepalive 失败: user=%s", user_id)
