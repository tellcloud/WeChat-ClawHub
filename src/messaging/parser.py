"""消息解析：解析微信原始消息内容。

职责（对应 README 目录结构中的 ``parser.py``）：

- 输入：transport 层长轮询回调给出的
  :class:`~transport.wechat.models.WeixinMessage`；
- 输出：:class:`ParsedMessage`——保留全部原始字段的中间结构，
  尚未做业务标准化（空白折叠、@ 处理等由 normalizer 负责）；
- 覆盖内容类型：文本、图片、语音（含服务端 ASR 文本）、文件、视频、
  引用消息；未知 / 空消息标记为 ``UNKNOWN``；
- 标记 ``is_echo``（Bot 自己发出的回声），便于上层跳过。

解析是纯函数式的：不发起网络请求，不修改输入。
"""

from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, Field

from transport.wechat.models import (
    MessageItem,
    MessageItemType,
    MessageType,
    RefMessage,
    WeixinMessage,
)

from .models import Attachment, ContentKind, QuotedMessage

logger = logging.getLogger(__name__)

#: transport 内容类型 → messaging 内容类型
_KIND_MAP: dict[MessageItemType, ContentKind] = {
    MessageItemType.NONE: ContentKind.UNKNOWN,
    MessageItemType.TEXT: ContentKind.TEXT,
    MessageItemType.IMAGE: ContentKind.IMAGE,
    MessageItemType.VOICE: ContentKind.VOICE,
    MessageItemType.FILE: ContentKind.FILE,
    MessageItemType.VIDEO: ContentKind.VIDEO,
}


def to_content_kind(item_type: MessageItemType) -> ContentKind:
    """把 transport 的 ``MessageItemType`` 映射为 messaging 的内容类型。"""
    return _KIND_MAP.get(item_type, ContentKind.UNKNOWN)


class ParsedMessage(BaseModel):
    """解析后的原始消息（未标准化）。

    与 :class:`~messaging.models.Message` 的区别：保留 transport 层的
    原始字段名与顺序，文本未做折叠 / @ 处理，供 normalizer 转换。
    """

    message_id: str = ""
    seq: int = 0
    from_user_id: str = Field(default="", description="发送者完整 ID（xxx@im.wechat）")
    to_user_id: str = Field(default="", description="接收者 ID（xxx@im.bot）")
    client_id: str = ""
    session_id: str = ""
    group_id: str = ""
    context_token: str = ""
    create_time_ms: int = 0
    update_time_ms: int = 0

    is_echo: bool = Field(default=False, description="Bot 自己发出的消息（应跳过）")

    #: 文本类内容拼接结果（多个文本 item 按序拼接）
    text: str = ""
    #: 语音识别文本（取第一个非空的 voice_item.text）
    voice_text: str = ""
    #: 出现的全部内容类型（按 item 顺序，去重）
    kinds: list[ContentKind] = Field(default_factory=list)
    attachments: list[Attachment] = Field(default_factory=list)
    quoted: Optional[QuotedMessage] = None

    #: 原始消息引用（normalizer / 调试用；不参与序列化比较）
    raw: Optional[WeixinMessage] = Field(default=None, exclude=True, repr=False)


class MessageParser:
    """微信原始消息解析器。

    用法::

        parser = MessageParser()
        parsed = parser.parse(weixin_message)
        if parsed.is_echo:
            return  # Bot 回声，跳过
    """

    def parse(self, raw: WeixinMessage) -> ParsedMessage:
        """解析一条微信原始消息。"""
        parsed = ParsedMessage(
            message_id=str(raw.message_id),
            seq=raw.seq,
            from_user_id=raw.from_user_id,
            to_user_id=raw.to_user_id,
            client_id=raw.client_id,
            session_id=raw.session_id,
            group_id=raw.group_id,
            context_token=raw.context_token,
            create_time_ms=raw.create_time_ms,
            update_time_ms=raw.update_time_ms,
            is_echo=raw.message_type is MessageType.BOT,
            raw=raw,
        )

        text_parts: list[str] = []
        seen_kinds: list[ContentKind] = []

        for item in raw.item_list:
            kind = to_content_kind(item.type)
            if kind not in seen_kinds:
                seen_kinds.append(kind)
            self._parse_item(item, kind, text_parts, parsed)

        parsed.kinds = seen_kinds
        parsed.text = "".join(text_parts)

        if not parsed.text.strip() and not parsed.attachments:
            logger.debug(
                "消息无可解析内容: message_id=%s kinds=%s",
                parsed.message_id,
                [k.value for k in seen_kinds],
            )
        return parsed

    def parse_many(self, messages: list[WeixinMessage]) -> list[ParsedMessage]:
        """批量解析（保持顺序）。"""
        return [self.parse(message) for message in messages]

    # -- 内部 ----------------------------------------------------------------

    def _parse_item(
        self,
        item: MessageItem,
        kind: ContentKind,
        text_parts: list[str],
        parsed: ParsedMessage,
    ) -> None:
        """按内容类型解析单个 ``MessageItem``。"""
        # 引用消息优先解析：它通常挂在文本 / 媒体 item 上，
        # 不能放在按类型分发的分支之后（那些分支会提前 return）
        if item.ref_msg is not None and parsed.quoted is None:
            parsed.quoted = self._parse_ref(item.ref_msg)

        if kind is ContentKind.TEXT and item.text_item is not None:
            text_parts.append(item.text_item.text)
            return

        if kind is ContentKind.VOICE and item.voice_item is not None:
            voice = item.voice_item
            parsed.attachments.append(
                Attachment(
                    kind=ContentKind.VOICE,
                    duration_ms=voice.playtime,
                    asr_text=voice.text,
                    media=voice.media,
                )
            )
            # 语音消息以服务端识别文本作为可路由文本（取第一个非空）
            if not parsed.voice_text and voice.text.strip():
                parsed.voice_text = voice.text
            return

        if kind is ContentKind.IMAGE and item.image_item is not None:
            image = item.image_item
            parsed.attachments.append(
                Attachment(
                    kind=ContentKind.IMAGE,
                    size_bytes=image.hd_size or image.mid_size,
                    url=image.url,
                    media=image.media,
                )
            )
            return

        if kind is ContentKind.VIDEO and item.video_item is not None:
            video = item.video_item
            parsed.attachments.append(
                Attachment(
                    kind=ContentKind.VIDEO,
                    size_bytes=video.video_size,
                    duration_ms=video.play_length,
                    md5=video.video_md5,
                    media=video.media,
                )
            )
            return

        if kind is ContentKind.FILE and item.file_item is not None:
            file = item.file_item
            parsed.attachments.append(
                Attachment(
                    kind=ContentKind.FILE,
                    file_name=file.file_name,
                    size_bytes=_to_int(file.len),
                    md5=file.md5,
                    media=file.media,
                )
            )
            return

        # 引用消息在分支之前统一处理（见方法开头），这里只记录未知类型
        if kind is ContentKind.UNKNOWN:
            logger.debug("跳过暂不支持的内容类型: item_type=%s", int(item.type))

    def _parse_ref(self, ref: RefMessage) -> QuotedMessage:
        """解析引用消息（取被引用消息的文本）。"""
        text = ""
        kind = ContentKind.UNKNOWN
        ref_item = ref.message_item
        if ref_item is not None:
            kind = to_content_kind(ref_item.type)
            text = ref_item.text
            if not text and ref_item.voice_item is not None:
                text = ref_item.voice_item.text
        return QuotedMessage(
            message_id=ref_item.msg_id if ref_item else "",
            text=text,
            kind=kind,
        )


def _to_int(value: str, default: int = 0) -> int:
    """容错地把协议里的字符串数字转为 int。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
