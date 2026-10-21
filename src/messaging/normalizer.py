"""消息标准化：将不同类型消息转换为统一格式。

职责（对应 README 目录结构中的 ``normalizer.py``）：

- 输入：:class:`~messaging.parser.ParsedMessage`（保留原始字段）；
- 输出：:class:`~messaging.models.Message`（系统内部统一结构）；
- 标准化规则：

  - 会话类型：``group_id`` 非空 → 群聊，否则私聊；
  - 会话键：群聊 ``group:<group_id>`` / 私聊 ``user:<sender_id>``；
  - 发送者键：取 ``@`` 前缀部分，作为跨消息稳定的用户标识；
  - 文本：全角空格 / 零宽字符归一 → 折叠连续空白 → 去首尾空白；
  - 群聊 @ 提及：提取昵称列表，并剥离文本开头的 @ 前缀；
  - 语音消息：无文本时以服务端识别文本（ASR）作为 ``text``；
  - 时间戳：协议 ``create_time_ms`` 为 0 时回落到 ``received_at``；
  - 内容类型：有文本 → ``TEXT``；否则取第一个附件类型；再否则 ``UNKNOWN``。
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel

from .models import Attachment, ChatType, ContentKind, Message
from .parser import ParsedMessage

#: 群聊 @ 提及：``@昵称`` 后接普通空格或四分之一 em 空格（U+2005）
_MENTION_RE = re.compile(r"@([^\s@\u2005]+)")

#: 文本开头的连续 @ 提及 token（含尾部空格）
_LEADING_MENTION_RE = re.compile(r"^(?:@[^\s@\u2005]+[\s\u2005]*)+")

#: 需要归一为普通空白的字符：全角空格、零宽不连字、四分之一 em 空格
_SPACE_CHARS = "\u3000\u200b\u2005"

#: 连续空白（含换行 / 制表）
_WHITESPACE_RUN_RE = re.compile(r"\s+")


class NormalizerConfig(BaseModel):
    """标准化行为配置。"""

    #: 是否折叠文本中的连续空白（保留可读性，避免微信客户端的诡异空格）
    collapse_whitespace: bool = True
    #: 是否剥离群聊文本开头的 @ 提及（AI 看到的文本更干净）
    strip_leading_mentions: bool = True
    #: 语音消息是否允许把 ASR 文本提升为消息文本
    promote_voice_text: bool = True
    #: 协议时间戳缺失时是否使用系统接收时间兜底
    fallback_received_at: bool = True


class MessageNormalizer:
    """把解析后的原始消息转换为统一消息结构。

    用法::

        normalizer = MessageNormalizer()
        message = normalizer.normalize(parsed)
    """

    def __init__(self, config: Optional[NormalizerConfig] = None) -> None:
        self.config = config or NormalizerConfig()

    def normalize(self, parsed: ParsedMessage) -> Message:
        """标准化一条已解析的消息。"""
        chat_type = ChatType.GROUP if parsed.group_id else ChatType.PRIVATE

        text = self._normalize_text(parsed.text)
        mentions = _MENTION_RE.findall(text)
        if self.config.strip_leading_mentions:
            text = _LEADING_MENTION_RE.sub("", text).strip()

        # 语音消息：无文本时以 ASR 结果作为文本
        asr_text = parsed.voice_text
        if not text and self.config.promote_voice_text and asr_text.strip():
            text = self._normalize_text(asr_text)

        received_at = datetime.now(timezone.utc)
        created_at = _ms_to_datetime(parsed.create_time_ms)
        if created_at is None and self.config.fallback_received_at:
            # 协议时间戳缺失时用系统接收时间兜底
            created_at = received_at

        message = Message(
            message_id=parsed.message_id,
            seq=parsed.seq,
            sender_id=parsed.from_user_id,
            sender_key=_sender_key(parsed.from_user_id),
            bot_id=parsed.to_user_id,
            chat_type=chat_type,
            group_id=parsed.group_id,
            session_key=self._session_key(
                chat_type, parsed.group_id, parsed.from_user_id
            ),
            kind=self._resolve_kind(text, parsed.attachments),
            text=text,
            asr_text=asr_text,
            mentions=mentions,
            attachments=list(parsed.attachments),
            quoted=parsed.quoted,
            context_token=parsed.context_token,
            created_at=created_at,
            received_at=received_at,
        )
        return message

    # -- 内部 ----------------------------------------------------------------

    @staticmethod
    def _normalize_text(text: str) -> str:
        """空白与特殊字符归一。"""
        if not text:
            return ""
        for char in _SPACE_CHARS:
            text = text.replace(char, " ")
        if _WHITESPACE_RUN_RE.search(text):
            text = _WHITESPACE_RUN_RE.sub(" ", text)
        return text.strip()

    @staticmethod
    def _session_key(chat_type: ChatType, group_id: str, sender_id: str) -> str:
        """生成 session 层使用的会话键。"""
        if chat_type is ChatType.GROUP:
            return f"group:{group_id}"
        return f"user:{sender_id}"

    @staticmethod
    def _resolve_kind(text: str, attachments: list[Attachment]) -> ContentKind:
        """判定消息的主内容类型。"""
        if text.strip():
            return ContentKind.TEXT
        for attachment in attachments:
            if attachment.kind is not ContentKind.UNKNOWN:
                return attachment.kind
        return ContentKind.UNKNOWN


def _sender_key(sender_id: str) -> str:
    """``xxx@im.wechat`` → ``xxx``（空串原样返回）。"""
    return sender_id.split("@", 1)[0] if sender_id else ""


def _ms_to_datetime(ms: int) -> Optional[datetime]:
    """Unix 毫秒 → ``datetime``；0 / 负值视为缺失。"""
    if ms <= 0:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
