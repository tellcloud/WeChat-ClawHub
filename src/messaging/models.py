"""消息处理层：系统内部统一消息模型。

本模块定义 messaging 层的核心数据结构，与 transport 层（微信协议模型）
解耦：parser / normalizer / router 都围绕这里的 :class:`Message` 工作，
新增接入渠道（不限于微信）时只需提供对应的 parser。

与 transport 层的关系：

- :class:`~transport.wechat.models.WeixinMessage` 是微信协议原始结构；
- :class:`Message` 是系统内部统一结构，供路由与上层模块使用；
- 媒体引用（CDN）以 :class:`~transport.wechat.models.CDNMedia` 原样透传，
  避免 messaging 层重复定义协议细节。
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field

from transport.wechat.models import CDNMedia


class ContentKind(StrEnum):
    """消息内容类型（messaging 层统一视角）。

    与 transport 的 ``MessageItemType`` 一一对应，但增加 ``UNKNOWN``
    用于表示无法归类的内容（空消息、暂不支持的协议类型等）。
    """

    TEXT = "text"
    IMAGE = "image"
    VOICE = "voice"
    FILE = "file"
    VIDEO = "video"
    UNKNOWN = "unknown"


class ChatType(StrEnum):
    """会话类型：私聊 / 群聊。"""

    PRIVATE = "private"
    GROUP = "group"


class Direction(StrEnum):
    """消息方向（为多账号 / 主动消息预留）。"""

    INBOUND = "inbound"  #: 用户 → Bot
    OUTBOUND = "outbound"  #: Bot → 用户


class Attachment(BaseModel):
    """非文本内容附件（图片 / 语音 / 文件 / 视频）。

    ``media`` 透传 transport 的 CDN 引用，下载 / 回传由媒体管线处理，
    messaging 层只负责携带信息，不做网络操作。
    """

    kind: ContentKind = ContentKind.UNKNOWN
    file_name: str = Field(default="", description="文件名（文件类消息）")
    size_bytes: int = Field(default=0, description="字节大小（0 表示未知）")
    duration_ms: int = Field(default=0, description="播放时长（毫秒，音视频）")
    md5: str = ""
    url: str = Field(default="", description="服务端直链（v2.1.1+ 可能提供）")
    asr_text: str = Field(default="", description="语音识别文本（语音类消息）")
    media: Optional[CDNMedia] = Field(default=None, description="CDN 媒体引用（透传）")


class QuotedMessage(BaseModel):
    """被引用的消息（用户回复 / 引用场景）。"""

    message_id: str = ""
    text: str = ""
    kind: ContentKind = ContentKind.UNKNOWN


class Message(BaseModel):
    """系统内部统一消息结构。

    由 normalizer 从 transport 原始消息转换而来，是 router 与上层
    模块（Script / AI / Agent）的唯一输入。

    字段说明：

    - ``sender_id``：完整用户 ID（``xxx@im.wechat``），回复时必须使用；
    - ``sender_key``：``@`` 前缀部分，跨渠道稳定的用户键，用于会话查找；
    - ``session_key``：会话键，群聊为 ``group:<group_id>``，私聊为
      ``user:<sender_id>``，session 层以此为键维护上下文；
    - ``text``：标准化后的纯文本（语音消息为识别结果）；
    - ``mentions``：文本中 @ 到的昵称列表（群聊场景）。
    """

    message_id: str = ""
    seq: int = 0
    direction: Direction = Direction.INBOUND

    sender_id: str = Field(default="", description="发送者完整 ID（xxx@im.wechat）")
    sender_key: str = Field(default="", description="发送者稳定键（@ 前缀部分）")
    bot_id: str = Field(default="", description="Bot 标识（xxx@im.bot）")

    chat_type: ChatType = ChatType.PRIVATE
    group_id: str = ""
    session_key: str = Field(default="", description="会话键：group:<id> / user:<id>")

    kind: ContentKind = ContentKind.UNKNOWN
    text: str = ""
    asr_text: str = Field(default="", description="语音识别文本（与 text 可能重复）")
    mentions: list[str] = Field(default_factory=list)
    attachments: list[Attachment] = Field(default_factory=list)
    quoted: Optional[QuotedMessage] = None

    context_token: str = Field(
        default="",
        description="会话令牌，回复时必须原样回传（透传自微信协议）",
    )
    created_at: Optional[datetime] = Field(
        default=None, description="消息创建时间（协议时间戳，缺失为 None）"
    )
    received_at: datetime = Field(
        default_factory=datetime.now, description="系统接收时间（兜底用）"
    )

    @property
    def is_group(self) -> bool:
        """是否群聊消息。"""
        return self.chat_type is ChatType.GROUP

    @property
    def has_text(self) -> bool:
        """是否携带可用文本。"""
        return bool(self.text.strip())
