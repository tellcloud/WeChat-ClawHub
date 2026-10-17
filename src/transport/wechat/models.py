"""微信 ClawBot / iLink Bot 协议模型。

基于腾讯官方 iLink Bot API（``https://ilinkai.weixin.qq.com``）整理，
协议要点：

- 认证：扫码登录获取 ``bot_token``（Bearer Token）；
- 收消息：``POST /ilink/bot/getupdates`` 长轮询（服务端挂起约 35 秒），
  通过 ``get_updates_buf`` 游标推进，防重复；
- 发消息：``POST /ilink/bot/sendmessage``，必须回传入站消息的
  ``context_token`` 才能关联到正确的对话窗口；
- 输入状态：``getconfig`` 获取 ``typing_ticket`` 后 ``sendtyping``。

本模块只定义数据结构与配置，不发起任何网络请求。
"""

from __future__ import annotations

import base64
import os
import random
from datetime import datetime, timezone
from enum import IntEnum, StrEnum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 协议常量
# ---------------------------------------------------------------------------

#: iLink Bot API 默认基址（登录响应里的 ``baseurl`` 可能覆盖它）
DEFAULT_BASE_URL = "https://ilinkai.weixin.qq.com"

#: 腾讯 CDN 基址（媒体上传 / 下载）
DEFAULT_CDN_BASE_URL = "https://novac2c.cdn.weixin.qq.com/c2c"

#: 端点统一前缀
API_PREFIX = "/ilink/bot/"

#: 请求头 iLink-App-Id 固定值
ILINK_APP_ID = "bot"

#: 登录二维码 bot_type 硬编码值（官方协议未文档化）
DEFAULT_BOT_TYPE = 3

#: 客户端版本号，对应 @tencent-weixin/openclaw-weixin v2.1.1
DEFAULT_CHANNEL_VERSION = "2.1.1"

#: getupdates 长轮询服务端挂起时长（毫秒）
LONGPOLL_TIMEOUT_MS = 35_000

#: 错误码：会话过期（需暂停该账户 API 调用约 1 小时或重新登录）
ERR_SESSION_EXPIRED = -14

#: 错误码：参数错误
ERR_INVALID_PARAM = -2

#: 二维码有效期约 5 分钟，过期后最多刷新次数
QR_MAX_REFRESH = 3


# ---------------------------------------------------------------------------
# 枚举
# ---------------------------------------------------------------------------


class MessageType(IntEnum):
    """消息方向。"""

    NONE = 0
    USER = 1  #: 用户发送给 Bot 的消息
    BOT = 2  #: Bot 发送的消息


class MessageState(IntEnum):
    """消息状态。"""

    NEW = 0
    GENERATING = 1  #: 流式生成中
    FINISH = 2  #: 已完成


class MessageItemType(IntEnum):
    """消息内容类型。"""

    NONE = 0
    TEXT = 1
    IMAGE = 2
    VOICE = 3
    FILE = 4
    VIDEO = 5


class QRStatus(StrEnum):
    """扫码登录状态机取值。

    ``wait`` → ``scaned`` → ``confirmed``；
    ``scaned_but_redirect`` 表示已扫码但需要切换到 ``redirect_host`` 继续；
    ``expired`` 表示二维码过期，需要刷新（最多 3 次）。
    """

    WAIT = "wait"
    SCANED = "scaned"
    SCANED_BUT_REDIRECT = "scaned_but_redirect"
    EXPIRED = "expired"
    CONFIRMED = "confirmed"


class TypingStatus(IntEnum):
    """“正在输入”状态。"""

    TYPING = 1  #: 开始输入
    CANCEL = 2  #: 取消输入


class UploadMediaType(IntEnum):
    """CDN 上传媒体类型。"""

    IMAGE = 1
    VIDEO = 2
    FILE = 3
    VOICE = 4


# ---------------------------------------------------------------------------
# 公共工具
# ---------------------------------------------------------------------------


def encode_client_version(version: str) -> int:
    """把 ``MM.NN.PP`` 版本号编码为 uint32，格式 ``0x00MMNNPP``。

    例如 ``"0.3.0"`` → ``768``，``"2.1.1"`` → ``131329``。
    对应请求头 ``iLink-App-ClientVersion``。
    """
    parts = version.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"非法的 channel_version: {version!r}，期望 MM.NN.PP 格式")
    major, minor, patch = (int(p) for p in parts)
    return (major << 16) | (minor << 8) | patch


def generate_uin() -> str:
    """生成 ``X-WECHAT-UIN`` 请求头值。

    协议要求每次请求随机 uint32 → 十进制字符串 → base64，用于防重放攻击。
    """
    value = random.getrandbits(32)
    return base64.b64encode(str(value).encode("utf-8")).decode("ascii")


def now_ms() -> int:
    """当前 Unix 毫秒时间戳。"""
    return int(datetime.now(timezone.utc).timestamp() * 1000)


# ---------------------------------------------------------------------------
# 凭据
# ---------------------------------------------------------------------------


class Credentials(BaseModel):
    """扫码登录成功后返回的认证凭据。"""

    ilink_bot_id: str = Field(description="Bot 标识符，格式 xxx@im.bot")
    bot_token: str = Field(description="Bearer token，后续所有 API 调用携带")
    ilink_user_id: str = Field(description="用户标识符")
    baseurl: str = Field(
        default=DEFAULT_BASE_URL,
        description="API 基址，可能与默认值不同，必须使用返回值",
    )

    @property
    def bearer_token(self) -> str:
        """去掉可能存在的 ``Bearer `` 前缀后的纯 token。"""
        token = self.bot_token.strip()
        if token.lower().startswith("bearer "):
            token = token[7:]
        return token


# ---------------------------------------------------------------------------
# 消息内容模型
# ---------------------------------------------------------------------------


class CDNMedia(BaseModel):
    """CDN 媒体引用（图片 / 语音 / 文件 / 视频共用）。"""

    encrypt_query_param: str = Field(default="", description="CDN 下载 / 上传参数")
    aes_key: str = Field(default="", description="base64 编码的 AES-128 密钥")
    encrypt_type: int = Field(default=0, description="加密方式：0=fileid, 1=packed")
    full_url: str = Field(
        default="",
        description="服务端直接返回的完整下载 URL（v2.1.1+，优先使用）",
    )


class TextItem(BaseModel):
    """文本内容（``MessageItemType.TEXT``）。"""

    text: str = ""


class ImageItem(BaseModel):
    """图片内容（``MessageItemType.IMAGE``）。"""

    media: Optional[CDNMedia] = None
    thumb_media: Optional[CDNMedia] = None
    aeskey: str = Field(default="", description="hex 格式 AES 密钥")
    url: str = ""
    mid_size: int = 0
    thumb_size: int = 0
    thumb_height: int = 0
    thumb_width: int = 0
    hd_size: int = 0


class VoiceItem(BaseModel):
    """语音内容（``MessageItemType.VOICE``）。

    ``text`` 为微信服务端语音识别结果，可能为空。
    """

    media: Optional[CDNMedia] = None
    encode_type: int = 0
    bits_per_sample: int = 0
    sample_rate: int = 0
    playtime: int = Field(default=0, description="播放时长（毫秒）")
    text: str = ""


class FileItem(BaseModel):
    """文件内容（``MessageItemType.FILE``）。"""

    media: Optional[CDNMedia] = None
    file_name: str = ""
    md5: str = Field(default="", description="明文 MD5")
    len: str = Field(default="", description="文件大小（字节）")


class VideoItem(BaseModel):
    """视频内容（``MessageItemType.VIDEO``）。"""

    media: Optional[CDNMedia] = None
    thumb_media: Optional[CDNMedia] = None
    video_size: int = 0
    play_length: int = Field(default=0, description="播放时长（毫秒）")
    video_md5: str = ""
    thumb_size: int = 0
    thumb_height: int = 0
    thumb_width: int = 0


class RefMessage(BaseModel):
    """引用消息（回复场景）。"""

    message_item: Optional[MessageItem] = None  # noqa: F821 - 前向引用
    title: str = ""


class MessageItem(BaseModel):
    """消息内容项，``type`` 决定哪个 ``*_item`` 字段有效。"""

    type: MessageItemType = MessageItemType.NONE
    create_time_ms: int = 0
    update_time_ms: int = 0
    is_completed: bool = True
    msg_id: str = ""
    ref_msg: Optional[RefMessage] = None
    text_item: Optional[TextItem] = None
    image_item: Optional[ImageItem] = None
    voice_item: Optional[VoiceItem] = None
    file_item: Optional[FileItem] = None
    video_item: Optional[VideoItem] = None

    @property
    def text(self) -> str:
        """便捷访问：``type=1`` 时返回文本内容，否则返回空串。"""
        if self.type == MessageItemType.TEXT and self.text_item is not None:
            return self.text_item.text
        return ""


MessageItem.model_rebuild()


class WeixinMessage(BaseModel):
    """iLink 入站 / 出站消息（WeixinMessage）。"""

    seq: int = 0
    message_id: int = 0
    from_user_id: str = Field(default="", description="发送者 ID，格式 xxx@im.wechat")
    to_user_id: str = Field(default="", description="接收者 ID，格式 xxx@im.bot")
    client_id: str = ""
    create_time_ms: int = 0
    update_time_ms: int = 0
    delete_time_ms: int = 0
    session_id: str = ""
    group_id: str = Field(default="", description="群聊场景非空")
    message_type: MessageType = MessageType.NONE
    message_state: MessageState = MessageState.NEW
    item_list: list[MessageItem] = Field(default_factory=list)
    context_token: str = Field(
        default="",
        description="会话令牌，回复时必须原样回传以关联正确的对话窗口",
    )

    @property
    def text(self) -> str:
        """拼接所有文本内容项的文本。"""
        return "".join(item.text for item in self.item_list)

    @property
    def is_group(self) -> bool:
        """是否群聊消息。"""
        return bool(self.group_id)


# ---------------------------------------------------------------------------
# 请求 / 响应模型
# ---------------------------------------------------------------------------


class BaseInfo(BaseModel):
    """POST 请求体公共字段。"""

    channel_version: str = DEFAULT_CHANNEL_VERSION


class QrcodeResponse(BaseModel):
    """``GET /ilink/bot/get_bot_qrcode`` 响应。"""

    qrcode: str = Field(description="会话标识符，用于轮询扫码状态")
    qrcode_img_content: str = Field(
        default="",
        description="二维码图片：data:image/png;base64,... 或图片 URL",
    )


class QrcodeStatusResponse(BaseModel):
    """``GET /ilink/bot/get_qrcode_status`` 响应。"""

    status: str = QRStatus.WAIT
    ilink_bot_id: str = ""
    bot_token: str = ""
    baseurl: str = ""
    ilink_user_id: str = ""
    redirect_host: str = Field(
        default="",
        description="status=scaned_but_redirect 时的跨机房调度主机",
    )

    def to_credentials(self) -> Optional[Credentials]:
        """``confirmed`` 时构造凭据，否则返回 ``None``。"""
        if self.status != QRStatus.CONFIRMED:
            return None
        return Credentials(
            ilink_bot_id=self.ilink_bot_id,
            bot_token=self.bot_token,
            ilink_user_id=self.ilink_user_id,
            baseurl=self.baseurl or DEFAULT_BASE_URL,
        )


class GetUpdatesRequest(BaseModel):
    """``POST /ilink/bot/getupdates`` 请求。"""

    get_updates_buf: str = Field(
        default="",
        description="同步游标：首次为空串，之后回传上次响应的值；不回传或回传旧值会导致重复收消息",
    )
    base_info: BaseInfo = Field(default_factory=BaseInfo)


class GetUpdatesResponse(BaseModel):
    """``POST /ilink/bot/getupdates`` 响应。"""

    ret: int = 0
    errcode: Optional[int] = None
    errmsg: Optional[str] = None
    msgs: list[WeixinMessage] = Field(default_factory=list)
    get_updates_buf: str = Field(default="", description="新的同步游标")
    longpolling_timeout_ms: int = LONGPOLL_TIMEOUT_MS


class TextMessageItem(BaseModel):
    """发送用文本内容项。"""

    type: Literal[1] = 1
    text_item: TextItem


class SendMsg(BaseModel):
    """``sendmessage`` 请求的 ``msg`` 字段。"""

    to_user_id: str
    client_id: str = Field(description="自定义客户端 ID，建议 uuid")
    message_type: MessageType = MessageType.BOT
    message_state: MessageState = MessageState.FINISH
    context_token: str = Field(
        default="",
        description="从 getUpdates 收到的 token；缺失时服务端使用最近活跃会话",
    )
    item_list: list[dict[str, Any]] = Field(
        default_factory=list,
        description='内容列表，如 [{"type": 1, "text_item": {"text": "..."}}]',
    )


class SendMessageRequest(BaseModel):
    """``POST /ilink/bot/sendmessage`` 请求。"""

    msg: SendMsg
    base_info: BaseInfo = Field(default_factory=BaseInfo)


class GetConfigRequest(BaseModel):
    """``POST /ilink/bot/getconfig`` 请求。"""

    ilink_user_id: str
    context_token: str = ""
    base_info: BaseInfo = Field(default_factory=BaseInfo)


class GetConfigResponse(BaseModel):
    """``POST /ilink/bot/getconfig`` 响应。"""

    ret: int = 0
    errmsg: Optional[str] = None
    typing_ticket: str = Field(
        default="", description="按用户获取，建议缓存（TTL 24 小时）"
    )


class SendTypingRequest(BaseModel):
    """``POST /ilink/bot/sendtyping`` 请求。"""

    ilink_user_id: str
    typing_ticket: str
    status: TypingStatus = TypingStatus.TYPING
    base_info: BaseInfo = Field(default_factory=BaseInfo)


class GetUploadUrlRequest(BaseModel):
    """``POST /ilink/bot/getuploadurl`` 请求（媒体上传预留）。"""

    filekey: str = Field(description="16 字节随机 hex")
    media_type: UploadMediaType
    to_user_id: str
    rawsize: int = Field(description="原文件字节数")
    rawfilemd5: str = Field(description="明文 MD5")
    filesize: int = Field(description="AES 加密后字节数")
    no_need_thumb: bool = True
    aeskey: str = Field(description="16 字节 AES 密钥的 hex 表示")
    base_info: BaseInfo = Field(default_factory=BaseInfo)


class GetUploadUrlResponse(BaseModel):
    """``POST /ilink/bot/getuploadurl`` 响应。"""

    upload_param: str = ""
    thumb_upload_param: str = ""
    upload_full_url: str = Field(default="", description="v2.1.1+ 完整上传 URL")


class ApiResponse(BaseModel):
    """通用 API 响应，至少包含 ``ret``。"""

    ret: int = 0
    errcode: Optional[int] = None
    errmsg: Optional[str] = None


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


class WeChatConfig(BaseModel):
    """transport/wechat 运行配置。

    优先级：显式传参 > 环境变量 > 默认值。
    支持的环境变量：

    - ``WECHAT_BASE_URL`` / ``WECHAT_CDN_BASE_URL``
    - ``WECHAT_CHANNEL_VERSION``
    - ``WECHAT_TOKEN_STORE``（凭据 / 游标持久化文件路径）
    - ``WECHAT_ROUTE_TAG``（可选 SKRouteTag）
    """

    base_url: str = DEFAULT_BASE_URL
    cdn_base_url: str = DEFAULT_CDN_BASE_URL
    channel_version: str = DEFAULT_CHANNEL_VERSION
    bot_type: int = DEFAULT_BOT_TYPE
    token_store_path: str = Field(
        default="Local/wechat_credentials.json",
        description="凭据与游标持久化文件（默认放在 Local/ 本地数据目录）",
    )
    route_tag: Optional[str] = None

    # 登录流程
    qr_poll_interval: float = Field(default=2.0, description="扫码状态轮询间隔（秒）")
    qr_max_refresh: int = Field(default=QR_MAX_REFRESH, description="二维码过期最多刷新次数")

    # 网络超时
    longpoll_timeout: float = Field(
        default=LONGPOLL_TIMEOUT_MS / 1000 + 10,
        description="getupdates 读取超时，需大于服务端 35 秒挂起",
    )
    api_timeout: float = Field(default=15.0, description="其余 API 超时（秒）")
    connect_timeout: float = Field(default=10.0)

    # 重试与退避
    retry_base_delay: float = Field(default=2.0, description="长轮询失败 1-2 次等待 2 秒")
    retry_max_delay: float = Field(default=30.0, description="长轮询失败 3 次以上等待 30 秒")
    session_expired_cooldown: float = Field(
        default=3600.0, description="errcode=-14 会话过期后的暂停时长（秒）"
    )

    # 输入状态
    typing_keepalive_interval: float = Field(default=5.0, description="typing keepalive 间隔（秒）")
    typing_ticket_ttl: float = Field(default=86400.0, description="typing_ticket 缓存 TTL（秒，约 24h）")

    @classmethod
    def from_env(cls, **overrides: Any) -> WeChatConfig:
        """从环境变量构造配置，``overrides`` 优先级最高。"""
        values: dict[str, Any] = {}
        env_map = {
            "WECHAT_BASE_URL": "base_url",
            "WECHAT_CDN_BASE_URL": "cdn_base_url",
            "WECHAT_CHANNEL_VERSION": "channel_version",
            "WECHAT_TOKEN_STORE": "token_store_path",
            "WECHAT_ROUTE_TAG": "route_tag",
        }
        for env_key, field in env_map.items():
            value = os.environ.get(env_key)
            if value:
                values[field] = value
        values.update(overrides)
        return cls(**values)
