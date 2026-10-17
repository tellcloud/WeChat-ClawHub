"""微信 ClawBot / iLink Bot 通信实现。

公开接口::

    from transport.wechat import WeChatClient, WeChatConfig

    async with WeChatClient(WeChatConfig.from_env()) as client:
        await client.login(on_qrcode=show_qrcode)     # 扫码登录
        await client.poll(handle_message)             # 长轮询收消息
        await client.send_message(user_id, "你好")     # 发送文本

模块划分（对应 README 目录结构）：

- :mod:`~transport.wechat.models`  — 协议模型与配置；
- :mod:`~transport.wechat.client`  — iLink API 底层 HTTP 调用；
- :mod:`~transport.wechat.auth`    — 扫码登录、Token 持久化；
- :mod:`~transport.wechat.poller`  — 长轮询消息接收；
- :mod:`~transport.wechat.sender`  — 文本与“正在输入”状态发送；
- :mod:`~transport.wechat.media`   — 图片等媒体上传 / 下载（AES-128-ECB + CDN）。
"""

from .auth import AuthManager, AuthenticationError, TokenStore, save_qrcode_image
from .client import (
    InvalidParamError,
    SessionExpiredError,
    WeChatAPIError,
    WeChatClient,
    WeChatHTTPError,
    WeChatTransportError,
    new_client_id,
)
from .media import (
    MediaError,
    MediaUploader,
    UploadedMedia,
    decode_aes_key,
    decrypt_aes128_ecb,
    download_media,
    encrypt_aes128_ecb,
)
from .models import (
    DEFAULT_BASE_URL,
    DEFAULT_CHANNEL_VERSION,
    ERR_SESSION_EXPIRED,
    ApiResponse,
    BaseInfo,
    CDNMedia,
    Credentials,
    FileItem,
    GetConfigRequest,
    GetConfigResponse,
    GetUpdatesRequest,
    GetUpdatesResponse,
    GetUploadUrlRequest,
    GetUploadUrlResponse,
    ImageItem,
    MessageItem,
    MessageItemType,
    MessageState,
    MessageType,
    QRStatus,
    QrcodeResponse,
    QrcodeStatusResponse,
    RefMessage,
    SendMessageRequest,
    SendMsg,
    SendTypingRequest,
    TextItem,
    TypingStatus,
    UploadMediaType,
    VideoItem,
    VoiceItem,
    WeChatConfig,
    WeixinMessage,
    encode_client_version,
    generate_uin,
)
from .poller import MessageHandler, MessagePoller, SessionExpiredHandler
from .sender import MessageSender

__all__ = [
    # 客户端与组件
    "WeChatClient",
    "AuthManager",
    "MessagePoller",
    "MessageSender",
    "MediaUploader",
    "UploadedMedia",
    "TokenStore",
    # 异常
    "WeChatTransportError",
    "WeChatAPIError",
    "WeChatHTTPError",
    "SessionExpiredError",
    "InvalidParamError",
    "AuthenticationError",
    "MediaError",
    # 配置与凭据
    "WeChatConfig",
    "Credentials",
    # 协议模型
    "WeixinMessage",
    "MessageItem",
    "MessageItemType",
    "MessageType",
    "MessageState",
    "TextItem",
    "ImageItem",
    "VoiceItem",
    "FileItem",
    "VideoItem",
    "CDNMedia",
    "RefMessage",
    "QrcodeResponse",
    "QrcodeStatusResponse",
    "GetUpdatesRequest",
    "GetUpdatesResponse",
    "SendMessageRequest",
    "SendMsg",
    "SendTypingRequest",
    "GetConfigRequest",
    "GetConfigResponse",
    "GetUploadUrlRequest",
    "GetUploadUrlResponse",
    "ApiResponse",
    "BaseInfo",
    "QRStatus",
    "TypingStatus",
    "UploadMediaType",
    # 工具
    "encode_client_version",
    "generate_uin",
    "new_client_id",
    "save_qrcode_image",
    "encrypt_aes128_ecb",
    "decrypt_aes128_ecb",
    "decode_aes_key",
    "download_media",
    "DEFAULT_BASE_URL",
    "DEFAULT_CHANNEL_VERSION",
    "ERR_SESSION_EXPIRED",
    # 回调类型
    "MessageHandler",
    "SessionExpiredHandler",
]
