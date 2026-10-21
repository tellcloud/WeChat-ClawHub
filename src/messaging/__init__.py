"""WeChat-ClawHub 消息处理层。

统一消息模型与消息路由，对应 README 目录结构中的 ``messaging/``：

- :mod:`~messaging.models`     — 系统内部统一 Message 结构；
- :mod:`~messaging.parser`     — 解析微信原始消息（transport → ParsedMessage）；
- :mod:`~messaging.normalizer` — 标准化为统一 Message；
- :mod:`~messaging.router`     — 路由到 Script / AI / Agent。

典型数据流::

    WeixinMessage ─parse▶ ParsedMessage ─normalize▶ Message ─route▶ RoutingResult

公开接口::

    from messaging import MessageParser, MessageNormalizer, MessageRouter

    parser = MessageParser()
    normalizer = MessageNormalizer()
    router = MessageRouter()

    message = normalizer.normalize(parser.parse(weixin_message))
    result = router.route(message)
    # result.target ∈ {SCRIPT, AI, AGENT, IGNORE}
"""

from .models import (
    Attachment,
    ChatType,
    ContentKind,
    Direction,
    Message,
    QuotedMessage,
)
from .normalizer import MessageNormalizer, NormalizerConfig
from .parser import MessageParser, ParsedMessage, to_content_kind
from .router import (
    CommandRegistry,
    CommandSpec,
    MatchType,
    MessageRouter,
    ModeProvider,
    RouterConfig,
    RouteTarget,
    RoutingResult,
    default_registry,
)

__all__ = [
    # 模型
    "Message",
    "Attachment",
    "QuotedMessage",
    "ContentKind",
    "ChatType",
    "Direction",
    # 解析 / 标准化
    "MessageParser",
    "ParsedMessage",
    "to_content_kind",
    "MessageNormalizer",
    "NormalizerConfig",
    # 路由
    "MessageRouter",
    "RouterConfig",
    "RoutingResult",
    "RouteTarget",
    "MatchType",
    "CommandRegistry",
    "CommandSpec",
    "ModeProvider",
    "default_registry",
]
