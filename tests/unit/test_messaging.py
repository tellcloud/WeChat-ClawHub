"""messaging 层离线单元测试。

不发起任何网络请求：直接构造 transport 的 ``WeixinMessage`` 作为输入，
覆盖 parser / normalizer / router 三个模块的核心路径与边界场景。

运行方式（任选）::

    python -m unittest discover -s tests/unit -v
    python -m pytest tests/unit/test_messaging.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# 保证可以导入 src/ 下的 messaging / transport 包
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    sys.path.insert(0, str(_path))

from messaging import (
    ChatType,
    CommandRegistry,
    CommandSpec,
    ContentKind,
    MatchType,
    MessageNormalizer,
    MessageParser,
    MessageRouter,
    RouterConfig,
    RouteTarget,
    to_content_kind,
)
from transport.wechat.models import (
    CDNMedia,
    FileItem,
    ImageItem,
    MessageItem,
    MessageItemType,
    MessageType,
    RefMessage,
    TextItem,
    VideoItem,
    VoiceItem,
    WeixinMessage,
)

# ---------------------------------------------------------------------------
# 测试工具
# ---------------------------------------------------------------------------


def make_message(
    text: str = "",
    *,
    item_list=None,
    from_user_id: str = "user42@im.wechat",
    to_user_id: str = "bot123@im.bot",
    group_id: str = "",
    message_type: MessageType = MessageType.USER,
    context_token: str = "ctx-token-1",
    create_time_ms: int = 1_700_000_000_000,
    message_id: int = 1001,
) -> WeixinMessage:
    """构造一条微信原始消息（默认单文本 item）。"""
    if item_list is None:
        item_list = [
            MessageItem(
                type=MessageItemType.TEXT,
                msg_id="item-1",
                text_item=TextItem(text=text),
            )
        ]
    return WeixinMessage(
        message_id=message_id,
        seq=7,
        from_user_id=from_user_id,
        to_user_id=to_user_id,
        session_id="sess-1",
        group_id=group_id,
        message_type=message_type,
        context_token=context_token,
        create_time_ms=create_time_ms,
        item_list=item_list,
    )


def make_text_item(text: str, msg_id: str = "item-x") -> MessageItem:
    return MessageItem(
        type=MessageItemType.TEXT, msg_id=msg_id, text_item=TextItem(text=text)
    )


def pipeline(message: WeixinMessage, router: MessageRouter | None = None):
    """parse → normalize 快捷链。"""
    parser = MessageParser()
    normalizer = MessageNormalizer()
    return normalizer.normalize(parser.parse(message))


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


class MessageParserTest(unittest.TestCase):
    """原始消息解析。"""

    def setUp(self) -> None:
        self.parser = MessageParser()

    def test_text_message(self) -> None:
        parsed = self.parser.parse(make_message("你好"))
        self.assertEqual(parsed.text, "你好")
        self.assertEqual(parsed.kinds, [ContentKind.TEXT])
        self.assertEqual(parsed.from_user_id, "user42@im.wechat")
        self.assertEqual(parsed.context_token, "ctx-token-1")
        self.assertFalse(parsed.is_echo)
        self.assertEqual(parsed.attachments, [])

    def test_multiple_text_items_concatenated(self) -> None:
        msg = make_message(
            item_list=[make_text_item("你好", "i1"), make_text_item("世界", "i2")]
        )
        parsed = self.parser.parse(msg)
        self.assertEqual(parsed.text, "你好世界")

    def test_voice_with_asr(self) -> None:
        msg = make_message(
            item_list=[
                MessageItem(
                    type=MessageItemType.VOICE,
                    msg_id="v1",
                    voice_item=VoiceItem(
                        playtime=3500, text="今天天气怎么样", encode_type=1
                    ),
                )
            ]
        )
        parsed = self.parser.parse(msg)
        self.assertEqual(parsed.voice_text, "今天天气怎么样")
        self.assertEqual(len(parsed.attachments), 1)
        attachment = parsed.attachments[0]
        self.assertIs(attachment.kind, ContentKind.VOICE)
        self.assertEqual(attachment.duration_ms, 3500)
        self.assertEqual(attachment.asr_text, "今天天气怎么样")
        self.assertEqual(parsed.kinds, [ContentKind.VOICE])

    def test_voice_without_asr(self) -> None:
        msg = make_message(
            item_list=[
                MessageItem(
                    type=MessageItemType.VOICE,
                    msg_id="v1",
                    voice_item=VoiceItem(playtime=1200),
                )
            ]
        )
        parsed = self.parser.parse(msg)
        self.assertEqual(parsed.voice_text, "")
        self.assertEqual(len(parsed.attachments), 1)

    def test_image_attachment_media_passthrough(self) -> None:
        media = CDNMedia(aes_key="a2V5", full_url="https://cdn.example/img")
        msg = make_message(
            item_list=[
                MessageItem(
                    type=MessageItemType.IMAGE,
                    msg_id="p1",
                    image_item=ImageItem(media=media, hd_size=20480, mid_size=4096),
                )
            ]
        )
        parsed = self.parser.parse(msg)
        self.assertEqual(parsed.kinds, [ContentKind.IMAGE])
        attachment = parsed.attachments[0]
        self.assertEqual(attachment.size_bytes, 20480)
        self.assertIs(attachment.media, media)

    def test_file_attachment(self) -> None:
        msg = make_message(
            item_list=[
                MessageItem(
                    type=MessageItemType.FILE,
                    msg_id="f1",
                    file_item=FileItem(file_name="报表.xlsx", len="102400", md5="abc"),
                )
            ]
        )
        parsed = self.parser.parse(msg)
        attachment = parsed.attachments[0]
        self.assertIs(attachment.kind, ContentKind.FILE)
        self.assertEqual(attachment.file_name, "报表.xlsx")
        self.assertEqual(attachment.size_bytes, 102400)
        self.assertEqual(attachment.md5, "abc")

    def test_file_attachment_bad_len(self) -> None:
        msg = make_message(
            item_list=[
                MessageItem(
                    type=MessageItemType.FILE,
                    msg_id="f1",
                    file_item=FileItem(file_name="x.bin", len="not-a-number"),
                )
            ]
        )
        parsed = self.parser.parse(msg)
        self.assertEqual(parsed.attachments[0].size_bytes, 0)

    def test_video_attachment(self) -> None:
        msg = make_message(
            item_list=[
                MessageItem(
                    type=MessageItemType.VIDEO,
                    msg_id="d1",
                    video_item=VideoItem(
                        play_length=8000, video_size=999, video_md5="v"
                    ),
                )
            ]
        )
        parsed = self.parser.parse(msg)
        attachment = parsed.attachments[0]
        self.assertIs(attachment.kind, ContentKind.VIDEO)
        self.assertEqual(attachment.duration_ms, 8000)
        self.assertEqual(attachment.size_bytes, 999)

    def test_quoted_message(self) -> None:
        quoted_item = make_text_item("原始消息内容", "ref-inner")
        msg = make_message(
            item_list=[
                MessageItem(
                    type=MessageItemType.TEXT,
                    msg_id="q1",
                    text_item=TextItem(text="回复一下"),
                    ref_msg=RefMessage(message_item=quoted_item, title="原始消息内容"),
                )
            ]
        )
        parsed = self.parser.parse(msg)
        self.assertIsNotNone(parsed.quoted)
        assert parsed.quoted is not None
        self.assertEqual(parsed.quoted.text, "原始消息内容")
        self.assertIs(parsed.quoted.kind, ContentKind.TEXT)
        self.assertEqual(parsed.quoted.message_id, "ref-inner")

    def test_echo_message_flagged(self) -> None:
        msg = make_message("机器人自己说的", message_type=MessageType.BOT)
        parsed = self.parser.parse(msg)
        self.assertTrue(parsed.is_echo)

    def test_empty_message(self) -> None:
        parsed = self.parser.parse(make_message(item_list=[]))
        self.assertEqual(parsed.text, "")
        self.assertEqual(parsed.kinds, [])
        self.assertEqual(parsed.attachments, [])

    def test_unknown_item_type(self) -> None:
        msg = make_message(
            item_list=[
                MessageItem(type=MessageItemType.NONE, msg_id="n1"),
            ]
        )
        parsed = self.parser.parse(msg)
        self.assertEqual(parsed.kinds, [ContentKind.UNKNOWN])

    def test_to_content_kind_mapping(self) -> None:
        self.assertIs(to_content_kind(MessageItemType.TEXT), ContentKind.TEXT)
        self.assertIs(to_content_kind(MessageItemType.IMAGE), ContentKind.IMAGE)
        self.assertIs(to_content_kind(MessageItemType.VOICE), ContentKind.VOICE)
        self.assertIs(to_content_kind(MessageItemType.FILE), ContentKind.FILE)
        self.assertIs(to_content_kind(MessageItemType.VIDEO), ContentKind.VIDEO)
        self.assertIs(to_content_kind(MessageItemType.NONE), ContentKind.UNKNOWN)

    def test_parse_many_keeps_order(self) -> None:
        msgs = [make_message("一", message_id=1), make_message("二", message_id=2)]
        parsed_list = self.parser.parse_many(msgs)
        self.assertEqual([p.text for p in parsed_list], ["一", "二"])


# ---------------------------------------------------------------------------
# normalizer
# ---------------------------------------------------------------------------


class MessageNormalizerTest(unittest.TestCase):
    """消息标准化。"""

    def setUp(self) -> None:
        self.normalizer = MessageNormalizer()

    def test_private_chat_keys(self) -> None:
        parsed = MessageParser().parse(make_message("hello"))
        message = self.normalizer.normalize(parsed)
        self.assertIs(message.chat_type, ChatType.PRIVATE)
        self.assertEqual(message.session_key, "user:user42@im.wechat")
        self.assertEqual(message.sender_key, "user42")
        self.assertEqual(message.bot_id, "bot123@im.bot")
        self.assertIs(message.kind, ContentKind.TEXT)
        self.assertEqual(message.text, "hello")

    def test_group_chat_keys_and_mentions(self) -> None:
        msg = make_message("@ClawBot 你好", group_id="group@im.wechat")
        message = self.normalizer.normalize(MessageParser().parse(msg))
        self.assertIs(message.chat_type, ChatType.GROUP)
        self.assertTrue(message.is_group)
        self.assertEqual(message.session_key, "group:group@im.wechat")
        self.assertEqual(message.mentions, ["ClawBot"])
        # 开头的 @ 提及被剥离，正文保留
        self.assertEqual(message.text, "你好")

    def test_whitespace_collapse(self) -> None:
        msg = make_message("  你好　\u2005世界  \n 朋友 ")
        message = self.normalizer.normalize(MessageParser().parse(msg))
        self.assertEqual(message.text, "你好 世界 朋友")

    def test_voice_asr_promoted_to_text(self) -> None:
        msg = make_message(
            item_list=[
                MessageItem(
                    type=MessageItemType.VOICE,
                    msg_id="v1",
                    voice_item=VoiceItem(text="  讲个笑话 "),
                )
            ]
        )
        message = self.normalizer.normalize(MessageParser().parse(msg))
        self.assertEqual(message.text, "讲个笑话")
        self.assertEqual(message.asr_text, "  讲个笑话 ")
        # 有文本时主类型为 TEXT，语音附件保留
        self.assertIs(message.kind, ContentKind.TEXT)
        self.assertEqual(len(message.attachments), 1)

    def test_media_only_message_kind(self) -> None:
        msg = make_message(
            item_list=[
                MessageItem(
                    type=MessageItemType.IMAGE,
                    msg_id="p1",
                    image_item=ImageItem(),
                )
            ]
        )
        message = self.normalizer.normalize(MessageParser().parse(msg))
        self.assertEqual(message.text, "")
        self.assertIs(message.kind, ContentKind.IMAGE)
        self.assertFalse(message.has_text)

    def test_empty_message_kind_unknown(self) -> None:
        parsed = MessageParser().parse(make_message(item_list=[]))
        message = self.normalizer.normalize(parsed)
        self.assertIs(message.kind, ContentKind.UNKNOWN)

    def test_created_at_from_protocol(self) -> None:
        message = self.normalizer.normalize(MessageParser().parse(make_message("hi")))
        self.assertIsNotNone(message.created_at)
        self.assertEqual(message.created_at.year, 2023)

    def test_created_at_fallback_when_missing(self) -> None:
        msg = make_message("hi", create_time_ms=0)
        message = self.normalizer.normalize(MessageParser().parse(msg))
        self.assertIsNotNone(message.created_at)
        self.assertEqual(message.created_at, message.received_at)

    def test_context_token_preserved(self) -> None:
        message = self.normalizer.normalize(MessageParser().parse(make_message("hi")))
        self.assertEqual(message.context_token, "ctx-token-1")

    def test_disable_strip_mentions(self) -> None:
        normalizer = MessageNormalizer()
        normalizer.config = normalizer.config.model_copy(
            update={"strip_leading_mentions": False}
        )
        msg = make_message("@ClawBot 你好", group_id="g@im.wechat")
        message = normalizer.normalize(MessageParser().parse(msg))
        self.assertEqual(message.text, "@ClawBot 你好")


# ---------------------------------------------------------------------------
# router
# ---------------------------------------------------------------------------


class MessageRouterTest(unittest.TestCase):
    """消息路由：三级匹配 + 粘性 + 群聊策略。"""

    def _router(self, **kwargs) -> MessageRouter:
        registry = CommandRegistry(
            [
                CommandSpec(
                    name="help",
                    keywords=("帮助", "菜单"),
                    help_text="/help — 显示所有可用指令",
                ),
                CommandSpec(name="role", keywords=("角色",)),
                CommandSpec(
                    name="weather",
                    keywords=("天气", "查天气"),
                    help_text="/weather <城市> — 查询天气",
                ),
            ]
        )
        return MessageRouter(registry, **kwargs)

    def _route_text(self, text: str, router: MessageRouter, **msg_kwargs):
        message = pipeline(make_message(text, **msg_kwargs))
        return router.route(message)

    # -- 精确匹配 ------------------------------------------------------------

    def test_exact_command(self) -> None:
        result = self._route_text("/help", self._router())
        self.assertIs(result.target, RouteTarget.SCRIPT)
        self.assertIs(result.match_type, MatchType.EXACT)
        self.assertEqual(result.command, "help")
        self.assertEqual(result.args, [])

    def test_command_with_args(self) -> None:
        result = self._route_text("/role set 程序员", self._router())
        self.assertEqual(result.command, "role")
        self.assertEqual(result.args, ["set", "程序员"])

    def test_command_with_quoted_args(self) -> None:
        registry = CommandRegistry([CommandSpec(name="say")])
        result = self._route_text('/say "hello world"', MessageRouter(registry))
        self.assertEqual(result.command, "say")
        self.assertEqual(result.args, ["hello world"])

    def test_command_case_insensitive(self) -> None:
        result = self._route_text("/HELP", self._router())
        self.assertEqual(result.command, "help")

    def test_leading_mention_before_command(self) -> None:
        router = MessageRouter(CommandRegistry([CommandSpec(name="help")]))
        result = self._route_text("@ClawBot /help", router, group_id="g@im.wechat")
        self.assertEqual(result.command, "help")

    # -- 模糊匹配 ------------------------------------------------------------

    def test_command_prefix_match(self) -> None:
        result = self._route_text("/we 北京", self._router())
        self.assertIs(result.match_type, MatchType.KEYWORD)
        self.assertEqual(result.command, "weather")
        self.assertEqual(result.args, ["北京"])

    def test_ambiguous_prefix_no_match(self) -> None:
        registry = CommandRegistry(
            [CommandSpec(name="role"), CommandSpec(name="run")]
        )
        result = self._route_text("/r x", MessageRouter(registry))
        self.assertIs(result.match_type, MatchType.FALLBACK)

    def test_keyword_match_without_prefix(self) -> None:
        result = self._route_text("查天气 北京", self._router())
        self.assertIs(result.target, RouteTarget.SCRIPT)
        self.assertIs(result.match_type, MatchType.KEYWORD)
        self.assertEqual(result.command, "weather")
        self.assertEqual(result.args, ["北京"])
        self.assertEqual(result.keyword, "查天气")

    def test_keyword_longest_wins(self) -> None:
        registry = CommandRegistry(
            [
                CommandSpec(name="weather", keywords=("天气",)),
                CommandSpec(name="forecast", keywords=("天气预报",)),
            ]
        )
        result = self._route_text("帮我看看天气预报", MessageRouter(registry))
        self.assertEqual(result.command, "forecast")

    def test_keyword_match_disabled(self) -> None:
        config = RouterConfig(enable_keyword_match=False)
        result = self._route_text("查天气 北京", self._router(config=config))
        self.assertIs(result.target, RouteTarget.AI)

    # -- 兜底 ----------------------------------------------------------------

    def test_unknown_command_falls_back_to_help(self) -> None:
        result = self._route_text("/nosuch foo", self._router())
        self.assertIs(result.target, RouteTarget.SCRIPT)
        self.assertIs(result.match_type, MatchType.FALLBACK)
        self.assertEqual(result.command, "help")
        self.assertEqual(result.args, ["nosuch", "foo"])

    # -- 默认 / 粘性 / 群聊 ----------------------------------------------

    def test_plain_text_goes_to_ai(self) -> None:
        result = self._route_text("帮我写个快速排序", self._router())
        self.assertIs(result.target, RouteTarget.AI)
        self.assertIs(result.match_type, MatchType.DEFAULT)

    def test_non_text_goes_to_ai_by_default(self) -> None:
        msg = make_message(
            item_list=[
                MessageItem(
                    type=MessageItemType.IMAGE, msg_id="p1", image_item=ImageItem()
                )
            ]
        )
        result = self._router().route(pipeline(msg))
        self.assertIs(result.target, RouteTarget.AI)

    def test_sticky_agent_mode(self) -> None:
        router = self._router(
            mode_provider=lambda session_key: RouteTarget.AGENT
        )
        result = self._route_text("帮我部署一下服务", router)
        self.assertIs(result.target, RouteTarget.AGENT)
        self.assertIs(result.match_type, MatchType.STICKY)

    def test_sticky_mode_overridden_by_command(self) -> None:
        router = self._router(mode_provider=lambda session_key: RouteTarget.AGENT)
        result = self._route_text("/help", router)
        self.assertIs(result.target, RouteTarget.SCRIPT)
        self.assertIs(result.match_type, MatchType.EXACT)

    def test_group_ignore_without_mention(self) -> None:
        config = RouterConfig(group_require_mention=True, bot_names=("ClawBot",))
        result = self._route_text(
            "大家今天怎么样", self._router(config=config), group_id="g@im.wechat"
        )
        self.assertIs(result.target, RouteTarget.IGNORE)

    def test_group_with_bot_mention_routed(self) -> None:
        config = RouterConfig(group_require_mention=True, bot_names=("ClawBot",))
        result = self._route_text(
            "@ClawBot 讲个笑话", self._router(config=config), group_id="g@im.wechat"
        )
        self.assertIs(result.target, RouteTarget.AI)

    def test_group_other_mention_ignored(self) -> None:
        config = RouterConfig(group_require_mention=True, bot_names=("ClawBot",))
        result = self._route_text(
            "@张三 你好", self._router(config=config), group_id="g@im.wechat"
        )
        self.assertIs(result.target, RouteTarget.IGNORE)

    def test_mode_provider_failure_is_safe(self) -> None:
        def broken(session_key: str):
            raise RuntimeError("session 层炸了")

        router = self._router(mode_provider=broken)
        result = self._route_text("你好", router)
        self.assertIs(result.target, RouteTarget.AI)

    def test_routing_result_carries_message(self) -> None:
        result = self._route_text("/help", self._router())
        self.assertEqual(result.message.text, "/help")
        self.assertEqual(result.message.session_key, "user:user42@im.wechat")


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


class CommandRegistryTest(unittest.TestCase):
    """命令注册表。"""

    def test_register_and_resolve_exact(self) -> None:
        registry = CommandRegistry([CommandSpec(name="Weather", keywords=("天气",))])
        spec = registry.resolve_exact("weather")
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.name, "weather")

    def test_register_rejects_empty_name(self) -> None:
        registry = CommandRegistry()
        with self.assertRaises(ValueError):
            registry.register(CommandSpec(name="  "))

    def test_register_overwrites(self) -> None:
        registry = CommandRegistry([CommandSpec(name="help", help_text="旧")])
        registry.register(CommandSpec(name="help", help_text="新"))
        spec = registry.resolve_exact("help")
        assert spec is not None
        self.assertEqual(spec.help_text, "新")

    def test_unregister(self) -> None:
        registry = CommandRegistry([CommandSpec(name="help")])
        registry.unregister("help")
        self.assertIsNone(registry.resolve_exact("help"))
        registry.unregister("help")  # 重复注销不报错

    def test_resolve_keyword_case_insensitive(self) -> None:
        registry = CommandRegistry([CommandSpec(name="weather", keywords=("Weather",))])
        spec = registry.resolve_keyword("今天WEATHER不错")
        self.assertIsNotNone(spec)

    def test_commands_sorted(self) -> None:
        registry = CommandRegistry(
            [
                CommandSpec(name="weather"),
                CommandSpec(name="agent"),
                CommandSpec(name="help"),
            ]
        )
        self.assertEqual(
            [s.name for s in registry.commands()], ["agent", "help", "weather"]
        )


if __name__ == "__main__":
    unittest.main()
