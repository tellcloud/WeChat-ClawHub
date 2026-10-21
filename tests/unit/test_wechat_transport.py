"""transport/wechat 离线单元测试。

使用 ``httpx.MockTransport`` 模拟 iLink 服务器，不发起真实网络请求，
覆盖：协议编解码、请求头、扫码登录状态机、游标推进、发送载荷、
typing 流程、凭据持久化、退避与重连。

运行方式（任选）::

    python -m unittest discover -s tests/unit -v
    python -m pytest tests/unit/test_wechat_transport.py -v
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# 保证可以导入 src/ 下的 transport 包
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    sys.path.insert(0, str(_path))

import httpx

from transport.wechat import (
    AuthenticationError,
    AuthManager,
    MediaError,
    MediaUploader,
    MessageItemType,
    MessagePoller,
    MessageSender,
    SessionExpiredError,
    TokenStore,
    UploadedMedia,
    WeChatClient,
    WeChatConfig,
    WeixinMessage,
    decode_aes_key,
    decrypt_aes128_ecb,
    download_media,
    encrypt_aes128_ecb,
)
from transport.wechat.client import WeChatHTTPError
from transport.wechat.models import (
    CDNMedia,
    MessageType,
    QRStatus,
    encode_client_version,
    generate_uin,
)


def make_config(store_path: Path) -> WeChatConfig:
    return WeChatConfig(
        token_store_path=str(store_path),
        qr_poll_interval=0.0,  # 测试不等待
        retry_base_delay=0.0,
        retry_max_delay=0.0,
        session_expired_cooldown=0.0,
        typing_keepalive_interval=0.05,
    )


def make_client(handler, store_path: Path, credentials=None) -> WeChatClient:
    """构造挂载 MockTransport 的客户端。"""
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return WeChatClient(
        config=make_config(store_path),
        credentials=credentials,
        http_client=http,
    )


CONFIRMED_CREDS = {
    "ilink_bot_id": "bot123@im.bot",
    "bot_token": "Bearer tk_abc",
    "ilink_user_id": "user42@im.wechat",
    "baseurl": "https://ilinkai.weixin.qq.com",
}


class ProtocolUtilsTest(unittest.TestCase):
    """协议编解码工具。"""

    def test_encode_client_version(self) -> None:
        self.assertEqual(encode_client_version("0.3.0"), 768)
        self.assertEqual(encode_client_version("2.1.1"), 131329)

    def test_encode_client_version_invalid(self) -> None:
        with self.assertRaises(ValueError):
            encode_client_version("2.1")

    def test_generate_uin_is_base64_of_digits(self) -> None:
        uin = generate_uin()
        decoded = base64.b64decode(uin).decode()
        self.assertTrue(decoded.isdigit())
        self.assertLessEqual(int(decoded), 0xFFFFFFFF)

    def test_bearer_token_strips_prefix(self) -> None:
        from transport.wechat import Credentials

        creds = Credentials(**{**CONFIRMED_CREDS, "bot_token": "Bearer tk_abc"})
        self.assertEqual(creds.bearer_token, "tk_abc")
        creds2 = Credentials(**{**CONFIRMED_CREDS, "bot_token": "tk_raw"})
        self.assertEqual(creds2.bearer_token, "tk_raw")


class QrLoginTest(unittest.IsolatedAsyncioTestCase):
    """扫码登录流程。"""

    async def test_login_success_persists_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "creds.json"
            qr_seen: list[str] = []
            status_calls = {"n": 0}

            async def handler(request: httpx.Request) -> httpx.Response:
                path = request.url.path
                if path.endswith("get_bot_qrcode"):
                    return httpx.Response(
                        200,
                        json={
                            "qrcode": "qrc_001",
                            "qrcode_img_content": "data:image/png;base64,AAAA",
                        },
                    )
                if path.endswith("get_qrcode_status"):
                    status_calls["n"] += 1
                    if status_calls["n"] == 1:
                        return httpx.Response(200, json={"status": QRStatus.WAIT})
                    if status_calls["n"] == 2:
                        return httpx.Response(200, json={"status": QRStatus.SCANED})
                    return httpx.Response(
                        200,
                        json={"status": QRStatus.CONFIRMED, **CONFIRMED_CREDS},
                    )
                raise AssertionError(f"unexpected path: {path}")

            async def on_qrcode(qrcode: str, img: str) -> None:
                qr_seen.append(qrcode)

            client = make_client(handler, store_path)
            async with client:
                credentials = await AuthManager(client).login(on_qrcode=on_qrcode)

            self.assertEqual(qr_seen, ["qrc_001"])
            self.assertEqual(credentials.bearer_token, "tk_abc")
            self.assertEqual(credentials.ilink_user_id, "user42@im.wechat")
            # 凭据已持久化，且 client 已带上登录态
            self.assertTrue(client.credentials is not None)
            restored = TokenStore(store_path).load_credentials()
            self.assertIsNotNone(restored)
            self.assertEqual(restored.bearer_token, "tk_abc")

    async def test_login_request_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "creds.json"
            captured: list[dict[str, str]] = []

            async def handler(request: httpx.Request) -> httpx.Response:
                captured.append(dict(request.headers))
                if request.url.path.endswith("get_qrcode_status"):
                    return httpx.Response(
                        200, json={"status": QRStatus.CONFIRMED, **CONFIRMED_CREDS}
                    )
                if request.url.path.endswith("getupdates"):
                    return httpx.Response(
                        200, json={"ret": 0, "msgs": [], "get_updates_buf": ""}
                    )
                return httpx.Response(
                    200, json={"qrcode": "qrc_h", "qrcode_img_content": ""}
                )

            client = make_client(handler, store_path)
            async with client:
                await AuthManager(client).login()
                # 登录后再发一次 POST，验证带认证的请求头
                await client.get_updates("")

            # GET 端点（get_qrcode_status）按协议只携带公共头
            get_headers = captured[0]
            self.assertEqual(get_headers.get("ilink-app-id"), "bot")
            self.assertEqual(get_headers.get("ilink-app-clientversion"), "131329")
            self.assertNotIn("authorization", get_headers)

            # POST 业务端点携带 Bearer 与防重放 UIN
            post_headers = captured[-1]
            self.assertEqual(post_headers.get("authorization"), "Bearer tk_abc")
            self.assertEqual(post_headers.get("authorizationtype"), "ilink_bot_token")
            self.assertIn("x-wechat-uin", post_headers)

    async def test_login_expired_refresh_then_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "creds.json"
            qr_count = {"n": 0}
            status_calls = {"n": 0}

            async def handler(request: httpx.Request) -> httpx.Response:
                if request.url.path.endswith("get_bot_qrcode"):
                    qr_count["n"] += 1
                    return httpx.Response(
                        200,
                        json={
                            "qrcode": f"qrc_{qr_count['n']}",
                            "qrcode_img_content": "",
                        },
                    )
                if request.url.path.endswith("get_qrcode_status"):
                    status_calls["n"] += 1
                    if status_calls["n"] == 1:
                        return httpx.Response(200, json={"status": QRStatus.EXPIRED})
                    return httpx.Response(
                        200, json={"status": QRStatus.CONFIRMED, **CONFIRMED_CREDS}
                    )
                raise AssertionError(request.url.path)

            client = make_client(handler, store_path)
            async with client:
                await AuthManager(client).login()
            self.assertEqual(qr_count["n"], 2)  # 刷新过一次

    async def test_login_expired_over_limit_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "creds.json"

            async def handler(request: httpx.Request) -> httpx.Response:
                if request.url.path.endswith("get_bot_qrcode"):
                    return httpx.Response(
                        200, json={"qrcode": "qrc_x", "qrcode_img_content": ""}
                    )
                return httpx.Response(200, json={"status": QRStatus.EXPIRED})

            client = make_client(handler, store_path)
            async with client:
                with self.assertRaises(AuthenticationError):
                    await AuthManager(client).login()

    async def test_login_redirect_switches_base_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "creds.json"
            hosts_seen: list[str] = []
            status_calls = {"n": 0}

            async def handler(request: httpx.Request) -> httpx.Response:
                hosts_seen.append(request.url.host)
                if request.url.path.endswith("get_qrcode_status"):
                    status_calls["n"] += 1
                    if status_calls["n"] == 1:
                        return httpx.Response(
                            200,
                            json={
                                "status": QRStatus.SCANED_BUT_REDIRECT,
                                "redirect_host": "idc2.weixin.qq.com",
                            },
                        )
                    return httpx.Response(
                        200, json={"status": QRStatus.CONFIRMED, **CONFIRMED_CREDS}
                    )
                return httpx.Response(
                    200, json={"qrcode": "qrc_r", "qrcode_img_content": ""}
                )

            client = make_client(handler, store_path)
            async with client:
                await AuthManager(client).login()
            self.assertEqual(hosts_seen[0], "ilinkai.weixin.qq.com")
            self.assertIn("idc2.weixin.qq.com", hosts_seen)


class PollerTest(unittest.IsolatedAsyncioTestCase):
    """长轮询与游标。"""

    def _authed_client(self, handler, store_path: Path) -> WeChatClient:
        from transport.wechat import Credentials

        return make_client(
            handler, store_path, credentials=Credentials(**CONFIRMED_CREDS)
        )

    async def test_poll_once_dispatches_and_persists_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "creds.json"
            buffers: list[str] = []
            call = {"n": 0}

            def make_msg(mid: int) -> dict:
                return {
                    "seq": 1,
                    "message_id": mid,
                    "from_user_id": "u1@im.wechat",
                    "to_user_id": "bot@im.bot",
                    "message_type": 1,
                    "message_state": 0,
                    "context_token": f"ctx_{mid}",
                    "item_list": [{"type": 1, "text_item": {"text": f"hello {mid}"}}],
                }

            async def handler(request: httpx.Request) -> httpx.Response:
                if request.url.path.endswith("getupdates"):
                    body = json.loads(request.content)
                    buffers.append(body["get_updates_buf"])
                    call["n"] += 1
                    if call["n"] == 1:
                        return httpx.Response(
                            200,
                            json={
                                "ret": 0,
                                "msgs": [make_msg(1)],
                                "get_updates_buf": "cursor_1",
                            },
                        )
                    return httpx.Response(
                        200,
                        json={"ret": 0, "msgs": [], "get_updates_buf": "cursor_1"},
                    )
                raise AssertionError(request.url.path)

            client = self._authed_client(handler, store_path)
            received: list[str] = []

            async def on_message(msg) -> None:
                received.append(msg.text)

            async with client:
                poller = MessagePoller(client)
                await poller.run(on_message, once=True)
                # 游标已持久化
                self.assertEqual(TokenStore(store_path).load_cursor(), "cursor_1")
                # 第二次轮询应回传游标
                await poller.run(on_message, once=True)

            self.assertEqual(received, ["hello 1"])
            self.assertEqual(buffers[0], "")  # 首次为空
            self.assertEqual(buffers[1], "cursor_1")  # 之后回传

    async def test_poll_retry_backoff_policy(self) -> None:
        """连续失败 1-2 次等待 retry_base_delay，3 次以上等待 retry_max_delay。"""
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "creds.json"

            async def handler(request: httpx.Request) -> httpx.Response:
                return httpx.Response(
                    200, json={"ret": 0, "msgs": [], "get_updates_buf": ""}
                )

            config = WeChatConfig(
                token_store_path=str(store_path),
                retry_base_delay=2.0,
                retry_max_delay=30.0,
            )
            http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            client = WeChatClient(config=config, http_client=http)
            poller = MessagePoller(client)

            delays: list[float] = []

            async def fake_sleep(delay: float) -> None:
                delays.append(delay)

            poller._sleep_or_stop = fake_sleep  # type: ignore[method-assign]

            from transport.wechat import WeChatAPIError

            await poller._handle_failure(WeChatAPIError(ret=-1))
            await poller._handle_failure(WeChatAPIError(ret=-1))
            await poller._handle_failure(WeChatAPIError(ret=-1))
            await poller._handle_failure(WeChatAPIError(ret=-1))

            self.assertEqual(delays, [2.0, 2.0, 30.0, 30.0])

    async def test_poll_recovers_after_transient_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "creds.json"
            call = {"n": 0}
            ctx: dict = {}

            async def handler(request: httpx.Request) -> httpx.Response:
                call["n"] += 1
                if call["n"] == 1:
                    return httpx.Response(500, text="boom")
                if call["n"] == 2:
                    ctx["poller"].stop()  # 恢复后即停止，保证确定性
                return httpx.Response(
                    200, json={"ret": 0, "msgs": [], "get_updates_buf": ""}
                )

            client = self._authed_client(handler, store_path)
            received: list[str] = []

            async def on_message(msg) -> None:
                received.append(msg.text)

            async with client:
                poller = MessagePoller(client)
                ctx["poller"] = poller
                await poller.run(on_message)

            self.assertEqual(call["n"], 2)
            self.assertEqual(poller._failure_count, 0)

    async def test_poll_session_expired_triggers_callback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "creds.json"
            expired_called = {"n": 0}

            async def handler(request: httpx.Request) -> httpx.Response:
                return httpx.Response(
                    200, json={"ret": -1, "errcode": -14, "errmsg": "session expired"}
                )

            async def on_expired() -> None:
                expired_called["n"] += 1

            client = self._authed_client(handler, store_path)
            async with client:
                poller = MessagePoller(client, on_session_expired=on_expired)
                with self.assertRaises(SessionExpiredError):
                    await poller.run(lambda msg: asyncio.sleep(0), once=True)
            self.assertEqual(expired_called["n"], 1)
            # 会话过期后凭据被清理
            self.assertIsNone(client.credentials)

    async def test_poll_http_error_raises_in_once_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "creds.json"

            async def handler(request: httpx.Request) -> httpx.Response:
                return httpx.Response(503, text="unavailable")

            client = self._authed_client(handler, store_path)
            async with client:
                poller = MessagePoller(client)
                with self.assertRaises(WeChatHTTPError):
                    await poller.run(lambda msg: asyncio.sleep(0), once=True)


class SenderTest(unittest.IsolatedAsyncioTestCase):
    """消息发送与 typing。"""

    def _authed_client(self, handler, store_path: Path) -> WeChatClient:
        from transport.wechat import Credentials

        return make_client(
            handler, store_path, credentials=Credentials(**CONFIRMED_CREDS)
        )

    async def test_send_text_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "creds.json"
            captured: dict = {}

            async def handler(request: httpx.Request) -> httpx.Response:
                captured["body"] = json.loads(request.content)
                captured["path"] = request.url.path
                captured["headers"] = dict(request.headers)
                return httpx.Response(200)

            client = self._authed_client(handler, store_path)
            async with client:
                sender = MessageSender(client)
                await sender.send_text(
                    "u1@im.wechat", "你好", context_token="ctx_abc"
                )

            self.assertTrue(captured["path"].endswith("sendmessage"))
            msg = captured["body"]["msg"]
            self.assertEqual(msg["to_user_id"], "u1@im.wechat")
            self.assertEqual(msg["context_token"], "ctx_abc")
            self.assertEqual(msg["message_type"], 2)
            self.assertEqual(msg["message_state"], 2)
            self.assertEqual(msg["item_list"][0]["type"], 1)
            self.assertEqual(msg["item_list"][0]["text_item"]["text"], "你好")
            self.assertTrue(msg["client_id"])
            self.assertEqual(captured["body"]["base_info"]["channel_version"], "2.1.1")

    async def test_typing_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "creds.json"
            requests_log: list[tuple[str, dict]] = []

            async def handler(request: httpx.Request) -> httpx.Response:
                path = request.url.path
                body = json.loads(request.content) if request.content else {}
                requests_log.append((path.rsplit("/", 1)[-1], body))
                if path.endswith("getconfig"):
                    return httpx.Response(
                        200, json={"ret": 0, "typing_ticket": "ticket_1"}
                    )
                return httpx.Response(200)

            client = self._authed_client(handler, store_path)
            async with client:
                sender = MessageSender(client)
                async with sender.typing("u1@im.wechat", "ctx_abc"):
                    await asyncio.sleep(0.12)  # 触发至少一次 keepalive

            endpoints = [name for name, _ in requests_log]
            # getconfig 缓存：ticket 只取一次
            self.assertEqual(endpoints.count("getconfig"), 1)
            self.assertIn("sendtyping", endpoints)
            # 首次与退出时各一次 status=1/2
            typing_bodies = [b for name, b in requests_log if name == "sendtyping"]
            self.assertEqual(typing_bodies[0]["status"], 1)
            self.assertEqual(typing_bodies[-1]["status"], 2)
            # keepalive 也是 status=1
            self.assertTrue(all(b["status"] in (1, 2) for b in typing_bodies))

    async def test_empty_text_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "creds.json"
            called = {"n": 0}

            async def handler(request: httpx.Request) -> httpx.Response:
                called["n"] += 1
                return httpx.Response(200)

            client = self._authed_client(handler, store_path)
            async with client:
                await MessageSender(client).send_text("u1@im.wechat", "")
            self.assertEqual(called["n"], 0)


class TokenStoreTest(unittest.IsolatedAsyncioTestCase):
    """凭据持久化。"""

    async def test_save_load_roundtrip(self) -> None:
        from transport.wechat import Credentials

        with tempfile.TemporaryDirectory() as tmp:
            store = TokenStore(Path(tmp) / "sub" / "creds.json")
            self.assertIsNone(store.load_credentials())
            store.save_credentials(Credentials(**CONFIRMED_CREDS))
            store.save_cursor("cursor_x")
            restored = store.load_credentials()
            self.assertEqual(restored.ilink_bot_id, "bot123@im.bot")
            self.assertEqual(store.load_cursor(), "cursor_x")
            store.clear_credentials()
            self.assertIsNone(store.load_credentials())
            self.assertEqual(store.load_cursor(), "cursor_x")  # 游标保留

    async def test_restore_into_client(self) -> None:
        from transport.wechat import Credentials

        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "creds.json"
            store = TokenStore(store_path)
            store.save_credentials(Credentials(**CONFIRMED_CREDS))
            client = WeChatClient(config=make_config(store_path))
            self.assertTrue(AuthManager(client, store).restore())
            self.assertIsNotNone(client.credentials)


class MessageModelTest(unittest.TestCase):
    """消息模型解析。"""

    def test_parse_weixin_message(self) -> None:
        from transport.wechat import WeixinMessage

        msg = WeixinMessage.model_validate(
            {
                "seq": 7,
                "message_id": 1001,
                "from_user_id": "u1@im.wechat",
                "to_user_id": "bot@im.bot",
                "message_type": 1,
                "message_state": 0,
                "context_token": "ctx_zz",
                "item_list": [
                    {"type": 1, "text_item": {"text": "第一段"}},
                    {"type": 1, "text_item": {"text": "第二段"}},
                ],
            }
        )
        self.assertEqual(msg.message_type, MessageType.USER)
        self.assertEqual(msg.text, "第一段第二段")
        self.assertEqual(msg.item_list[0].type, MessageItemType.TEXT)
        self.assertFalse(msg.is_group)

    def test_group_message(self) -> None:
        from transport.wechat import WeixinMessage

        msg = WeixinMessage.model_validate(
            {"group_id": "g1@im.wechat", "message_type": 1}
        )
        self.assertTrue(msg.is_group)


class MediaCryptoTest(unittest.TestCase):
    """AES-128-ECB 加解密与密钥解析。"""

    def test_fips197_known_vector(self) -> None:
        # FIPS-197 附录示例：AES-128 ECB（原始块无填充）
        # 我们的实现带 PKCS7，因此比较密文的第一个块
        key = bytes(range(16))
        plaintext = bytes.fromhex("00112233445566778899aabbccddeeff")
        ciphertext = encrypt_aes128_ecb(plaintext, key)
        self.assertEqual(
            ciphertext[:16].hex(), "69c4e0d86a7b0430d8cdb78070b4c55a"
        )
        self.assertEqual(len(ciphertext), 32)  # 满块后 PKCS7 再补一块
        self.assertEqual(decrypt_aes128_ecb(ciphertext, key), plaintext)

    def test_roundtrip_random_sizes(self) -> None:
        key = os.urandom(16)
        for size in (0, 1, 15, 16, 17, 1000):
            data = os.urandom(size)
            self.assertEqual(
                decrypt_aes128_ecb(encrypt_aes128_ecb(data, key), key), data
            )

    def test_encrypted_size(self) -> None:
        from transport.wechat.media import encrypted_size

        self.assertEqual(encrypted_size(0), 16)
        self.assertEqual(encrypted_size(15), 16)
        self.assertEqual(encrypted_size(16), 32)
        self.assertEqual(encrypted_size(17), 32)

    def test_decode_aes_key_both_formats(self) -> None:
        key = os.urandom(16)
        # 格式 A：base64(原始 16 字节)
        self.assertEqual(decode_aes_key(base64.b64encode(key).decode()), key)
        # 格式 B：base64(32 字符 hex 字符串)
        self.assertEqual(
            decode_aes_key(base64.b64encode(key.hex().encode()).decode()), key
        )

    def test_decode_aes_key_invalid(self) -> None:
        with self.assertRaises(MediaError):
            decode_aes_key(base64.b64encode(b"too-short").decode())


class MediaTransferTest(unittest.IsolatedAsyncioTestCase):
    """CDN 上传 / 下载与图片发送。"""

    IMAGE = b"\x89PNG\r\n\x1a\n" + b"fake-image-payload-" * 8

    def _authed_client(self, handler, store_path: Path) -> WeChatClient:
        from transport.wechat import Credentials

        return make_client(
            handler, store_path, credentials=Credentials(**CONFIRMED_CREDS)
        )

    async def test_upload_image_full_flow(self) -> None:
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "creds.json"
            upload_requests: dict = {}
            cdn_bodies: list[bytes] = []

            async def handler(request: httpx.Request) -> httpx.Response:
                path = request.url.path
                if path.endswith("getuploadurl"):
                    upload_requests["body"] = json.loads(request.content)
                    return httpx.Response(
                        200,
                        json={
                            "upload_param": "up_param_1",
                            "upload_full_url": "https://cdn.test/upload?token=xyz",
                        },
                    )
                if request.url.host == "cdn.test" and request.method == "POST":
                    cdn_bodies.append(request.content)
                    return httpx.Response(
                        200, headers={"x-encrypted-param": "dl_param_1"}
                    )
                raise AssertionError(f"unexpected: {request.method} {request.url}")

            client = self._authed_client(handler, store_path)
            async with client:
                uploader = MediaUploader(client)
                uploaded = await uploader.upload_image("u1@im.wechat", self.IMAGE)
                media = uploaded.media

            # getuploadurl 请求载荷符合协议
            body = upload_requests["body"]
            self.assertEqual(body["media_type"], 1)
            self.assertEqual(body["to_user_id"], "u1@im.wechat")
            self.assertEqual(body["rawsize"], len(self.IMAGE))
            self.assertEqual(body["rawfilemd5"], hashlib.md5(self.IMAGE).hexdigest())
            self.assertEqual(body["filesize"], ((len(self.IMAGE) // 16) + 1) * 16)
            self.assertEqual(len(body["filekey"]), 32)  # 16 字节 hex
            self.assertEqual(len(body["aeskey"]), 32)  # 16 字节 hex
            self.assertTrue(body["no_need_thumb"])

            # CDN 上传的是密文，且能解密回原图
            self.assertEqual(len(cdn_bodies), 1)
            key = decode_aes_key(media.aes_key)
            self.assertEqual(decrypt_aes128_ecb(cdn_bodies[0], key), self.IMAGE)

            # aes_key 采用 base64(hex 字符串) 编码（与官方插件一致，否则客户端打不开）
            decoded_key_field = base64.b64decode(media.aes_key)
            self.assertEqual(len(decoded_key_field), 32)
            self.assertEqual(decoded_key_field.decode(), uploaded.aes_key_hex)
            self.assertEqual(bytes.fromhex(uploaded.aes_key_hex), key)

            # 返回的 CDNMedia 携带下载参数；cipher_size 供 mid_size 使用
            self.assertEqual(media.encrypt_query_param, "dl_param_1")
            self.assertEqual(uploaded.raw_size, len(self.IMAGE))
            self.assertEqual(uploaded.cipher_size, ((len(self.IMAGE) // 16) + 1) * 16)

    async def test_upload_video_full_flow(self) -> None:
        import hashlib

        VIDEO = b"\x00\x00\x00\x18ftypmp42" + b"fake-video-payload-" * 20

        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "creds.json"
            upload_requests: dict = {}
            cdn_bodies: list[bytes] = []

            async def handler(request: httpx.Request) -> httpx.Response:
                path = request.url.path
                if path.endswith("getuploadurl"):
                    upload_requests["body"] = json.loads(request.content)
                    return httpx.Response(
                        200,
                        json={
                            "upload_param": "up_v",
                            "upload_full_url": "https://cdn.test/upload?token=v",
                        },
                    )
                if request.url.host == "cdn.test" and request.method == "POST":
                    cdn_bodies.append(request.content)
                    return httpx.Response(200, headers={"x-encrypted-param": "dl_v"})
                raise AssertionError(f"unexpected: {request.method} {request.url}")

            client = self._authed_client(handler, store_path)
            async with client:
                uploaded = await MediaUploader(client).upload_video(
                    "u1@im.wechat", VIDEO
                )

            body = upload_requests["body"]
            self.assertEqual(body["media_type"], 2)  # VIDEO
            self.assertEqual(body["rawsize"], len(VIDEO))
            self.assertEqual(body["rawfilemd5"], hashlib.md5(VIDEO).hexdigest())
            self.assertEqual(len(cdn_bodies), 1)
            key = decode_aes_key(uploaded.media.aes_key)
            self.assertEqual(decrypt_aes128_ecb(cdn_bodies[0], key), VIDEO)
            self.assertEqual(uploaded.media.encrypt_query_param, "dl_v")

    async def test_send_video_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "creds.json"
            captured: dict = {}

            async def handler(request: httpx.Request) -> httpx.Response:
                captured["body"] = json.loads(request.content)
                return httpx.Response(200)

            client = self._authed_client(handler, store_path)
            async with client:
                sender = MessageSender(client)
                await sender.send_video(
                    "u1@im.wechat",
                    UploadedMedia(
                        media=CDNMedia(
                            encrypt_query_param="dl_v",
                            aes_key=base64.b64encode(b"b" * 32).decode(),
                            encrypt_type=1,
                        ),
                        filekey="f" * 32,
                        aes_key_hex="b" * 32,
                        raw_size=5000,
                        cipher_size=5008,
                    ),
                    context_token="ctx_v",
                )

            msg = captured["body"]["msg"]
            self.assertEqual(msg["context_token"], "ctx_v")
            item = msg["item_list"][0]
            self.assertEqual(item["type"], 5)  # VIDEO
            # video_size 必须为密文字节数
            self.assertEqual(item["video_item"]["video_size"], 5008)
            self.assertEqual(item["video_item"]["media"]["encrypt_query_param"], "dl_v")

    async def test_upload_falls_back_to_cdn_base_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "creds.json"
            seen: dict = {}

            async def handler(request: httpx.Request) -> httpx.Response:
                path = request.url.path
                if path.endswith("getuploadurl"):
                    # 旧版服务端不返回 upload_full_url
                    return httpx.Response(200, json={"upload_param": "up_only"})
                if request.url.host == "novac2c.cdn.weixin.qq.com":
                    seen["url"] = str(request.url)
                    return httpx.Response(200, headers={"x-encrypted-param": "dl2"})
                raise AssertionError(f"unexpected: {request.url}")

            client = self._authed_client(handler, store_path)
            async with client:
                uploaded = await MediaUploader(client).upload_image(
                    "u1@im.wechat", self.IMAGE
                )

            self.assertIn("encrypted_query_param=up_only", seen["url"])
            self.assertIn("filekey=", seen["url"])
            self.assertEqual(uploaded.media.encrypt_query_param, "dl2")

    async def test_download_media_full_url(self) -> None:
        import os

        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "creds.json"
            key = os.urandom(16)
            cipher = encrypt_aes128_ecb(self.IMAGE, key)

            async def handler(request: httpx.Request) -> httpx.Response:
                self.assertEqual(str(request.url), "https://cdn.test/full-url")
                return httpx.Response(200, content=cipher)

            client = self._authed_client(handler, store_path)
            async with client:
                media = CDNMedia(
                    full_url="https://cdn.test/full-url",
                    aes_key=base64.b64encode(key).decode(),
                )
                data = await download_media(client, media)
            self.assertEqual(data, self.IMAGE)

    async def test_download_media_query_param(self) -> None:
        import os

        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "creds.json"
            key = os.urandom(16)
            cipher = encrypt_aes128_ecb(self.IMAGE, key)

            async def handler(request: httpx.Request) -> httpx.Response:
                self.assertIn("/c2c/download", request.url.path)
                self.assertIn("encrypted_query_param=qp_1", str(request.url))
                return httpx.Response(200, content=cipher)

            client = self._authed_client(handler, store_path)
            async with client:
                media = CDNMedia(
                    encrypt_query_param="qp_1",
                    aes_key=base64.b64encode(key.hex().encode()).decode(),  # 格式 B
                )
                data = await download_media(client, media)
            self.assertEqual(data, self.IMAGE)

    async def test_download_media_no_reference_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "creds.json"

            async def handler(request: httpx.Request) -> httpx.Response:
                raise AssertionError("不应发起请求")

            client = self._authed_client(handler, store_path)
            async with client:
                with self.assertRaises(MediaError):
                    await download_media(
                        client,
                        CDNMedia(aes_key=base64.b64encode(b"0" * 16).decode()),
                    )

    async def test_send_image_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "creds.json"
            captured: dict = {}

            async def handler(request: httpx.Request) -> httpx.Response:
                captured["body"] = json.loads(request.content)
                return httpx.Response(200)

            client = self._authed_client(handler, store_path)
            async with client:
                sender = MessageSender(client)
                await sender.send_image(
                    "u1@im.wechat",
                    UploadedMedia(
                        media=CDNMedia(
                            encrypt_query_param="dl_x",
                            aes_key=base64.b64encode(b"a" * 32).decode(),
                            encrypt_type=1,
                        ),
                        filekey="f" * 32,
                        aes_key_hex="a" * 32,
                        raw_size=1000,
                        cipher_size=1008,
                    ),
                    context_token="ctx_img",
                )

            msg = captured["body"]["msg"]
            self.assertEqual(msg["context_token"], "ctx_img")
            self.assertEqual(msg["message_type"], 2)
            item = msg["item_list"][0]
            self.assertEqual(item["type"], 2)
            # mid_size 必须为密文字节数，否则微信客户端打不开图片
            self.assertEqual(item["image_item"]["mid_size"], 1008)
            self.assertEqual(item["image_item"]["media"]["encrypt_query_param"], "dl_x")
            self.assertEqual(
                item["image_item"]["media"]["aes_key"],
                base64.b64encode(b"a" * 32).decode(),
            )
            self.assertEqual(item["image_item"]["media"]["encrypt_type"], 1)


class MainHandlerRoutingReplyTest(unittest.IsolatedAsyncioTestCase):
    """main.py 消息处理：路由 → 模块分发（端到端离线）。

    AI 目标交给注入的 AIService（离线假 Provider）；/role 指令由 AI
    模块执行；/hello 等 Script 指令由注入的 ScriptService 真实执行
    （沙箱子进程）；Agent 目标回复「暂未完成Agent模块的开发工作」；
    IGNORE 不回复，Bot 回声直接跳过。
    """

    def _authed_client(self, handler, store_path: Path) -> WeChatClient:
        from transport.wechat import Credentials

        creds = Credentials(**CONFIRMED_CREDS)
        return make_client(handler, store_path, credentials=creds)

    def _make_ai_service(self, tmp: str):
        """在临时目录上装配离线 AI 服务（隔离 personas 与长期记忆）。"""
        import main

        return main.AIService(
            main.OfflineLLMProvider(),
            config=main.AIServiceConfig(
                personas_dir=Path(tmp) / "personas",
                memory_path=Path(tmp) / "ai_memory.json",
                glm=main.GLMConfig(api_key="test-key"),
            ),
        )

    def _make_script_service(self, tmp: str):
        """在临时目录上装配 Script 服务（隔离沙箱工作目录）。"""
        import main

        return main.ScriptService.from_directory(
            main.SCRIPTS_DIR,
            config=main.ScriptConfig(
                scripts_dir=main.SCRIPTS_DIR,
                work_root=Path(tmp) / "script_runs",
            ),
        )

    async def _run_handler(self, message: WeixinMessage) -> list[dict]:
        """用 mock 服务器跑一遍 handle_message，返回发出的消息载荷。"""
        import main

        sent: list[dict] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("getconfig"):
                return httpx.Response(200, json={"ret": 0, "typing_ticket": "tk"})
            if path.endswith("sendtyping"):
                return httpx.Response(200)
            if path.endswith("sendmessage"):
                sent.append(json.loads(request.content))
                return httpx.Response(200)
            raise AssertionError(f"unexpected: {path}")

        with tempfile.TemporaryDirectory() as tmp:
            client = self._authed_client(handler, Path(tmp) / "creds.json")
            script_service = self._make_script_service(tmp)
            router = main.MessageRouter()
            main.wire_script_service(router, script_service)
            await main.handle_message(
                MessageSender(client),
                main.MessageParser(),
                main.MessageNormalizer(),
                router,
                message,
                self._make_ai_service(tmp),
                script_service,
            )
        return sent

    @staticmethod
    def _raw_message(payload: dict) -> WeixinMessage:
        return WeixinMessage.model_validate(payload)

    async def test_text_message_routes_to_ai(self) -> None:
        message = self._raw_message(
            {
                "message_id": 10,
                "from_user_id": "u1@im.wechat",
                "to_user_id": "b@im.bot",
                "message_type": 1,
                "message_state": 0,
                "context_token": "ctx_t",
                "item_list": [{"type": 1, "text_item": {"text": "hi"}}],
            }
        )
        sent = await self._run_handler(message)

        self.assertEqual(len(sent), 1)
        self.assertEqual(
            sent[0]["msg"]["item_list"][0]["text_item"]["text"],
            "（离线测试）已收到：hi",
        )
        self.assertEqual(sent[0]["msg"]["context_token"], "ctx_t")

    async def test_role_command_handled_by_ai_module(self) -> None:
        """/role 指令由 AI 模块执行（角色切换 + 问候语）。"""
        message = self._raw_message(
            {
                "message_id": 14,
                "from_user_id": "u1@im.wechat",
                "to_user_id": "b@im.bot",
                "message_type": 1,
                "message_state": 0,
                "context_token": "ctx_role",
                "item_list": [{"type": 1, "text_item": {"text": "/role set default"}}],
            }
        )
        sent = await self._run_handler(message)

        self.assertEqual(len(sent), 1)
        self.assertEqual(
            sent[0]["msg"]["item_list"][0]["text_item"]["text"],
            "已切换到角色「default」。你好，我是 ClawHub，有什么可以帮你？",
        )
        self.assertEqual(sent[0]["msg"]["context_token"], "ctx_role")

    async def test_command_message_routes_to_script(self) -> None:
        """/hello 路由到 Script 模块并真实执行（沙箱子进程）。"""
        message = self._raw_message(
            {
                "message_id": 12,
                "from_user_id": "u1@im.wechat",
                "to_user_id": "b@im.bot",
                "message_type": 1,
                "message_state": 0,
                "context_token": "ctx_cmd",
                "item_list": [{"type": 1, "text_item": {"text": "/hello 世界"}}],
            }
        )
        sent = await self._run_handler(message)

        self.assertEqual(len(sent), 1)
        self.assertEqual(
            sent[0]["msg"]["item_list"][0]["text_item"]["text"],
            "Hello, 世界！这是 hello 示例脚本的回复。",
        )
        self.assertEqual(sent[0]["msg"]["context_token"], "ctx_cmd")

    async def test_image_message_routes_to_ai(self) -> None:
        message = self._raw_message(
            {
                "message_id": 9,
                "from_user_id": "u1@im.wechat",
                "to_user_id": "b@im.bot",
                "message_type": 1,
                "message_state": 0,
                "context_token": "ctx_img",
                "item_list": [
                    {
                        "type": 2,
                        "image_item": {
                            "media": {
                                "full_url": "https://cdn.test/full-url",
                                "aes_key": "a2V5",
                            }
                        },
                    }
                ],
            }
        )
        sent = await self._run_handler(message)

        # 图片消息默认路由到 AI：非文本内容回复能力边界提示
        self.assertEqual(len(sent), 1)
        self.assertEqual(
            sent[0]["msg"]["item_list"][0]["text_item"]["text"],
            "我目前只能理解文本消息～图片、语音、文件还看不来。",
        )
        self.assertEqual(sent[0]["msg"]["context_token"], "ctx_img")

    async def test_video_message_routes_to_ai(self) -> None:
        message = self._raw_message(
            {
                "message_id": 11,
                "from_user_id": "u1@im.wechat",
                "to_user_id": "b@im.bot",
                "message_type": 1,
                "message_state": 0,
                "context_token": "ctx_v",
                "item_list": [
                    {
                        "type": 5,
                        "video_item": {
                            "media": {
                                "full_url": "https://cdn.test/full-video",
                                "aes_key": "a2V5",
                            }
                        },
                    }
                ],
            }
        )
        sent = await self._run_handler(message)

        self.assertEqual(len(sent), 1)
        self.assertEqual(
            sent[0]["msg"]["item_list"][0]["text_item"]["text"],
            "我目前只能理解文本消息～图片、语音、文件还看不来。",
        )
        self.assertEqual(sent[0]["msg"]["context_token"], "ctx_v")

    async def test_echo_message_not_replied(self) -> None:
        """Bot 自己发出的回声（message_type=2）不处理、不回复。"""
        message = self._raw_message(
            {
                "message_id": 13,
                "from_user_id": "u1@im.wechat",
                "to_user_id": "b@im.bot",
                "message_type": 2,
                "message_state": 2,
                "context_token": "ctx_echo",
                "item_list": [{"type": 1, "text_item": {"text": "机器人自己的回声"}}],
            }
        )
        sent = await self._run_handler(message)

        self.assertEqual(sent, [])

if __name__ == "__main__":
    unittest.main(verbosity=2)
