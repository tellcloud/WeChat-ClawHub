"""AI 对话模块离线单元测试。

不发起任何真实网络请求：GLM Provider 使用 ``httpx.MockTransport``
模拟智谱服务器；AIService 使用 FakeProvider / FakeSink 注入。
覆盖 persona / memory / context / providers / service 五个模块的
核心路径与边界场景。

运行方式（任选）::

    python -m unittest discover -s tests/unit -v
    python -m pytest tests/unit/test_ai.py -v
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# 保证可以导入 src/ 下的 messaging 包与根目录的 modules 包
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    sys.path.insert(0, str(_path))

import httpx

from modules.ai import (
    AIService,
    AIServiceConfig,
    ChatRequest,
    ChatRole,
    ContextBuilder,
    ContextConfig,
    ConversationMemory,
    FactExtractor,
    GLMConfig,
    GLMProvider,
    HeuristicFactExtractor,
    LLMProvider,
    LongTermMemory,
    MemoryEntry,
    Persona,
    PersonaError,
    PersonaManager,
    PersonaNotFoundError,
    ProviderAPIError,
    ProviderAuthError,
    ProviderConfig,
    ProviderRateLimitError,
    Usage,
    available_providers,
    create_provider,
)
from modules.ai.memory import ChatMessage
from messaging.models import ContentKind, Message, QuotedMessage

# ---------------------------------------------------------------------------
# 测试工具
# ---------------------------------------------------------------------------


def make_message(
    text: str = "",
    *,
    session_key: str = "user:u1@im.wechat",
    sender_id: str = "u1@im.wechat",
    kind: ContentKind = ContentKind.TEXT,
    quoted_text: str = "",
    message_id: str = "m-1",
) -> Message:
    """构造一条标准化消息（AI 模块的输入）。"""
    return Message(
        message_id=message_id,
        sender_id=sender_id,
        session_key=session_key,
        kind=kind,
        text=text,
        context_token="ctx-1",
        quoted=QuotedMessage(text=quoted_text) if quoted_text else None,
    )


class FakeProvider(LLMProvider):
    """测试用 Provider：按队列返回预设回复，可模拟抛错。"""

    def __init__(
        self,
        replies: list[str] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        super().__init__(ProviderConfig(api_key="fake-key"))
        self._replies = list(replies or [])
        self._error = error
        self.requests: list[ChatRequest] = []

    @property
    def name(self) -> str:
        return "fake"

    async def _chat_once(self, request: ChatRequest):
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        text = self._replies.pop(0) if self._replies else "（默认假回复）"
        from modules.ai.providers.base import ChatResponse

        return ChatResponse(
            content=text, model="fake-model", usage=Usage(total_tokens=7)
        )


class FakeSink:
    """测试用回复出口：记录发送的文本。"""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.typing_used = False

    async def send_text(
        self,
        to_user_id: str,
        text: str,
        *,
        context_token: str = "",
        ilink_user_id: str | None = None,
    ) -> None:
        self.sent.append((to_user_id, text))

    @contextlib.asynccontextmanager
    async def typing(self, user_id: str, context_token: str = ""):
        self.typing_used = True
        yield


class FakeSummarizer:
    """测试用摘要器：返回固定摘要。"""

    def __init__(self, text: str = "这是更早对话的摘要。") -> None:
        self.text = text
        self.calls: list[list[ChatMessage]] = []

    async def summarize(self, messages: list[ChatMessage]) -> str:
        self.calls.append(messages)
        return self.text


def write_persona(directory: Path, filename: str, payload: dict) -> Path:
    """写一个角色配置文件。"""
    path = directory / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def make_glm(handler, **config_kwargs) -> GLMProvider:
    """构造挂载 MockTransport 的 GLM Provider。"""
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://open.bigmodel.cn/api/paas/v4",
    )
    config = GLMConfig(api_key="test-key", retry_base_delay=0.0, **config_kwargs)
    return GLMProvider(config, client=http)


def glm_ok_response(content: str = "你好，我是 GLM。") -> dict:
    """GLM 成功响应体。"""
    return {
        "model": "glm-4-flash",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


# ---------------------------------------------------------------------------
# PersonaManager
# ---------------------------------------------------------------------------


class PersonaManagerTest(unittest.IsolatedAsyncioTestCase):
    """角色管理：加载、热重载、匹配、增删。"""

    def test_load_from_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_persona(
                root,
                "default.json",
                {"name": "default", "systemPrompt": "默认人格", "greeting": "hi"},
            )
            write_persona(
                root,
                "programmer.json",
                {
                    "name": "程序员",
                    "system_prompt": "工程师人格",
                    "temperature": 0.5,
                    "model": "glm-4-air",
                    "greeting": "你好鸭",
                    "description": "编程助手",
                },
            )
            manager = PersonaManager(root, auto_reload=False)

            personas = manager.list()
            self.assertEqual([p.name for p in personas], ["default", "程序员"])

            # 按名称 / 文件 stem / 大小写不敏感均可命中
            self.assertEqual(manager.get("程序员").model, "glm-4-air")
            self.assertEqual(manager.get("programmer").name, "程序员")
            self.assertEqual(manager.get("PROGRAMMER").name, "程序员")
            self.assertEqual(manager.get("Default").system_prompt, "默认人格")

            with self.assertRaises(PersonaNotFoundError) as ctx:
                manager.get("不存在")
            self.assertIn("程序员", str(ctx.exception))

    def test_missing_directory_falls_back_to_builtin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = PersonaManager(Path(tmp) / "no_such_dir", auto_reload=False)
            personas = manager.list()
            self.assertEqual([p.name for p in personas], ["default"])
            self.assertTrue(manager.get_default().system_prompt)

    def test_invalid_file_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "broken.json").write_text("{not json", encoding="utf-8")
            write_persona(root, "ok.json", {"name": "ok", "systemPrompt": "x"})
            manager = PersonaManager(root, auto_reload=False)
            self.assertEqual([p.name for p in manager.list()], ["default", "ok"])

    def test_name_defaults_to_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_persona(Path(tmp), "translator.json", {"systemPrompt": "翻译"})
            manager = PersonaManager(Path(tmp), auto_reload=False)
            self.assertEqual(manager.get("translator").name, "translator")

    def test_create_and_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = PersonaManager(root, auto_reload=False)
            path = manager.create(Persona(name="诗人", systemPrompt="写诗"))
            self.assertTrue(path.is_file())

            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["name"], "诗人")
            self.assertEqual(saved["systemPrompt"], "写诗")

            with self.assertRaises(PersonaError):
                manager.create(Persona(name="诗人", systemPrompt="重复"))

            # save 覆盖已存在角色
            manager.save(Persona(name="诗人", systemPrompt="写现代诗"))
            self.assertEqual(manager.get("诗人").system_prompt, "写现代诗")

    def test_hot_reload_on_file_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = write_persona(
                root, "chef.json", {"name": "厨师", "systemPrompt": "做菜"}
            )
            manager = PersonaManager(root, auto_reload=True)
            self.assertEqual(manager.get("厨师").system_prompt, "做菜")

            # 改内容 + 手动 bump mtime（绕过文件系统时间精度问题）
            path.write_text(
                json.dumps(
                    {"name": "厨师", "systemPrompt": "做西餐"}, ensure_ascii=False
                ),
                encoding="utf-8",
            )
            os.utime(path, (path.stat().st_atime + 10, path.stat().st_mtime + 10))

            self.assertEqual(manager.get("厨师").system_prompt, "做西餐")

    def test_resolve_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = PersonaManager(Path(tmp), auto_reload=False)
            # 空名 / 不存在都回退默认角色
            self.assertEqual(manager.resolve("").name, "default")
            self.assertEqual(manager.resolve(None).name, "default")
            self.assertEqual(manager.resolve("幻影").name, "default")

    def test_unregister(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_persona(Path(tmp), "a.json", {"name": "a", "systemPrompt": "x"})
            manager = PersonaManager(Path(tmp), auto_reload=False)
            self.assertTrue(manager.unregister("a"))
            self.assertFalse(manager.exists("a"))
            self.assertFalse(manager.unregister("a"))


# ---------------------------------------------------------------------------
# ConversationMemory
# ---------------------------------------------------------------------------


class ConversationMemoryTest(unittest.IsolatedAsyncioTestCase):
    """短期上下文：隔离、窗口、压缩。"""

    async def test_add_get_and_isolation(self) -> None:
        memory = ConversationMemory(max_messages=10)
        memory.add("user:u1", "default", ChatMessage(role=ChatRole.USER, content="hi"))
        memory.add("user:u1", "程序员", ChatMessage(role=ChatRole.USER, content="code"))
        memory.add("user:u2", "default", ChatMessage(role=ChatRole.USER, content="yo"))

        self.assertEqual(len(memory.get("user:u1", "default")), 1)
        self.assertEqual(len(memory.get("user:u1", "程序员")), 1)
        self.assertEqual(len(memory.get("user:u2", "default")), 1)
        self.assertEqual(memory.get("user:u1", "default")[0].content, "hi")

    def test_maxlen_trimming(self) -> None:
        memory = ConversationMemory(max_messages=3)
        for i in range(5):
            memory.add("s", "p", ChatMessage(role=ChatRole.USER, content=str(i)))
        contents = [m.content for m in memory.get("s", "p")]
        self.assertEqual(contents, ["2", "3", "4"])

    def test_clear(self) -> None:
        memory = ConversationMemory()
        memory.add("s", "p1", ChatMessage(role=ChatRole.USER, content="a"))
        memory.add("s", "p2", ChatMessage(role=ChatRole.USER, content="b"))
        self.assertEqual(memory.clear("s", "p1"), 1)
        self.assertEqual(len(memory.get("s", "p1")), 0)
        self.assertEqual(memory.clear("s"), 1)  # 清剩余角色
        self.assertEqual(memory.clear("s"), 0)

    async def test_compress_merges_summary(self) -> None:
        memory = ConversationMemory(max_messages=20)
        for i in range(4):
            memory.add("s", "p", ChatMessage(role=ChatRole.USER, content=f"m{i}"))
        memory.set_summary("s", "p", "旧摘要")

        summarizer = FakeSummarizer("新摘要")
        digest = await memory.compress("s", "p", keep_recent=1, summarizer=summarizer)

        self.assertEqual(digest, "新摘要")
        self.assertEqual(len(summarizer.calls[0]), 3)  # 前 3 条被压缩
        kept = memory.get("s", "p")
        self.assertEqual([m.content for m in kept], ["m3"])
        self.assertEqual(memory.summary("s", "p"), "旧摘要\n新摘要")

    async def test_compress_noop_when_within_keep_recent(self) -> None:
        memory = ConversationMemory()
        memory.add("s", "p", ChatMessage(role=ChatRole.USER, content="only"))
        digest = await memory.compress(
            "s", "p", keep_recent=5, summarizer=FakeSummarizer()
        )
        self.assertEqual(digest, "")
        self.assertEqual(len(memory.get("s", "p")), 1)

    def test_stats(self) -> None:
        memory = ConversationMemory()
        memory.add("s", "p", ChatMessage(role=ChatRole.USER, content="a"))
        self.assertEqual(memory.stats(), {"s|p": 1})


# ---------------------------------------------------------------------------
# LongTermMemory
# ---------------------------------------------------------------------------


class LongTermMemoryTest(unittest.TestCase):
    """长期记忆：写入、检索、合并、持久化。"""

    def test_remember_and_recall(self) -> None:
        memory = LongTermMemory(None)
        memory.remember("user:u1", "用户喜欢喝美式咖啡")
        memory.remember("user:u1", "用户在做 WeChat-ClawHub 项目")

        hits = memory.recall("user:u1", "咖啡")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].content, "用户喜欢喝美式咖啡")
        self.assertEqual(hits[0].hits, 1)

        # 再次召回 hits 累加
        memory.recall("user:u1", "咖啡")
        self.assertEqual(memory.recall("user:u1", "咖啡")[0].hits, 3)

    def test_recall_without_query_returns_recent(self) -> None:
        memory = LongTermMemory(None)
        memory.remember("user:u1", "事实一")
        memory.remember("user:u1", "事实二")
        recent = memory.recall("user:u1", "", limit=1)
        self.assertEqual(recent[0].content, "事实二")

    def test_exact_duplicate_refreshed_not_duplicated(self) -> None:
        memory = LongTermMemory(None)
        entry1, merged1 = memory.remember("user:u1", "我喜欢 Python")
        entry2, merged2 = memory.remember("user:u1", "我喜欢 Python")
        self.assertFalse(merged1)
        self.assertTrue(merged2)
        self.assertEqual(len(memory.entries("user:u1")), 1)
        self.assertEqual(entry1.created_at, entry2.created_at)

    def test_similar_entries_merged(self) -> None:
        memory = LongTermMemory(None)
        memory.remember("user:u1", "用户的名字是张三")
        _, merged = memory.remember("user:u1", "用户的名字是张三丰")
        self.assertTrue(merged)
        entries = memory.entries("user:u1")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].content, "用户的名字是张三丰")

    def test_forget(self) -> None:
        memory = LongTermMemory(None)
        memory.remember("user:u1", "用户喜欢喝美式咖啡")
        memory.remember("user:u1", "用户怕孤单")

        self.assertEqual(memory.forget("user:u1", "咖啡"), 1)
        self.assertEqual(len(memory.entries("user:u1")), 1)
        self.assertEqual(memory.forget("user:u1"), 1)  # content=None 清空会话
        self.assertEqual(memory.entries("user:u1"), [])
        self.assertEqual(memory.forget("user:u1"), 0)

    def test_max_entries_eviction(self) -> None:
        memory = LongTermMemory(None, max_entries_per_session=2)
        memory.remember("user:u1", "事实一")
        memory.remember("user:u1", "事实二")
        memory.remember("user:u1", "事实三")
        contents = [e.content for e in memory.entries("user:u1")]
        self.assertEqual(contents, ["事实二", "事实三"])

    def test_empty_content_rejected(self) -> None:
        memory = LongTermMemory(None)
        with self.assertRaises(ValueError):
            memory.remember("user:u1", "   ")

    def test_persistence_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mem.json"
            memory = LongTermMemory(path)
            memory.remember("user:u1", "用户的名字叫老王", source="msg-9")

            restored = LongTermMemory(path)
            entries = restored.entries("user:u1")
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].content, "用户的名字叫老王")
            self.assertEqual(entries[0].source, "msg-9")
            self.assertEqual(restored.session_count(), 1)

    def test_corrupt_file_starts_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mem.json"
            path.write_text("{broken", encoding="utf-8")
            memory = LongTermMemory(path)
            self.assertEqual(memory.session_count(), 0)

    def test_session_isolation(self) -> None:
        memory = LongTermMemory(None)
        memory.remember("user:u1", "u1 的事实")
        self.assertEqual(memory.recall("user:u2", "事实"), [])
        self.assertEqual(len(memory.recall("user:u1", "事实")), 1)


# ---------------------------------------------------------------------------
# HeuristicFactExtractor
# ---------------------------------------------------------------------------


class HeuristicFactExtractorTest(unittest.TestCase):
    """启发式事实抽取。"""

    def setUp(self) -> None:
        self.extractor = HeuristicFactExtractor()

    def test_extract_patterns(self) -> None:
        cases = {
            "记住我喜欢喝美式咖啡": ["我喜欢喝美式咖啡"],
            "我叫王小虎": ["王小虎"],
            "我的名字是阿明": ["阿明"],
            "我喜欢周末去爬山": ["周末去爬山"],
            "我的职业是产品经理": ["职业是产品经理"],
            "Remember I prefer dark mode": ["I prefer dark mode"],
            "my name is Bob": ["Bob"],
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.extractor.extract(text), expected)

    def test_question_not_extracted(self) -> None:
        self.assertEqual(self.extractor.extract("你记得我喜欢什么吗？"), [])
        self.assertEqual(self.extractor.extract("我叫什么名字?"), [])

    def test_no_match_returns_empty(self) -> None:
        self.assertEqual(self.extractor.extract("今天天气怎么样"), [])
        self.assertEqual(self.extractor.extract(""), [])

    def test_trailing_punctuation_stripped(self) -> None:
        self.assertEqual(self.extractor.extract("记住我喜欢喝咖啡。"), ["我喜欢喝咖啡"])

    def test_is_protocol_compatible(self) -> None:
        self.assertIsInstance(self.extractor, FactExtractor)


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------


class ContextBuilderTest(unittest.TestCase):
    """上下文组装。"""

    def setUp(self) -> None:
        self.persona = Persona(name="default", system_prompt="你是助手")
        self.builder = ContextBuilder()

    def test_full_assembly(self) -> None:
        history = [
            ChatMessage(role=ChatRole.USER, content="上一轮问题"),
            ChatMessage(role=ChatRole.ASSISTANT, content="上一轮回答"),
        ]
        memories = [MemoryEntry(content="用户喜欢喝美式咖啡")]
        assembled = self.builder.build(
            persona=self.persona,
            history=history,
            user_text="推荐个咖啡店",
            summary="用户在筹备咖啡店开业。",
            memories=memories,
        )

        messages = assembled.messages
        # 顺序：system → 历史 → 当前用户
        self.assertEqual(messages[0].role, ChatRole.SYSTEM)
        self.assertEqual(messages[1].role, ChatRole.USER)
        self.assertEqual(messages[2].role, ChatRole.ASSISTANT)
        self.assertEqual(messages[-1].role, ChatRole.USER)
        self.assertEqual(messages[-1].content, "推荐个咖啡店")

        system = messages[0].content
        self.assertIn("你是助手", system)
        self.assertIn("[长期记忆]", system)
        self.assertIn("- 用户喜欢喝美式咖啡", system)
        self.assertIn("[对话摘要]", system)
        self.assertIn("用户在筹备咖啡店开业。", system)

        self.assertEqual(assembled.history_count, 2)
        self.assertEqual(assembled.memory_count, 1)
        self.assertGreater(assembled.summary_chars, 0)
        self.assertFalse(assembled.truncated)

    def test_minimal_assembly(self) -> None:
        assembled = self.builder.build(persona=self.persona, user_text="在吗")
        self.assertEqual(len(assembled.messages), 2)  # system + user
        self.assertEqual(assembled.memory_count, 0)
        self.assertEqual(assembled.history_count, 0)

    def test_quoted_message_prefix(self) -> None:
        assembled = self.builder.build(
            persona=self.persona, user_text="展开讲讲", quoted_text="之前提到的内容"
        )
        self.assertEqual(
            assembled.messages[-1].content, "（引用消息：之前提到的内容）\n展开讲讲"
        )

    def test_budget_truncates_oldest_history(self) -> None:
        builder = ContextBuilder(ContextConfig(max_chars=60))
        history = [
            ChatMessage(role=ChatRole.USER, content="第一轮" * 10),
            ChatMessage(role=ChatRole.ASSISTANT, content="第一轮答" * 10),
            ChatMessage(role=ChatRole.USER, content="最新问题"),
        ]
        assembled = builder.build(
            persona=self.persona, history=history, user_text="继续"
        )
        self.assertTrue(assembled.truncated)
        # 最新的历史与当前消息必须保留
        self.assertEqual(assembled.messages[-1].content, "继续")
        self.assertEqual(assembled.messages[-2].content, "最新问题")

    def test_disable_memory_and_summary(self) -> None:
        builder = ContextBuilder(
            ContextConfig(include_long_term=False, include_summary=False)
        )
        assembled = builder.build(
            persona=self.persona,
            user_text="hi",
            summary="摘要不应出现",
            memories=[MemoryEntry(content="记忆不应出现")],
        )
        system = assembled.messages[0].content
        self.assertNotIn("摘要不应出现", system)
        self.assertNotIn("记忆不应出现", system)
        self.assertEqual(assembled.memory_count, 0)
        self.assertEqual(assembled.summary_chars, 0)

    def test_to_request_persona_overrides(self) -> None:
        persona = Persona(
            name="p",
            system_prompt="x",
            temperature=0.2,
            model="glm-4-air",
            max_tokens=99,
        )
        assembled = self.builder.build(persona=persona, user_text="hi")
        request = assembled.to_request(
            persona=persona,
            default_model="glm-4-flash",
            default_temperature=0.7,
            default_max_tokens=1024,
        )
        self.assertEqual(request.model, "glm-4-air")
        self.assertEqual(request.temperature, 0.2)
        self.assertEqual(request.max_tokens, 99)

        plain = Persona(name="d", system_prompt="x")
        request2 = self.builder.build(persona=plain, user_text="hi").to_request(
            persona=plain,
            default_model="glm-4-flash",
            default_temperature=0.7,
            default_max_tokens=1024,
        )
        self.assertEqual(request2.model, "glm-4-flash")
        self.assertEqual(request2.temperature, 0.7)
        self.assertEqual(request2.max_tokens, 1024)


# ---------------------------------------------------------------------------
# GLMProvider
# ---------------------------------------------------------------------------


class GLMProviderTest(unittest.IsolatedAsyncioTestCase):
    """GLM Provider：请求构造、响应解析、错误映射、重试。"""

    async def test_chat_success(self) -> None:
        captured: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=glm_ok_response("排序代码给你"))

        provider = make_glm(handler)
        response = await provider.chat(
            ChatRequest(
                messages=[
                    ChatMessage(role=ChatRole.SYSTEM, content="你是助手"),
                    ChatMessage(role=ChatRole.USER, content="写个快排"),
                ],
                temperature=0.5,
            )
        )

        # 请求校验
        request = captured[0]
        self.assertTrue(request.url.path.endswith("/chat/completions"))
        self.assertEqual(request.headers["authorization"], "Bearer test-key")
        payload = json.loads(request.content)
        self.assertEqual(payload["model"], "glm-4-flash")
        self.assertEqual(payload["stream"], False)
        self.assertEqual(payload["temperature"], 0.5)
        self.assertEqual(
            payload["messages"][0], {"role": "system", "content": "你是助手"}
        )
        self.assertEqual(
            payload["messages"][1], {"role": "user", "content": "写个快排"}
        )
        self.assertEqual(payload["max_tokens"], 1024)

        # 响应映射
        self.assertEqual(response.content, "排序代码给你")
        self.assertEqual(response.model, "glm-4-flash")
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.usage.total_tokens, 15)

    async def test_request_overrides_win(self) -> None:
        captured: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=glm_ok_response())

        provider = make_glm(handler, model="glm-4-air", temperature=0.1, max_tokens=50)
        await provider.chat(
            ChatRequest(
                messages=[ChatMessage(role=ChatRole.USER, content="hi")],
                model="glm-4-long",
                temperature=0.9,
                max_tokens=7,
                top_p=0.8,
            )
        )
        payload = json.loads(captured[0].content)
        self.assertEqual(payload["model"], "glm-4-long")
        self.assertEqual(payload["temperature"], 0.9)
        self.assertEqual(payload["max_tokens"], 7)
        self.assertEqual(payload["top_p"], 0.8)

    async def test_missing_api_key_raises_auth_error(self) -> None:
        provider = GLMProvider(GLMConfig(api_key=""))
        with self.assertRaises(ProviderAuthError):
            await provider.chat(
                ChatRequest(messages=[ChatMessage(role=ChatRole.USER, content="hi")])
            )

    async def test_auth_error_not_retried(self) -> None:
        calls = {"n": 0}

        async def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(
                401, json={"error": {"code": "401", "message": "invalid token"}}
            )

        provider = make_glm(handler, max_retries=2)
        with self.assertRaises(ProviderAuthError) as ctx:
            await provider.chat(
                ChatRequest(messages=[ChatMessage(role=ChatRole.USER, content="hi")])
            )
        self.assertEqual(calls["n"], 1)
        self.assertEqual(ctx.exception.status, 401)

    async def test_rate_limit_retries_then_succeeds(self) -> None:
        calls = {"n": 0}

        async def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(
                    429,
                    json={"error": {"code": "1113", "message": "rate limited"}},
                    headers={"Retry-After": "0"},
                )
            return httpx.Response(200, json=glm_ok_response("恢复啦"))

        provider = make_glm(handler, max_retries=2)
        response = await provider.chat(
            ChatRequest(messages=[ChatMessage(role=ChatRole.USER, content="hi")])
        )
        self.assertEqual(response.content, "恢复啦")
        self.assertEqual(calls["n"], 2)

    async def test_rate_limit_exhausted_raises(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                json={"error": {"code": "1113", "message": "rate limited"}},
                headers={"Retry-After": "5"},
            )

        provider = make_glm(handler, max_retries=0)
        with self.assertRaises(ProviderRateLimitError) as ctx:
            await provider.chat(
                ChatRequest(messages=[ChatMessage(role=ChatRole.USER, content="hi")])
            )
        self.assertEqual(ctx.exception.retry_after, 5.0)

    async def test_server_error_retries_then_succeeds(self) -> None:
        calls = {"n": 0}

        async def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] <= 2:
                return httpx.Response(503, json={"error": {"message": "busy"}})
            return httpx.Response(200, json=glm_ok_response("终于好了"))

        provider = make_glm(handler, max_retries=3)
        response = await provider.chat(
            ChatRequest(messages=[ChatMessage(role=ChatRole.USER, content="hi")])
        )
        self.assertEqual(response.content, "终于好了")
        self.assertEqual(calls["n"], 3)

    async def test_bad_request_not_retried(self) -> None:
        calls = {"n": 0}

        async def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(
                400, json={"error": {"code": "1214", "message": "bad"}}
            )

        provider = make_glm(handler, max_retries=3)
        with self.assertRaises(ProviderAPIError) as ctx:
            await provider.chat(
                ChatRequest(messages=[ChatMessage(role=ChatRole.USER, content="hi")])
            )
        self.assertEqual(calls["n"], 1)
        self.assertEqual(ctx.exception.status, 400)

    async def test_timeout_retries(self) -> None:
        calls = {"n": 0}

        async def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.TimeoutException("read timeout", request=request)
            return httpx.Response(200, json=glm_ok_response("超时后恢复"))

        provider = make_glm(handler, max_retries=1)
        response = await provider.chat(
            ChatRequest(messages=[ChatMessage(role=ChatRole.USER, content="hi")])
        )
        self.assertEqual(response.content, "超时后恢复")
        self.assertEqual(calls["n"], 2)

    async def test_legacy_error_format(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"code": "1000", "msg": "内部错误"})

        provider = make_glm(handler, max_retries=0)
        with self.assertRaises(ProviderAPIError) as ctx:
            await provider.chat(
                ChatRequest(messages=[ChatMessage(role=ChatRole.USER, content="hi")])
            )
        self.assertIn("1000", str(ctx.exception))
        self.assertIn("内部错误", str(ctx.exception))

    async def test_empty_choices_raises(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": []})

        provider = make_glm(handler)
        with self.assertRaises(ProviderAPIError):
            await provider.chat(
                ChatRequest(messages=[ChatMessage(role=ChatRole.USER, content="hi")])
            )

    async def test_reasoning_content_fallback(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "reasoning_content": "思考链",
                            },
                        }
                    ]
                },
            )

        provider = make_glm(handler)
        response = await provider.chat(
            ChatRequest(messages=[ChatMessage(role=ChatRole.USER, content="hi")])
        )
        self.assertEqual(response.content, "思考链")

    async def test_from_env_priority(self) -> None:
        env = {
            "GLM_API_KEY": "glm-key",
            "LLM_API_KEY": "llm-key",
            "GLM_MODEL": "glm-4-plus",
            "GLM_BASE_URL": "https://custom.example/v4",
            "GLM_TIMEOUT": "45",
        }
        with patch.dict(os.environ, env, clear=False):
            config = GLMConfig.from_env()
        self.assertEqual(config.api_key, "glm-key")  # GLM_ 优先
        self.assertEqual(config.model, "glm-4-plus")
        self.assertEqual(config.base_url, "https://custom.example/v4")
        self.assertEqual(config.timeout, 45.0)

        with patch.dict(os.environ, {"LLM_API_KEY": "llm-key"}, clear=False):
            os.environ.pop("GLM_API_KEY", None)
            config2 = GLMConfig.from_env()
        self.assertEqual(config2.api_key, "llm-key")

    async def test_close_owned_client(self) -> None:
        provider = GLMProvider(GLMConfig(api_key="k"))
        await provider._get_client()  # noqa: SLF001 - 触发懒创建
        self.assertIsNotNone(provider._client)  # noqa: SLF001
        await provider.close()
        self.assertIsNone(provider._client)  # noqa: SLF001


class ProviderFactoryTest(unittest.TestCase):
    """Provider 工厂。"""

    def test_create_glm(self) -> None:
        provider = create_provider("glm", GLMConfig(api_key="k"))
        self.assertIsInstance(provider, GLMProvider)
        self.assertEqual(provider.name, "glm")

    def test_create_with_dict(self) -> None:
        provider = create_provider("glm", {"api_key": "k", "model": "glm-4-air"})
        self.assertIsInstance(provider, GLMProvider)

    def test_unknown_provider(self) -> None:
        with self.assertRaises(ValueError):
            create_provider("openai")

    def test_available_providers(self) -> None:
        self.assertEqual(available_providers(), ["glm"])


# ---------------------------------------------------------------------------
# AIService
# ---------------------------------------------------------------------------


def make_service(
    provider: LLMProvider | None,
    tmp: str,
    *,
    summarizer=None,
    **config_kwargs,
) -> AIService:
    """在临时目录上装配 AIService（personas + memory 均隔离）。"""
    root = Path(tmp)
    write_persona(
        root,
        "default.json",
        {"name": "default", "systemPrompt": "你是 ClawHub", "greeting": "你好呀"},
    )
    write_persona(
        root,
        "programmer.json",
        {
            "name": "程序员",
            "systemPrompt": "你是工程师",
            "greeting": "你好，我是你的编程助手",
            "temperature": 0.5,
        },
    )
    config = AIServiceConfig(
        personas_dir=root,
        memory_path=root / "ai_memory.json",
        glm=GLMConfig(api_key="fake", retry_base_delay=0.0),
        **config_kwargs,
    )
    kwargs = {}
    if summarizer is not None:
        kwargs["summarizer"] = summarizer
    return AIService(provider, config=config, **kwargs)


class AIServiceChatFlowTest(unittest.IsolatedAsyncioTestCase):
    """AI 对话主流程。"""

    async def test_full_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeProvider(["你好，有什么可以帮你？"])
            service = make_service(provider, tmp)
            sink = FakeSink()

            reply = await service.handle_message(make_message("在吗"), sink)

            self.assertEqual(reply, "你好，有什么可以帮你？")
            self.assertEqual(sink.sent, [("u1@im.wechat", "你好，有什么可以帮你？")])
            self.assertTrue(sink.typing_used)

            # 短期记忆记录了一轮对话
            history = service.conversation.get("user:u1@im.wechat", "default")
            self.assertEqual(len(history), 2)
            self.assertEqual(history[0].role, ChatRole.USER)
            self.assertEqual(history[1].content, "你好，有什么可以帮你？")

            # 请求里带上了角色 system prompt
            request = provider.requests[0]
            self.assertEqual(request.messages[0].role, ChatRole.SYSTEM)
            self.assertIn("ClawHub", request.messages[0].content)
            self.assertEqual(request.messages[-1].content, "在吗")

    async def test_multi_turn_history_grows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeProvider(["第一答", "第二答"])
            service = make_service(provider, tmp)
            await service.handle_message(make_message("问题一"))
            await service.handle_message(make_message("问题二"))

            history = service.conversation.get("user:u1@im.wechat", "default")
            self.assertEqual(len(history), 4)
            # 第二轮的请求包含第一轮历史
            second_request = provider.requests[1]
            contents = [m.content for m in second_request.messages]
            self.assertIn("问题一", contents)
            self.assertIn("第一答", contents)
            self.assertEqual(second_request.messages[-1].content, "问题二")

    async def test_long_term_memory_extract_and_recall(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeProvider(["记住了", "推荐美式"])
            service = make_service(provider, tmp)

            await service.handle_message(make_message("记住我喜欢喝美式咖啡"))
            entries = service.long_term.entries("user:u1@im.wechat")
            self.assertEqual([e.content for e in entries], ["我喜欢喝美式咖啡"])

            # 第二轮相关问题：长期记忆被注入 system prompt
            await service.handle_message(make_message("咖啡推荐？"))
            system_prompt = provider.requests[-1].messages[0].content
            self.assertIn("[长期记忆]", system_prompt)
            self.assertIn("我喜欢喝美式咖啡", system_prompt)

    async def test_non_text_message_reply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(FakeProvider(), tmp)
            sink = FakeSink()
            reply = await service.handle_message(
                make_message(kind=ContentKind.IMAGE), sink
            )
            self.assertEqual(
                reply, "我目前只能理解文本消息～图片、语音、文件还看不来。"
            )
            self.assertEqual(len(sink.sent), 1)

    async def test_unconfigured_provider_reply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(None, tmp)
            reply = await service.handle_message(make_message("在吗"))
            self.assertIn("GLM_API_KEY", reply)

    async def test_provider_error_friendly_reply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(
                FakeProvider(error=ProviderAPIError("boom", status=500)), tmp
            )
            reply = await service.handle_message(make_message("在吗"))
            self.assertIn("开小差", reply)
            # 失败的轮次不写入短期记忆
            self.assertEqual(
                service.conversation.get("user:u1@im.wechat", "default"), []
            )

    async def test_empty_reply_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(FakeProvider(["   "]), tmp)
            reply = await service.handle_message(make_message("在吗"))
            self.assertIn("开小差", reply)

    async def test_quoted_message_included(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeProvider(["收到"])
            service = make_service(provider, tmp)
            await service.handle_message(
                make_message("展开讲讲", quoted_text="之前的消息")
            )
            user_message = provider.requests[0].messages[-1]
            self.assertEqual(user_message.content, "（引用消息：之前的消息）\n展开讲讲")

    async def test_summary_compression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeProvider([f"答{i}" for i in range(4)])
            summarizer = FakeSummarizer("早期对话摘要")
            service = make_service(
                provider,
                tmp,
                summarizer=summarizer,
                compress_threshold=2,
                keep_recent=1,
            )
            for i in range(3):
                await service.handle_message(make_message(f"问题{i}"))

            self.assertTrue(summarizer.calls)  # 摘要器被调用
            self.assertEqual(
                service.conversation.summary("user:u1@im.wechat", "default"),
                "早期对话摘要",
            )
            # 最后一轮请求的 system prompt 带摘要
            self.assertIn("早期对话摘要", provider.requests[-1].messages[0].content)


class AIServicePersonaCommandTest(unittest.IsolatedAsyncioTestCase):
    """/role 指令与角色切换。"""

    async def test_role_set_and_switch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(FakeProvider(), tmp)
            sink = FakeSink()

            reply = await service.handle_command(
                "role", ["set", "程序员"], make_message("/role set 程序员"), sink
            )
            self.assertEqual(reply, "已切换到角色「程序员」。你好，我是你的编程助手")
            self.assertEqual(
                service.current_persona_name("user:u1@im.wechat"), "程序员"
            )
            self.assertEqual(sink.sent[0][1], reply)

            # 按文件名切换
            reply = await service.handle_command(
                "role", ["set", "default"], make_message(), None
            )
            self.assertIn("已切换到角色「default」", reply)

    async def test_role_set_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(FakeProvider(), tmp)
            reply = await service.handle_command(
                "role", ["set", "黑客"], make_message(), None
            )
            self.assertIn("不存在", reply)
            self.assertIn("程序员", reply)  # 列出可用角色

    async def test_role_set_missing_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(FakeProvider(), tmp)
            reply = await service.handle_command("role", ["set"], make_message(), None)
            self.assertIn("用法", reply)

    async def test_role_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(FakeProvider(), tmp)
            reply = await service.handle_command("role", ["list"], make_message(), None)
            self.assertIn("default", reply)
            self.assertIn("程序员", reply)
            self.assertIn("← 当前", reply)  # default 是当前角色

    async def test_role_current_and_info(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(FakeProvider(), tmp)
            reply = await service.handle_command("role", [], make_message(), None)
            self.assertIn("当前角色：「default」", reply)

            reply = await service.handle_command(
                "role", ["info", "程序员"], make_message(), None
            )
            self.assertIn("你是工程师", reply)
            self.assertIn("温度：0.5", reply)

    async def test_role_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(FakeProvider(), tmp)
            # 运行中新增角色文件
            write_persona(
                Path(tmp), "poet.json", {"name": "诗人", "systemPrompt": "写诗"}
            )
            reply = await service.handle_command(
                "role", ["reload"], make_message(), None
            )
            self.assertIn("已重载角色配置，共 3 个角色", reply)
            self.assertTrue(service.personas.exists("诗人"))

    async def test_role_unknown_subcommand(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(FakeProvider(), tmp)
            reply = await service.handle_command(
                "role", ["dance"], make_message(), None
            )
            self.assertIn("用法", reply)

    async def test_persona_contexts_independent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeProvider(["答A", "答B"])
            service = make_service(provider, tmp)
            session = "user:u1@im.wechat"

            await service.handle_message(make_message("默认角色的问题"))
            await service.handle_command(
                "role", ["set", "程序员"], make_message(), None
            )
            await service.handle_message(make_message("程序员的问题"))

            self.assertEqual(len(service.conversation.get(session, "default")), 2)
            self.assertEqual(len(service.conversation.get(session, "程序员")), 2)

            # 程序员角色的请求使用程序员的 system prompt 与温度
            request = provider.requests[-1]
            self.assertIn("你是工程师", request.messages[0].content)
            self.assertEqual(request.temperature, 0.5)

    async def test_persona_state_per_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(FakeProvider(), tmp)
            await service.handle_command(
                "role",
                ["set", "程序员"],
                make_message(session_key="user:u1@im.wechat"),
                None,
            )
            # 另一个会话不受影响
            self.assertEqual(
                service.current_persona_name("user:u2@im.wechat"), "default"
            )


class AIServiceConfigTest(unittest.TestCase):
    """服务配置默认值。"""

    def test_defaults(self) -> None:
        config = AIServiceConfig()
        self.assertEqual(config.default_persona, "default")
        self.assertEqual(config.history_limit, 40)
        self.assertTrue(config.enable_long_term_memory)
        self.assertTrue(config.enable_auto_facts)
        self.assertEqual(config.glm.model, "glm-4-flash")


if __name__ == "__main__":
    unittest.main()
