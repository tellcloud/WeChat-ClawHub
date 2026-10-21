"""记忆系统：管理短期上下文和长期记忆。

对应 README 目录结构中的 ``modules/ai/memory.py``，覆盖任务树 E07：

- **短期上下文**（:class:`ConversationMemory`）：按
  ``(session_key, persona_name)`` 独立维护的滑动窗口——切换角色不丢
  另一个角色的上下文（README「按角色独立维护对话上下文」）；窗口溢出前
  先把更早的消息交给摘要器压缩成一段「对话摘要」，避免硬截断丢失信息；
- **长期记忆**（:class:`LongTermMemory`）：跨会话持久化的用户事实
  （称呼、偏好、项目背景等），JSON 文件存储（``Local/ai_memory.json``），
  关键词检索（中文二元组 + 英文单词），写入时自动合并相似条目；
- **摘要器**（:class:`LLMSummarizer` / :class:`NullSummarizer`）：
  短期记忆压缩的 LLM 实现与空实现（离线测试用）；
- **事实抽取**（:class:`HeuristicFactExtractor`）：从用户消息中启发式
  提取值得长期记住的事实（「记住…」「我叫…」「我喜欢…」等）。

所有结构均可离线运行：不配置 LLM 时摘要退化为空串、抽取走启发式规则。
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from .providers.base import ChatMessage, ChatRequest, LLMProvider

logger = logging.getLogger(__name__)

#: 长期记忆默认存储位置（Local/ 已被 .gitignore 忽略）
DEFAULT_MEMORY_PATH = Path("Local/ai_memory.json")

#: 短期记忆默认窗口长度（消息条数）
DEFAULT_HISTORY_LIMIT = 40

#: 摘要指令：要求客观、紧凑、保留关键事实
SUMMARY_PROMPT = (
    "请把下面的对话历史压缩成一段简洁的摘要，供后续对话参考。"
    "要求：保留用户的关键信息（称呼、偏好、需求、已确认的结论），"
    "丢弃寒暄与重复内容；用第三人称客观陈述；不超过 150 字；直接输出摘要本身。"
)


# ---------------------------------------------------------------------------
# 短期上下文
# ---------------------------------------------------------------------------


@runtime_checkable
class MemorySummarizer(Protocol):
    """摘要器协议：把一段历史消息压缩为一段文本。"""

    async def summarize(self, messages: list[ChatMessage]) -> str: ...


class NullSummarizer:
    """空摘要器：不压缩（离线 / 未配置 LLM 时使用）。"""

    async def summarize(self, messages: list[ChatMessage]) -> str:
        return ""


class LLMSummarizer:
    """基于 LLM 的对话摘要器。"""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        max_tokens: int = 200,
        prompt: str = SUMMARY_PROMPT,
        transcript_limit: int = 6000,
    ) -> None:
        self.provider = provider
        self.max_tokens = max_tokens
        self.prompt = prompt
        self.transcript_limit = transcript_limit

    async def summarize(self, messages: list[ChatMessage]) -> str:
        """调用 LLM 压缩历史消息；失败时返回空串（不阻塞主流程）。"""
        if not messages:
            return ""
        transcript = format_transcript(messages)
        if len(transcript) > self.transcript_limit:
            transcript = transcript[-self.transcript_limit :]
        request = ChatRequest(
            messages=[
                ChatMessage(role="system", content=self.prompt),
                ChatMessage(role="user", content=transcript),
            ],
            max_tokens=self.max_tokens,
            temperature=0.3,
        )
        try:
            response = await self.provider.chat(request)
        except Exception:
            logger.exception("对话摘要失败，本次跳过压缩")
            return ""
        return response.content.strip()


def format_transcript(messages: list[ChatMessage]) -> str:
    """把消息列表转写为 ``用户/助手: 内容`` 多行文本。"""
    labels = {"user": "用户", "assistant": "助手", "system": "系统"}
    lines = []
    for message in messages:
        label = labels.get(message.role.value, message.role.value)
        content = message.content.strip()
        if content:
            lines.append(f"{label}: {content}")
    return "\n".join(lines)


@dataclass
class _ConversationBuffer:
    """单个 (会话, 角色) 的对话缓冲。"""

    messages: deque[ChatMessage] = field(default_factory=deque)
    summary: str = ""


class ConversationMemory:
    """短期上下文：按 (会话, 角色) 独立的滑动窗口。

    用法::

        memory = ConversationMemory(max_messages=40)
        memory.add("user:u1", "default", ChatMessage(role="user", content="hi"))
        history = memory.get("user:u1", "default")

    :param max_messages: 每个 (会话, 角色) 最多保留的消息条数。
    """

    def __init__(self, max_messages: int = DEFAULT_HISTORY_LIMIT) -> None:
        if max_messages < 2:
            raise ValueError("max_messages 至少为 2")
        self.max_messages = max_messages
        self._buffers: dict[tuple[str, str], _ConversationBuffer] = {}

    # -- 读写 ---------------------------------------------------------------

    def add(self, session_key: str, persona_name: str, message: ChatMessage) -> None:
        """追加一条消息（超出窗口时丢弃最旧的）。"""
        buffer = self._buffer(session_key, persona_name)
        buffer.messages.append(message)

    def get(self, session_key: str, persona_name: str) -> list[ChatMessage]:
        """当前历史消息副本（不含摘要）。"""
        buffer = self._buffers.get((session_key, persona_name.lower()))
        return list(buffer.messages) if buffer else []

    def summary(self, session_key: str, persona_name: str) -> str:
        """已压缩的对话摘要（无则为空串）。"""
        buffer = self._buffers.get((session_key, persona_name.lower()))
        return buffer.summary if buffer else ""

    def set_summary(self, session_key: str, persona_name: str, text: str) -> None:
        """直接设置摘要（恢复 / 测试用）。"""
        self._buffer(session_key, persona_name).summary = text

    # -- 压缩 ---------------------------------------------------------------

    async def compress(
        self,
        session_key: str,
        persona_name: str,
        keep_recent: int,
        summarizer: MemorySummarizer,
    ) -> str:
        """把 ``keep_recent`` 之前的消息压缩为摘要。

        被压缩的消息从窗口中移除，摘要文本合并进既有摘要。
        :returns: 本次新增的摘要片段（无压缩时为空串）。
        """
        buffer = self._buffers.get((session_key, persona_name.lower()))
        if buffer is None or len(buffer.messages) <= keep_recent:
            return ""

        drop_count = len(buffer.messages) - keep_recent
        old = [buffer.messages.popleft() for _ in range(drop_count)]
        if not old:
            return ""

        digest = await summarizer.summarize(old)
        if digest:
            buffer.summary = (
                f"{buffer.summary}\n{digest}" if buffer.summary else digest
            )
            logger.info(
                "对话已压缩: session=%s persona=%s 移除 %d 条，摘要 %d 字",
                session_key,
                persona_name,
                drop_count,
                len(digest),
            )
        return digest

    # -- 清理 ---------------------------------------------------------------

    def clear(self, session_key: str, persona_name: Optional[str] = None) -> int:
        """清除会话历史（指定角色或全部角色），返回清除的消息条数。"""
        if persona_name is not None:
            return self._clear_one(session_key, persona_name)
        keys = [key for key in self._buffers if key[0] == session_key]
        return sum(self._clear_key(key) for key in keys)

    def clear_all(self) -> None:
        """清除全部历史。"""
        self._buffers.clear()

    # -- 统计 ---------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        """各 (会话, 角色) 的消息条数（调试 / 测试用）。"""
        return {
            f"{session}|{persona}": len(buffer.messages)
            for (session, persona), buffer in self._buffers.items()
        }

    # -- 内部 ---------------------------------------------------------------

    def _buffer(self, session_key: str, persona_name: str) -> _ConversationBuffer:
        key = (session_key, persona_name.lower())
        buffer = self._buffers.get(key)
        if buffer is None:
            buffer = _ConversationBuffer(messages=deque(maxlen=self.max_messages))
            self._buffers[key] = buffer
        return buffer

    def _clear_one(self, session_key: str, persona_name: str) -> int:
        return self._clear_key((session_key, persona_name.lower()))

    def _clear_key(self, key: tuple[str, str]) -> int:
        buffer = self._buffers.pop(key, None)
        return len(buffer.messages) if buffer else 0


# ---------------------------------------------------------------------------
# 长期记忆
# ---------------------------------------------------------------------------


class MemoryEntry(BaseModel):
    """一条长期记忆。"""

    content: str = Field(description="记忆内容（一句事实）")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="创建 / 更新时间",
    )
    source: str = Field(default="", description="来源消息 ID（追溯用）")
    hits: int = Field(default=0, description="被检索命中的次数")


class LongTermMemory:
    """长期记忆：按会话持久化的用户事实库。

    - 存储：JSON 文件（默认 ``Local/ai_memory.json``），原子写入；
    - 写入：自动去重 / 合并相似条目（相似度超过 ``merge_threshold`` 时
      以新内容替换旧条目，即 README 的「自动合并」）；
    - 检索：中文二元组 + 英文单词的关键词重合度打分，兼顾命中次数加权。

    用法::

        memory = LongTermMemory(Path("Local/ai_memory.json"))
        memory.remember("user:u1", "用户喜欢喝美式咖啡")
        facts = memory.recall("user:u1", "咖啡", limit=3)

    :param store_path: 持久化文件；``None`` 表示仅内存运行；
    :param max_entries_per_session: 单会话记忆条数上限（FIFO 淘汰）；
    :param merge_threshold: 相似度合并阈值（0~1）。
    """

    VERSION = 1

    def __init__(
        self,
        store_path: Optional[Path] = None,
        *,
        max_entries_per_session: int = 100,
        merge_threshold: float = 0.75,
    ) -> None:
        self.store_path = Path(store_path) if store_path else None
        self.max_entries_per_session = max(1, max_entries_per_session)
        self.merge_threshold = min(max(merge_threshold, 0.0), 1.0)
        self._sessions: dict[str, list[MemoryEntry]] = {}
        self._dirty = False
        self._load()

    # -- 写入 ---------------------------------------------------------------

    def remember(
        self,
        session_key: str,
        content: str,
        *,
        source: str = "",
    ) -> tuple[MemoryEntry, bool]:
        """记住一条事实。

        :returns: ``(条目, 是否合并到已有条目)``；内容为空时返回
            ``(None, False)`` 意义上的空操作——实际返回已有条目或新条目。
        """
        text = _normalize_text(content)
        if not text:
            raise ValueError("记忆内容不能为空")

        entries = self._sessions.setdefault(session_key, [])
        tokens = _tokenize(text)

        # 完全重复：仅刷新时间戳
        for entry in entries:
            if _normalize_text(entry.content) == text:
                entry.created_at = datetime.now(timezone.utc)
                if source:
                    entry.source = source
                self._dirty = True
                self._autosave()
                return entry, True

        # 相似合并：替换最相似的旧条目
        best: Optional[tuple[float, MemoryEntry]] = None
        for entry in entries:
            score = _similarity(tokens, _tokenize(entry.content))
            if best is None or score > best[0]:
                best = (score, entry)
        if best is not None and best[0] >= self.merge_threshold:
            entry = best[1]
            entry.content = text
            entry.created_at = datetime.now(timezone.utc)
            entry.source = source or entry.source
            self._dirty = True
            self._autosave()
            logger.info("长期记忆已合并: %r → %r", entry.content, text)
            return entry, True

        entry = MemoryEntry(content=text, source=source)
        entries.append(entry)
        # FIFO 淘汰
        while len(entries) > self.max_entries_per_session:
            entries.pop(0)
        self._dirty = True
        self._autosave()
        return entry, False

    def forget(self, session_key: str, content: Optional[str] = None) -> int:
        """删除记忆。

        :param content: 指定内容（相似度匹配，删最相似的一条）；
            ``None`` 表示清空该会话的全部记忆。
        :returns: 删除的条数。
        """
        entries = self._sessions.get(session_key)
        if not entries:
            return 0
        if content is None:
            removed = len(entries)
            del self._sessions[session_key]
        else:
            target = _normalize_text(content)
            index = _find_similar(entries, target, threshold=0.6)
            if index is None:
                return 0
            entries.pop(index)
            removed = 1
            if not entries:
                del self._sessions[session_key]
        self._dirty = True
        self._autosave()
        return removed

    # -- 检索 ---------------------------------------------------------------

    def recall(
        self,
        session_key: str,
        query: str = "",
        *,
        limit: int = 3,
    ) -> list[MemoryEntry]:
        """检索与 ``query`` 相关的记忆（按相关度排序）。

        query 为空时返回最近的记忆。命中的条目会累加 ``hits``。
        """
        entries = self._sessions.get(session_key) or []
        if not entries:
            return []

        if not query.strip():
            selected = sorted(entries, key=lambda e: e.created_at, reverse=True)[:limit]
        else:
            query_tokens = _tokenize(query)
            scored: list[tuple[float, MemoryEntry]] = []
            for entry in entries:
                score = _similarity(query_tokens, _tokenize(entry.content))
                if score > 0:
                    # 命中次数与新鲜度加权
                    score += 0.02 * min(entry.hits, 10)
                    scored.append((score, entry))
            scored.sort(key=lambda item: (-item[0], -item[1].created_at.timestamp()))
            selected = [entry for _, entry in scored[:limit]]

        changed = False
        for entry in selected:
            entry.hits += 1
            changed = True
        if changed:
            self._dirty = True
            self._autosave()
        return selected

    def entries(self, session_key: str) -> list[MemoryEntry]:
        """该会话全部记忆（按时间正序）。"""
        return list(self._sessions.get(session_key) or [])

    def session_count(self) -> int:
        """有记忆的会话数。"""
        return len(self._sessions)

    def clear(self) -> None:
        """清空全部会话的记忆。"""
        self._sessions.clear()
        self._dirty = True
        self._autosave()

    # -- 持久化 -------------------------------------------------------------

    def save(self) -> None:
        """立即写盘（无存储路径时为无操作）。"""
        if self.store_path is None:
            return
        payload = {
            "version": self.VERSION,
            "sessions": {
                session: [entry.model_dump(mode="json") for entry in entries]
                for session, entries in self._sessions.items()
            },
        }
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.store_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp_path, self.store_path)  # 原子替换，避免写一半损坏
        self._dirty = False

    def _autosave(self) -> None:
        if self.store_path is not None:
            self.save()

    def _load(self) -> None:
        """从磁盘加载（文件缺失 / 损坏时从空库开始）。"""
        if self.store_path is None or not self.store_path.is_file():
            return
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "长期记忆文件读取失败，从空库开始: %s: %s", self.store_path, exc
            )
            return
        sessions = data.get("sessions") if isinstance(data, dict) else None
        if not isinstance(sessions, dict):
            return
        for session_key, items in sessions.items():
            if not isinstance(items, list):
                continue
            entries: list[MemoryEntry] = []
            for item in items:
                try:
                    entries.append(MemoryEntry.model_validate(item))
                except Exception:
                    logger.warning(
                        "跳过无效记忆条目: session=%s item=%r", session_key, item
                    )
            if entries:
                self._sessions[str(session_key)] = entries
        logger.info(
            "长期记忆已加载: %d 个会话，共 %d 条",
            len(self._sessions),
            sum(len(v) for v in self._sessions.values()),
        )


# ---------------------------------------------------------------------------
# 事实抽取
# ---------------------------------------------------------------------------


@runtime_checkable
class FactExtractor(Protocol):
    """事实抽取器协议：从用户消息中提取值得长期记住的事实。"""

    def extract(self, text: str) -> list[str]: ...


#: 触发长期记忆的模式（捕获组名统一为 fact）。
#: 顺序即优先级：先出现的模式更具体，命中后跳过与之重叠的泛化匹配。
_FACT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:请?记住|记一下|记好|别忘了|不要忘记|帮忙记住)[：:，,\s]*(?P<fact>.+)"),
    re.compile(r"(?:我的名字是|我叫|我的名叫)[：:，,\s]*(?P<fact>.+)"),
    re.compile(r"我(?:非常)?(?:喜欢|钟爱|爱)(?P<fact>.+)"),
    re.compile(r"我(?:不太)?(?:讨厌|不喜欢)(?P<fact>.+)"),
    re.compile(r"我的(?P<fact>\S{1,12}?(?:是|为).+)"),
    re.compile(r"remember[:\s]+(?P<fact>.+)", re.IGNORECASE),
    re.compile(r"my name is[:\s]+(?P<fact>.+)", re.IGNORECASE),
    re.compile(r"i (?:really )?(?:like|love)[:\s]+(?P<fact>.+)", re.IGNORECASE),
)

#: 句尾标点（抽取后剥除）
_TRAILING_PUNCT = "。！？!?，,；;．.…~～ \t\n\r"

#: 事实长度边界
_MIN_FACT_LEN = 2
_MAX_FACT_LEN = 120


class HeuristicFactExtractor:
    """启发式事实抽取器（离线可用，无需 LLM）。

    识别「记住…」「我叫…」「我喜欢…」「我的 X 是 Y」等模式；
    疑问句不抽取（问句不是事实）。
    """

    def extract(self, text: str) -> list[str]:
        """从一条用户消息中提取事实列表（通常 0~2 条）。"""
        stripped = text.strip()
        if not stripped or stripped.endswith(("？", "?")):
            return []  # 疑问句不构成事实

        facts: list[str] = []
        accepted_spans: list[tuple[int, int]] = []
        for pattern in _FACT_PATTERNS:
            for match in pattern.finditer(stripped):
                span = match.span()
                # 跳过与已接受的更具体模式重叠的泛化匹配
                # （「记住我喜欢喝咖啡」只产出「我喜欢喝咖啡」，不再产「喝咖啡」）
                if any(
                    span[0] < end and start < span[1]
                    for start, end in accepted_spans
                ):
                    continue
                fact = self._clean(match.group("fact"))
                if not fact or fact in facts:
                    continue
                facts.append(fact)
                accepted_spans.append(span)
        return facts

    @staticmethod
    def _clean(raw: str) -> str:
        """清洗捕获到的事实文本。"""
        fact = raw.strip().strip(_TRAILING_PUNCT).strip()
        fact = _normalize_text(fact)
        if len(fact) < _MIN_FACT_LEN or len(fact) > _MAX_FACT_LEN:
            return ""
        return fact


# ---------------------------------------------------------------------------
# 文本工具
# ---------------------------------------------------------------------------

#: 英文 / 数字词
_WORD_RE = re.compile(r"[a-z0-9]+")

#: CJK 字符段
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def _normalize_text(text: str) -> str:
    """折叠空白（长期记忆比较对空白不敏感）。"""
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> set[str]:
    """中文二元组 + 英文单词的分词（免第三方依赖，覆盖中英混合）。"""
    lowered = text.lower()
    tokens: set[str] = set(_WORD_RE.findall(lowered))
    for run in _CJK_RE.findall(lowered):
        if len(run) == 1:
            tokens.add(run)
        else:
            tokens.update(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def _similarity(a: set[str], b: set[str]) -> float:
    """两个词集合的余弦相似度（0~1）。"""
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    if overlap == 0:
        return 0.0
    return overlap / math.sqrt(len(a) * len(b))


def _find_similar(
    entries: list[MemoryEntry], target: str, *, threshold: float
) -> Optional[int]:
    """找到与 ``target`` 最相似且超过阈值的条目下标。

    评分 = 余弦相似度 + 0.5 × 包含度（query 词被条目覆盖的比例），
    短查询（如「咖啡」）也能命中长条目。
    """
    tokens = _tokenize(target)
    if not tokens:
        return None
    best_index: Optional[int] = None
    best_score = threshold
    for index, entry in enumerate(entries):
        entry_tokens = _tokenize(entry.content)
        score = _similarity(tokens, entry_tokens)
        containment = len(tokens & entry_tokens) / len(tokens)
        score += 0.5 * containment
        if score >= best_score:
            best_score = score
            best_index = index
    return best_index
