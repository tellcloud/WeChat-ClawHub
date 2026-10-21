# main.py将作为唯一的项目入口文件
"""WeChat-ClawHub 唯一入口：生产运行 + messaging 离线测试。

两种用法：

1. 生产模式（默认）：扫码登录 → 长轮询收消息 → messaging 流水线
   （解析 → 标准化 → 路由）→ 按目标分发。Script 指令由
   :class:`~modules.script.ScriptService` 真实执行（沙箱子进程，
   ``scripts/`` 目录自动扫描注册）；AI 模块已就绪（GLM 后端，
   角色 / 记忆 / 上下文完整闭环，``/role`` 指令归 AI 模块执行）；
   Agent 模块就绪前回复「暂未完成Agent模块的开发工作」；IGNORE 不回复。

       python main.py                # 扫码登录 + 长轮询（Ctrl-C 退出）
       python main.py --once         # 只登录并轮询一次
       python main.py --reset        # 清除登录态，强制重新扫码
       python main.py --debug        # DEBUG 日志

   AI 对话需要智谱 API Key：设置环境变量 ``GLM_API_KEY`` 后生效
   （复制 ``.env.example`` 为 ``.env`` 并填入，每次运行自动加载，
   见 :data:`ENV_FILE`）；未配置时 AI 模块会回复引导配置的提示
   （角色切换等本地功能不受影响）。

2. messaging 离线测试（无需登录 / 网络，直接构造微信原始消息）：

       python main.py --test-messaging                  # 内置场景套件
       python main.py --test-messaging "/weather 北京"   # 测试单条消息
       python main.py --test-messaging -i               # 交互式 REPL

   场景套件覆盖：指令精确匹配、前缀 / 关键词模糊匹配、未知指令兜底、
   普通对话走 AI（离线假 Provider）、群聊 @ 策略、粘性智能体、
   图片 / 语音等非文本消息，以及脚本指令的真实执行（``/hello``
   沙箱子进程、``/say`` 内置处理器、``/hello boom`` 失败路径），
   并校验每种路由目标实际产生的回复文本。

   注意：Git Bash 会把 ``/weather`` 这类参数转换成 Windows 路径，
   已内置自动还原；也可以 ``MSYS_NO_PATHCONV=1 python main.py ...`` 规避。

登录成功后凭据保存在 Local/wechat_credentials.json（.gitignore 已忽略），
之后重复运行无需再次扫码；会话过期（errcode=-14）会自动重新扫码恢复。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
import unicodedata
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv

# src 布局：messaging / transport 物理上位于 src/ 下，将 src/ 加入 sys.path
# 使其可以顶层包名导入（pip install -e . 安装时由 editable 钩子完成）。
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

#: 项目根 .env：由 python-dotenv 在入口 import 时自动加载，之后各
#: ``*Config.from_env()``（WeChatConfig / GLMConfig / AIServiceConfig 等）
#: 读取 ``os.environ`` 时即可获得其中的配置。
#: 规则：已存在的系统环境变量优先，.env 仅作缺省值；文件缺失时静默跳过。
ENV_FILE = Path(__file__).resolve().parent / ".env"
_DOTENV_LOADED = load_dotenv(ENV_FILE, encoding="utf-8")

from modules.ai import (
    AIService,
    AIServiceConfig,
    GLMConfig,
    GLMProvider,
    LLMProvider,
    ProviderConfig,
)
from modules.ai.providers.base import ChatRequest, ChatResponse, Usage
from messaging import (
    MatchType,
    MessageNormalizer,
    MessageParser,
    MessageRouter,
    RouterConfig,
    RouteTarget,
    RoutingResult,
)
from messaging.router import CommandRegistry, CommandSpec
from modules.script import (
    ExecutionContext,
    ScriptConfig,
    ScriptService,
    not_found_reply,
)
from transport.wechat import (
    AuthManager,
    ImageItem,
    MessageItem,
    MessageItemType,
    MessagePoller,
    MessageSender,
    MessageType,
    TextItem,
    VoiceItem,
    WeChatClient,
    WeChatConfig,
    WeixinMessage,
    save_qrcode_image,
)

#: 本地数据目录（.gitignore 已忽略，不会提交）
LOCAL_DIR = Path(__file__).resolve().parent / "Local"
QR_IMAGE = LOCAL_DIR / "login_qrcode.png"
#: 用户脚本目录（``*.py`` 自动扫描注册为指令，见 modules/script）
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

if _DOTENV_LOADED:
    logger.info("已加载环境变量文件: %s（已设置的系统环境变量优先）", ENV_FILE)

#: 路由目标 → 模块未就绪提示（对应服务未注入时的兜底）。
#: AI 模块已就绪（GLM 后端，见 modules/ai/），由注入的 AIService 处理；
#: Script 模块已就绪（沙箱执行，见 modules/script），由注入的
#: ScriptService 处理；两者未注入时 AI / SCRIPT 目标走该映射兜底。
#: Agent 模块就绪前统一回复该文本；IGNORE 不回复。
MODULE_NOT_READY_REPLIES: dict[RouteTarget, str] = {
    RouteTarget.SCRIPT: "暂未完成Script模块的开发工作",
    RouteTarget.AI: "暂未完成AI模块的开发工作",
    RouteTarget.AGENT: "暂未完成Agent模块的开发工作",
}

#: AI 模块接管的指令名（README「AI 角色对话」中的 /role 系列）
AI_COMMANDS = frozenset({"role"})


class OfflineLLMProvider(LLMProvider):
    """离线测试用 Provider：不发起网络请求，回显最后一条用户消息。

    供 --test-messaging 场景套件与 transport 测试使用，
    保证 AI 流程（角色 / 记忆 / 上下文 / 回复出口）可离线断言。
    """

    def __init__(self) -> None:
        super().__init__(ProviderConfig(api_key="offline"))
        #: 记录收到的请求（断言上下文组装用）
        self.requests: list[ChatRequest] = []

    @property
    def name(self) -> str:
        return "offline"

    async def _chat_once(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        last_user = next(
            (m.content for m in reversed(request.messages) if m.role.value == "user"),
            "",
        )
        return ChatResponse(
            content=f"（离线测试）已收到：{last_user}",
            model="offline",
            usage=Usage(total_tokens=1),
        )


def build_ai_service(provider: LLMProvider | None = None) -> AIService:
    """装配 AI 服务。

    生产（``provider=None``）：读取 ``GLM_API_KEY`` 等环境变量构造 GLM
    Provider；未配置 Key 时 Provider 为 ``None``，AI 对话会回复引导配置
    的提示（角色切换等本地功能仍可用）。

    测试：传入 :class:`OfflineLLMProvider` 等假 Provider 即可完整离线运行。
    """
    if provider is None:
        glm_config = GLMConfig.from_env()
        provider = GLMProvider(glm_config) if glm_config.api_key else None
    service = AIService(provider=provider, config=AIServiceConfig.from_env())
    logger.info(
        "AI 服务已装配: provider=%s personas=%s",
        service.provider.name if service.provider else "未配置（缺少 GLM_API_KEY）",
        service.config.personas_dir,
    )
    return service


def build_script_service(
    scripts_dir: Path | None = None, *, demo: bool = False
) -> ScriptService:
    """装配 Script 服务：扫描脚本目录 + 未就绪模块指令桥接。

    - 目录中的 ``*.py`` 按 ``# script:`` 元信息自动注册为指令；
    - ``/agent`` 等已知但归属未就绪模块的指令返回占位提示
      （``/role`` 由 AI 模块接管，不在此列）；
    - 沙箱工作目录位于 ``Local/script_runs``（.gitignore 已忽略）。

    :param scripts_dir: 脚本目录，默认项目根下的 ``scripts/``；
    :param demo: 是否注册离线演示指令 ``/say``（复读参数）。
    """
    directory = scripts_dir or SCRIPTS_DIR
    service = ScriptService.from_directory(
        directory,
        config=ScriptConfig(
            scripts_dir=directory,
            work_root=LOCAL_DIR / "script_runs",
        ),
        unimplemented={"agent": MODULE_NOT_READY_REPLIES[RouteTarget.AGENT]},
    )
    if demo:
        service.registry.register_handler(
            "say",
            lambda args, ctx: " ".join(args),
            keywords=("复读",),
            help_text="/say <文本> — 原样回复（演示引号参数）",
            origin="test",
        )
    logger.info(
        "Script 服务已装配: 目录=%s 指令=%s",
        directory,
        [spec.name for spec in service.registry.commands()],
    )
    return service


def wire_script_service(
    router: MessageRouter, script_service: ScriptService
) -> None:
    """把 Script 指令同步进路由注册表，并补充帮助菜单行。

    两步：

    1. ``script_service.register_into(router.registry)``：让路由层的
       三级匹配认识脚本指令（精确 / 前缀 / 关键词）；
    2. 把路由表中不属于 Script 模块的命令（如 ``/role`` / ``/agent``）
       帮助文本写入 Script 帮助菜单，保证 ``/help`` 输出完整。

    幂等：可对同一 router / service 重复调用（测试套件每场景重建
    路由器），同步为覆盖语义、补充行整体替换而非追加。
    """
    script_service.register_into(router.registry)
    script_names = {spec.name for spec in script_service.registry.commands()}
    script_service.extra_help_lines[:] = [
        spec.help_text
        for spec in router.registry.commands()
        if spec.name not in script_names and spec.help_text
    ]


# ---------------------------------------------------------------------------
# 生产消息处理
# ---------------------------------------------------------------------------


async def on_qrcode(qrcode: str, img_content: str) -> None:
    """二维码就绪：保存 PNG 到 Local/ 并提示用户扫码。"""
    print("\n" + "=" * 60)
    print("请用微信扫描下方二维码登录 ClawBot（有效期约 5 分钟）")
    saved = save_qrcode_image(img_content, QR_IMAGE)
    if saved is not None:
        print(f"二维码图片已保存: {saved}")
        print("请打开该图片，用手机微信扫码并在手机上确认")
    elif img_content.startswith("http"):
        print(f"二维码链接: {img_content}")
    else:
        print(f"qrcode={qrcode}（未拿到图片内容，请检查网络/协议版本）")
    print("=" * 60 + "\n")


class MessageSink(Protocol):
    """消息回复出口协议。

    :class:`~transport.wechat.sender.MessageSender`（生产）与
    :class:`RecordingSender`（离线测试）都满足该协议，
    ``handle_message`` 只依赖这两个方法。
    """

    async def send_text(
        self,
        to_user_id: str,
        text: str,
        *,
        context_token: str = "",
        ilink_user_id: str | None = None,
    ) -> None: ...

    def typing(
        self, user_id: str, context_token: str = ""
    ) -> AbstractAsyncContextManager[None]: ...


async def handle_message(
    sender: MessageSink,
    parser: MessageParser,
    normalizer: MessageNormalizer,
    router: MessageRouter,
    message: WeixinMessage,
    ai_service: AIService | None = None,
    script_service: ScriptService | None = None,
) -> RoutingResult | None:
    """生产消息处理：messaging 流水线 + 模块分发。

    流程：解析微信原始消息（跳过 Bot 回声）→ 标准化 → 路由决策 →
    按目标分发：

    - ``SCRIPT`` 且指令为 ``/role``：AI 模块接管的角色指令；
    - ``SCRIPT`` 其余指令：交给 :class:`~modules.script.ScriptService`
      真实执行（沙箱子进程 / 内置处理器，回复文本由执行结果决定）；
    - ``AI``：交给 :class:`~modules.ai.service.AIService`（角色 + 记忆 + 上下文
      + GLM，内部经 ``sender`` 回复）；
    - ``AGENT``：模块未就绪，统一回复提示；
    - ``IGNORE``：不回复。

    :param ai_service: AI 服务实例；``None`` 时 AI 目标回复未就绪提示。
    :param script_service: Script 服务实例；``None`` 时 SCRIPT 目标
        回复未就绪提示。
    :returns: 路由决策；Bot 回声返回 ``None``。
    """
    parsed = parser.parse(message)
    if parsed.is_echo:
        return None  # Bot 自己发出的回声，跳过

    normalized = normalizer.normalize(parsed)
    result = router.route(normalized)
    logger.info(
        "路由决策: target=%s match=%s command=%s args=%s session=%s text=%r",
        result.target.value,
        result.match_type.value,
        result.command or "-",
        result.args or "-",
        normalized.session_key,
        normalized.text[:50],
    )

    if result.target is RouteTarget.IGNORE:
        logger.info("消息被路由忽略: session=%s", normalized.session_key)
        return result

    # /role 等 AI 角色指令：由 AI 模块执行
    if (
        result.target is RouteTarget.SCRIPT
        and result.command in AI_COMMANDS
        and ai_service is not None
    ):
        await ai_service.handle_command(
            result.command, result.args, normalized, sender
        )
        return result

    # AI 对话：完整流程由 AIService 编排（内部负责 typing 与回复）
    if result.target is RouteTarget.AI and ai_service is not None:
        await ai_service.handle_message(normalized, sender)
        return result

    # Script 指令：真实执行（typing 覆盖执行 + 回复，脚本可能耗时数秒）
    if result.target is RouteTarget.SCRIPT and script_service is not None:
        async with sender.typing(message.from_user_id, message.context_token):
            reply = await script_service.handle(
                result.command or "",
                result.args,
                ExecutionContext(
                    session_key=normalized.session_key,
                    user_id=normalized.sender_id,
                    command=result.command or "",
                    args=result.args or [],
                    message_id=normalized.message_id,
                    is_group=normalized.is_group,
                ),
            )
            await sender.send_text(
                message.from_user_id, reply, context_token=message.context_token
            )
        logger.info("已回复: %r", reply)
        return result

    fallback_reply = MODULE_NOT_READY_REPLIES.get(result.target)
    if fallback_reply is None:
        logger.warning("路由目标 %s 没有对应回复策略，已忽略", result.target.value)
        return result

    async with sender.typing(message.from_user_id, message.context_token):
        await sender.send_text(
            message.from_user_id, fallback_reply, context_token=message.context_token
        )
    logger.info("已回复: %r", fallback_reply)
    return result


# ---------------------------------------------------------------------------
# messaging 离线测试
# ---------------------------------------------------------------------------


class RecordingSender:
    """离线测试替身：与 ``MessageSender`` 接口兼容，只记录不发送。"""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_text(
        self,
        to_user_id: str,
        text: str,
        *,
        context_token: str = "",
        ilink_user_id: str | None = None,
    ) -> None:
        self.sent.append(text)

    @asynccontextmanager
    async def typing(
        self, user_id: str, context_token: str = ""
    ) -> AsyncIterator[None]:
        yield


def test_registry() -> CommandRegistry:
    """测试用命令注册表（含 README 示例指令 weather 与引号演示 say）。"""
    return CommandRegistry(
        [
            CommandSpec(
                name="help",
                keywords=("帮助", "菜单"),
                help_text="/help — 显示所有可用指令",
            ),
            CommandSpec(
                name="role",
                keywords=("角色",),
                help_text="/role list|set <名称> — 列出 / 切换 AI 角色",
            ),
            CommandSpec(
                name="agent",
                keywords=("智能体",),
                help_text="/agent list|use <名称> — 列出 / 切换智能体",
            ),
            CommandSpec(
                name="weather",
                keywords=("weather", "天气", "查天气"),
                help_text="/weather <城市> — 查询天气",
            ),
            CommandSpec(name="say", help_text='/say <文本> — 原样回复（演示引号参数）'),
        ]
    )


@dataclass
class Scenario:
    """离线测试场景：一条原始消息 + 期望的路由结果。"""

    name: str
    text: str
    expect_target: RouteTarget
    expect_command: str | None = None
    expect_match: MatchType | None = None
    expect_args: list[str] | None = None
    #: 精确期望回复；None 时用 MODULE_NOT_READY_REPLIES 映射（IGNORE 期望空列表）
    expect_replies: list[str] | None = None
    group: bool = False
    require_mention: bool = False
    sticky_agent: bool = False
    item_list: list[MessageItem] | None = None
    media_label: str = ""


#: 内置场景套件
SCENARIOS: list[Scenario] = [
    Scenario(
        "指令精确匹配",
        "/weather 北京",
        RouteTarget.SCRIPT,
        expect_command="weather",
        expect_match=MatchType.EXACT,
        expect_args=["北京"],
        expect_replies=[not_found_reply("weather")],
    ),
    Scenario(
        "指令带中文参数",
        "/role set 程序员",
        RouteTarget.SCRIPT,
        expect_command="role",
        expect_match=MatchType.EXACT,
        expect_args=["set", "程序员"],
        expect_replies=[
            "已切换到角色「程序员」。你好，我是你的编程助手，有什么代码问题尽管问。"
        ],
    ),
    Scenario(
        "引号参数",
        '/say "hello world"',
        RouteTarget.SCRIPT,
        expect_command="say",
        expect_match=MatchType.EXACT,
        expect_args=["hello world"],
        expect_replies=["hello world"],
    ),
    Scenario(
        "未知指令兜底帮助",
        "/nosuch foo",
        RouteTarget.SCRIPT,
        expect_command="help",
        expect_match=MatchType.FALLBACK,
        expect_args=["nosuch", "foo"],
    ),
    Scenario(
        "指令名前缀模糊",
        "/we 北京",
        RouteTarget.SCRIPT,
        expect_command="weather",
        expect_match=MatchType.KEYWORD,
        expect_args=["北京"],
        expect_replies=[not_found_reply("weather")],
    ),
    Scenario(
        "无前缀关键词匹配",
        "查天气 上海",
        RouteTarget.SCRIPT,
        expect_command="weather",
        expect_match=MatchType.KEYWORD,
        expect_args=["上海"],
        expect_replies=[not_found_reply("weather")],
    ),
    Scenario(
        "普通对话走 AI",
        "帮我写一个快速排序",
        RouteTarget.AI,
        expect_match=MatchType.DEFAULT,
        expect_replies=["（离线测试）已收到：帮我写一个快速排序"],
    ),
    Scenario(
        "群聊 @Bot（默认策略）",
        "@ClawHub 在吗",
        RouteTarget.AI,
        expect_match=MatchType.DEFAULT,
        group=True,
        expect_replies=["（离线测试）已收到：在吗"],
    ),
    Scenario(
        "群聊 @Bot（严格策略）",
        "@ClawHub 讲个笑话",
        RouteTarget.AI,
        expect_match=MatchType.DEFAULT,
        group=True,
        require_mention=True,
        expect_replies=["（离线测试）已收到：讲个笑话"],
    ),
    Scenario(
        "群聊未 @（严格策略）",
        "大家晚上吃啥",
        RouteTarget.IGNORE,
        expect_match=MatchType.DEFAULT,
        group=True,
        require_mention=True,
    ),
    Scenario(
        "粘性智能体模式",
        "帮我把需求拆成任务",
        RouteTarget.AGENT,
        expect_match=MatchType.STICKY,
        sticky_agent=True,
    ),
    Scenario(
        "粘性模式指令优先",
        "/help",
        RouteTarget.SCRIPT,
        expect_command="help",
        expect_match=MatchType.EXACT,
        sticky_agent=True,
    ),
    Scenario(
        "图片消息走 AI",
        "",
        RouteTarget.AI,
        expect_match=MatchType.DEFAULT,
        item_list=[
            MessageItem(
                type=MessageItemType.IMAGE,
                msg_id="p1",
                image_item=ImageItem(),
            )
        ],
        media_label="<图片>",
        expect_replies=["我目前只能理解文本消息～图片、语音、文件还看不来。"],
    ),
    Scenario(
        "语音 ASR 关键词",
        "",
        RouteTarget.SCRIPT,
        expect_command="weather",
        expect_match=MatchType.KEYWORD,
        expect_args=["北京"],
        item_list=[
            MessageItem(
                type=MessageItemType.VOICE,
                msg_id="v1",
                voice_item=VoiceItem(text="查天气 北京", playtime=2600),
            )
        ],
        media_label="<语音>查天气 北京",
        expect_replies=[not_found_reply("weather")],
    ),
    # -- Script 模块真实执行（沙箱子进程 / 内置处理器 / 失败路径） ----------
    Scenario(
        "脚本精确匹配（真实执行）",
        "/hello 世界",
        RouteTarget.SCRIPT,
        expect_command="hello",
        expect_match=MatchType.EXACT,
        expect_args=["世界"],
        expect_replies=["Hello, 世界！这是 hello 示例脚本的回复。"],
    ),
    Scenario(
        "脚本关键词匹配（真实执行）",
        "hello 世界",
        RouteTarget.SCRIPT,
        expect_command="hello",
        expect_match=MatchType.KEYWORD,
        expect_args=["世界"],
        expect_replies=["Hello, 世界！这是 hello 示例脚本的回复。"],
    ),
    Scenario(
        "脚本失败路径（退出码）",
        "/hello boom",
        RouteTarget.SCRIPT,
        expect_command="hello",
        expect_match=MatchType.EXACT,
        expect_args=["boom"],
        expect_replies=[
            "脚本 /hello 执行失败（退出码 1）：演示失败路径：这是一个故意的错误"
        ],
    ),
    Scenario(
        "Agent 指令未就绪",
        "/agent list",
        RouteTarget.SCRIPT,
        expect_command="agent",
        expect_match=MatchType.EXACT,
        expect_args=["list"],
        expect_replies=[MODULE_NOT_READY_REPLIES[RouteTarget.AGENT]],
    ),
]


def build_test_router(
    scenario: Scenario | None = None,
    script_service: ScriptService | None = None,
) -> MessageRouter:
    """构造测试路由器：按场景应用群聊 / 粘性配置。

    ``script_service`` 非空时把脚本指令同步进注册表（精确 / 前缀 /
    关键词匹配均可命中），并刷新帮助菜单补充行。
    """
    config = RouterConfig(
        group_require_mention=bool(scenario is not None and scenario.require_mention),
        bot_names=("ClawHub",),
    )
    mode_provider = None
    if scenario is not None and scenario.sticky_agent:
        mode_provider = lambda _session_key: RouteTarget.AGENT  # noqa: E731
    router = MessageRouter(
        test_registry(), mode_provider=mode_provider, config=config
    )
    if script_service is not None:
        wire_script_service(router, script_service)
    return router


def make_raw_message(
    text: str = "",
    *,
    item_list: list[MessageItem] | None = None,
    group_id: str = "",
    from_user_id: str = "tester@im.wechat",
    message_id: int = 1,
) -> WeixinMessage:
    """构造一条微信原始消息（测试输入）。"""
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
        seq=1,
        from_user_id=from_user_id,
        to_user_id="clawhub@im.bot",
        session_id="sess-test",
        group_id=group_id,
        message_type=MessageType.USER,
        context_token="ctx-test",
        create_time_ms=1_700_000_000_000,
        item_list=item_list,
    )


def check_scenario(
    scenario: Scenario,
    result: RoutingResult,
    replies: list[str],
    script_service: ScriptService | None = None,
) -> list[str]:
    """校验单场景的路由结果与回复，返回问题列表（空 = 通过）。

    期望回复优先级：``scenario.expect_replies`` → IGNORE 空列表 →
    SCRIPT 帮助菜单（由 Script 服务渲染，内容随注册表变化）→
    ``MODULE_NOT_READY_REPLIES`` 映射。
    """
    problems: list[str] = []
    if result.target is not scenario.expect_target:
        problems.append(
            f"目标应为 {scenario.expect_target.value}，实际 {result.target.value}"
        )
    if result.command != scenario.expect_command:
        problems.append(
            f"命令应为 {scenario.expect_command}，实际 {result.command}"
        )
    if (
        scenario.expect_match is not None
        and result.match_type is not scenario.expect_match
    ):
        problems.append(
            f"匹配方式应为{scenario.expect_match.value}，"
            f"实际{result.match_type.value}"
        )
    if scenario.expect_args is not None and result.args != scenario.expect_args:
        problems.append(f"参数应为 {scenario.expect_args}，实际 {result.args}")
    if scenario.expect_replies is not None:
        expected_replies = scenario.expect_replies
    elif scenario.expect_target is RouteTarget.IGNORE:
        expected_replies = []
    elif (
        scenario.expect_target is RouteTarget.SCRIPT
        and script_service is not None
    ):
        # 未显式指定期望回复的 SCRIPT 场景即帮助菜单（/help 与兜底）
        expected_replies = [script_service.render_help()]
    else:
        expected_replies = [MODULE_NOT_READY_REPLIES[scenario.expect_target]]
    if replies != expected_replies:
        problems.append(f"回复应为 {expected_replies}，实际 {replies}")
    return problems


async def run_scenario_suite() -> int:
    """运行内置场景套件，返回失败场景数。"""
    parser = MessageParser()
    normalizer = MessageNormalizer()
    # AI 场景走离线假 Provider：完整跑角色 / 记忆 / 上下文流程但不联网
    ai_service = build_ai_service(OfflineLLMProvider())
    # Script 场景走真实执行（scripts/ 目录扫描 + /say 演示指令）
    script_service = script_service_for_test()

    print("=" * 86)
    print("messaging 场景套件（离线，不连接微信）：解析 → 标准化 → 路由 → 回复")
    print("=" * 86)

    failures: list[str] = []
    for index, scenario in enumerate(SCENARIOS, start=1):
        router = build_test_router(scenario, script_service)
        sender = RecordingSender()
        message = make_raw_message(
            scenario.text,
            item_list=scenario.item_list,
            group_id="group777@im.wechat" if scenario.group else "",
        )
        result = await handle_message(
            sender, parser, normalizer, router, message, ai_service, script_service
        )
        assert result is not None  # 场景消息均非 Bot 回声

        label = scenario.text or scenario.media_label
        row = (
            f"{pad(str(index), 3)}"
            f"{pad(scenario.name, 22)}"
            f"{pad(truncate(label, 20), 21)}"
            f"{pad(result.target.value, 8)}"
            f"{pad(result.match_type.value, 9)}"
            f"{pad(result.command or '-', 8)}"
        )
        problems = check_scenario(scenario, result, sender.sent, script_service)
        if problems:
            failures.extend(f"{scenario.name}: {problem}" for problem in problems)
            print(f"{row}✗ {'; '.join(problems)}")
        else:
            reply = sender.sent[0] if sender.sent else "（不回复）"
            print(f"{row}✓ {reply}")

    print("-" * 86)
    total = len(SCENARIOS)
    print(f"结果: {total - len(failures)}/{total} 场景通过")
    if failures:
        print("失败详情:")
        for item in failures:
            print(f"  - {item}")
    return len(failures)


def script_service_for_test() -> ScriptService:
    """测试用 Script 服务（scripts/ 目录扫描 + /say 演示指令）。"""
    return build_script_service(demo=True)


async def test_single_message(text: str) -> None:
    """测试单条消息：打印流水线各环节结果与最终回复。"""
    text = repair_msys_mangled(text)
    parser = MessageParser()
    normalizer = MessageNormalizer()
    script_service = script_service_for_test()
    router = build_test_router(script_service=script_service)
    sender = RecordingSender()
    ai_service = build_ai_service(OfflineLLMProvider())

    result = await handle_message(
        sender,
        parser,
        normalizer,
        router,
        make_raw_message(text),
        ai_service,
        script_service,
    )

    print(f"输入   : {text!r}")
    if result is None:
        print("（Bot 回声，跳过）")
        return
    normalized = result.message
    print(
        f"解析   : text={normalized.text!r} kind={normalized.kind.value} "
        f"附件数={len(normalized.attachments)}"
    )
    print(
        f"标准化 : session={normalized.session_key} sender={normalized.sender_key} "
        f"mentions={normalized.mentions}"
    )
    print(
        f"路由   : target={result.target.value} match={result.match_type.value} "
        f"command={result.command or '-'} args={result.args}"
    )
    print(f"理由   : {result.reason}")
    if sender.sent:
        print(f"回复   : {sender.sent[0]}")
    else:
        print("回复   : （不回复）")


async def test_interactive() -> None:
    """交互式 REPL：输入消息实时查看路由决策与回复。"""
    parser = MessageParser()
    normalizer = MessageNormalizer()
    state = {"group": False, "agent": False}

    print("=" * 86)
    print("messaging 交互式测试（离线，不连接微信）")
    print("元命令: !group 切换群聊 | !agent 切换粘性智能体 | !help 帮助 | quit 退出")
    print("=" * 86)

    ai_service = build_ai_service(OfflineLLMProvider())
    script_service = script_service_for_test()
    while True:
        try:
            line = input("消息> ").strip()
        except EOFError:
            print()
            break
        if not line:
            continue
        if line in {"quit", "exit", "q"}:
            break
        if line == "!group":
            state["group"] = not state["group"]
            print(f"群聊模式: {'开' if state['group'] else '关'}")
            continue
        if line == "!agent":
            state["agent"] = not state["agent"]
            print(f"粘性智能体模式: {'开' if state['agent'] else '关'}")
            continue
        if line == "!help":
            print("直接输入消息内容即可测试；!group / !agent 切换场景；quit 退出")
            continue

        scenario = Scenario(
            name="REPL",
            text=line,
            expect_target=RouteTarget.AI,  # REPL 不校验，仅复用构造逻辑
            group=state["group"],
            sticky_agent=state["agent"],
        )
        router = build_test_router(scenario, script_service)
        sender = RecordingSender()
        message = make_raw_message(
            line, group_id="group777@im.wechat" if state["group"] else ""
        )
        result = await handle_message(
            sender, parser, normalizer, router, message, ai_service, script_service
        )
        if result is None:
            continue
        summary = f"→ [{result.target.value}/{result.match_type.value}"
        if result.command:
            summary += f" cmd={result.command} args={result.args}"
        summary += f"] {result.reason}"
        print(summary)
        if sender.sent:
            print(f"  回复: {sender.sent[0]}")
        else:
            print("  回复: （不回复）")


# ---------------------------------------------------------------------------
# 显示宽度工具（中文占 2 列）
# ---------------------------------------------------------------------------

#: MSYS（Git Bash）路径转换特征：``/weather`` → ``<Git 目录> PortableGit/ weather``
_MSYS_PATH_PREFIX_RE = re.compile(
    r"^[A-Za-z]:[\\/][^\n]*?[\\/]PortableGit[\\/](?=\S)"
)


def repair_msys_mangled(text: str) -> str:
    """还原被 MSYS 转换过的命令行参数。

    Git Bash 会把以 ``/`` 开头的参数重写成 Windows 路径，例如
    ``/weather 北京`` → ``E:/.../PortableGit/weather 北京``。
    该前缀还原为 ``/``；不影响普通文本（也可用 ``MSYS_NO_PATHCONV=1`` 规避）。
    """
    repaired = _MSYS_PATH_PREFIX_RE.sub("/", text)
    if repaired != text:
        logger.warning(
            "检测到 MSYS 路径转换，输入已还原为 %r（可用 MSYS_NO_PATHCONV=1 避免）",
            repaired,
        )
    return repaired


def display_width(text: str) -> int:
    """按东亚字符宽度计算显示宽度。"""
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def pad(text: str, width: int) -> str:
    """按显示宽度右补齐空格。"""
    return text + " " * max(width - display_width(text), 0)


def truncate(text: str, width: int) -> str:
    """按显示宽度截断，超出部分以 … 结尾。"""
    if display_width(text) <= width:
        return text
    result = ""
    used = 0
    for char in text:
        char_width = 2 if unicodedata.east_asian_width(char) in "WF" else 1
        if used + char_width > width - 1:
            break
        result += char
        used += char_width
    return result + "…"


# ---------------------------------------------------------------------------
# 生产运行
# ---------------------------------------------------------------------------


async def run(once: bool = False, reset: bool = False) -> None:
    """启动生产主流程：扫码登录 → 长轮询 → messaging 流水线。"""
    config = WeChatConfig.from_env()
    logger.info(
        "配置: base_url=%s channel_version=%s store=%s",
        config.base_url,
        config.channel_version,
        config.token_store_path,
    )

    async with WeChatClient(config) as client:
        auth = AuthManager(client)

        if reset:
            logger.info("--reset：清除已保存的登录态")
            auth.logout()

        if not auth.restore():
            logger.info("无有效登录态，发起扫码登录")
            await auth.login(on_qrcode=on_qrcode)
        else:
            logger.info("已从本地恢复登录态（跳过扫码）")

        sender = MessageSender(client)
        parser = MessageParser()
        normalizer = MessageNormalizer()
        # 默认命令表（help / role / agent）；各业务模块就绪后自行注册指令
        router = MessageRouter()
        # AI 模块：GLM 后端（GLM_API_KEY 就绪即用，否则回复引导配置提示）
        ai_service = build_ai_service()
        # Script 模块：扫描 scripts/ 目录注册脚本指令，同步进路由表
        script_service = build_script_service()
        wire_script_service(router, script_service)

        async def on_expired() -> None:
            """会话过期自动重连：重新扫码后轮询自动恢复。"""
            logger.warning("检测到会话过期，尝试自动重新登录")
            await auth.login(on_qrcode=on_qrcode)

        poller = MessagePoller(client, auth.store, on_session_expired=on_expired)

        async def handler(message: WeixinMessage) -> None:
            await handle_message(
                sender, parser, normalizer, router, message, ai_service, script_service
            )

        if once:
            logger.info("单次轮询模式：等待一条消息（最长约 35 秒）...")
            await poller.run(handler, once=True)
            logger.info("单次轮询结束，退出")
            return

        logger.info("进入长轮询模式：请在微信里给 ClawBot 发消息（Ctrl-C 退出）")
        await poller.run(handler)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WeChat-ClawHub 入口（生产运行 + messaging 离线测试）"
    )
    parser.add_argument(
        "--test-messaging",
        nargs="?",
        const="",
        default=None,
        metavar="TEXT",
        help="离线测试 messaging：无参数跑场景套件，带参数测试单条消息",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="配合 --test-messaging 进入交互式 REPL",
    )
    parser.add_argument(
        "--once", action="store_true", help="只登录并轮询一次后退出（快速验证收发链路）"
    )
    parser.add_argument(
        "--reset", action="store_true", help="清除已保存的登录态，强制重新扫码"
    )
    parser.add_argument("--debug", action="store_true", help="开启 DEBUG 日志")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.test_messaging is not None:
        # 测试模式自带的格式化输出更直观，压低日志噪音
        logging.getLogger().setLevel(logging.WARNING)

    try:
        if args.test_messaging is not None:
            if args.interactive:
                asyncio.run(test_interactive())
            elif args.test_messaging:
                asyncio.run(test_single_message(args.test_messaging))
            else:
                failures = asyncio.run(run_scenario_suite())
                if failures:
                    raise SystemExit(1)
        else:
            asyncio.run(run(once=args.once, reset=args.reset))
    except KeyboardInterrupt:
        logger.info("收到 Ctrl-C，已退出")
    print("bye")


if __name__ == "__main__":
    main()
