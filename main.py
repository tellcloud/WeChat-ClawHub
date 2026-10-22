# main.py将作为唯一的项目入口文件
"""WeChat-ClawHub 唯一入口：生产运行（扫码登录 → 长轮询 → 分发）。

用法::

    python main.py                # 扫码登录 + 长轮询（Ctrl-C 退出）
    python main.py --once         # 只登录并轮询一次
    python main.py --reset        # 清除登录态，强制重新扫码
    python main.py --debug        # DEBUG 日志

    python main.py --test-messaging [TEXT | -i]   # messaging 离线测试

生产流程：扫码登录 → 长轮询收消息 → messaging 流水线（解析 → 标准化 →
路由）→ 按目标分发。Script 指令由 :class:`~modules.script.ScriptService`
真实执行（沙箱子进程，``scripts/`` 目录自动扫描注册）；AI 模块已就绪
（GLM 后端，角色 / 记忆 / 上下文完整闭环，``/role`` 与 ``/llm``
指令归 AI 模块执行）；Agent 模块就绪前回复「暂未完成Agent模块的开发工作」；
IGNORE 不回复。

AI 对话需要 LLM API Key：设置环境变量 ``GLM_API_KEY``（智谱）、
``DEEPSEEK_API_KEY``（DeepSeek）或 ``QWEN_API_KEY``（阿里云百炼）后
生效（复制 ``.env.example`` 为 ``.env`` 并填入，每次运行自动加载，
见 :data:`ENV_FILE`）；未配置时 AI 模块会回复引导配置的提示
（角色切换等本地功能不受影响）。已配置多个后端时，可随时用
``/llm list`` 查看、``/llm set <名称>`` 切换（GLM / DeepSeek / Qwen）。

``--test-messaging`` 为 messaging 离线测试入口（无需登录 / 网络，
直接构造微信原始消息跑完整流水线并校验回复），支持场景套件 /
单条消息 / 交互式 REPL 三种模式；实现位于
:mod:`tests.offline.messaging_pipeline`，本文件只保留入口分发。

登录成功后凭据保存在 Local/wechat_credentials.json（.gitignore 已忽略），
之后重复运行无需再次扫码；会话过期（errcode=-14）会自动重新扫码恢复。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from contextlib import AbstractAsyncContextManager
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
)
from messaging import (
    MessageNormalizer,
    MessageParser,
    MessageRouter,
    RouteTarget,
    RoutingResult,
)
from modules.script import (
    ExecutionContext,
    ScriptConfig,
    ScriptService,
)
from transport.wechat import (
    AuthManager,
    MessagePoller,
    MessageSender,
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

#: AI 模块接管的指令名（README「AI 角色对话」中的 /role 系列与 /llm 系列）
AI_COMMANDS = frozenset({"role", "llm"})


def build_ai_service(provider: LLMProvider | None = None) -> AIService:
    """装配 AI 服务。

    生产（``provider=None``）：读取 ``GLM_API_KEY`` 等环境变量构造 GLM
    Provider（默认后端）；未配置 Key 时 Provider 为 ``None``，AI 对话会
    回复引导配置的提示（角色切换等本地功能仍可用）；用户随后可用
    ``/llm set deepseek`` 等指令在已配置的后端间切换。

    测试：传入假 Provider（如 :mod:`tests.offline.messaging_pipeline`
    中的 :class:`~tests.offline.messaging_pipeline.OfflineLLMProvider`）
    即可完整离线运行。
    """
    if provider is None:
        glm_config = GLMConfig.from_env()
        provider = GLMProvider(glm_config) if glm_config.api_key else None
    service = AIService(provider=provider, config=AIServiceConfig.from_env())
    logger.info(
        "AI 服务已装配: provider=%s personas=%s",
        service.provider.name if service.provider else "未配置（缺少 API Key）",
        service.config.personas_dir,
    )
    return service


def build_script_service(scripts_dir: Path | None = None) -> ScriptService:
    """装配 Script 服务：扫描脚本目录 + 未就绪模块指令桥接。

    - 目录中的 ``*.py`` 按 ``# script:`` 元信息自动注册为指令；
    - ``/agent`` 等已知但归属未就绪模块的指令返回占位提示
      （``/role`` 由 AI 模块接管，不在此列）；
    - 沙箱工作目录位于 ``Local/script_runs``（.gitignore 已忽略）。

    :param scripts_dir: 脚本目录，默认项目根下的 ``scripts/``。
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
        print(f"二维码链接: {qrcode}")
    else:
        print(f"qrcode={qrcode}（未拿到图片内容，请检查网络/协议版本）")
    print("=" * 60 + "\n")


class MessageSink(Protocol):
    """消息回复出口协议。

    :class:`~transport.wechat.sender.MessageSender`（生产）与
    :class:`tests.offline.messaging_pipeline.RecordingSender`
    （离线测试）都满足该协议，``handle_message`` 只依赖这两个方法。
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

    - ``SCRIPT`` 且指令为 ``/role`` / ``/llm``：AI 模块接管的指令；
    - ``SCRIPT`` 其余指令：交给 :class:`~modules.script.ScriptService`
      真实执行（沙箱子进程 / 内置处理器，回复文本由执行结果决定）；
    - ``AI``：交给 :class:`~modules.ai.service.AIService`（角色 + 记忆 + 上下文
      + GLM / DeepSeek，内部经 ``sender`` 回复）；
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

    # /role / llm 等 AI 模块指令：由 AI 模块执行
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
        # 默认命令表（help / role / llm / agent）；各业务模块就绪后自行注册指令
        router = MessageRouter()
        # AI 模块：GLM 后端（API Key 就绪即用，否则回复引导配置提示；
        # 用户可 /llm set deepseek 切换到已配置的其他后端）
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
            # messaging 离线测试实现位于 tests/offline/messaging_pipeline.py，
            # 此处仅做入口分发（延迟导入，生产运行不加载测试代码）。
            from tests.offline.messaging_pipeline import (
                run_interactive,
                run_scenario_suite,
                run_single_message,
            )

            if args.interactive:
                asyncio.run(run_interactive())
            elif args.test_messaging:
                asyncio.run(run_single_message(args.test_messaging))
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
