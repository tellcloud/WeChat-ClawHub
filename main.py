# main.py将作为唯一的项目入口文件
"""WeChat-ClawHub 唯一入口：src/transport/wechat 层联通性测试（回声机器人）。

功能对应 M0 验收标准「微信 → 系统 → 微信」：
扫码登录 → 长轮询收消息 → 原样回复，并演示「正在输入」状态与自动重连。

支持的消息类型：
    - 文本：原样回复 Echo: xxx
    - 图片：下载后原样回传（AES-128-ECB + CDN 媒体管线）
    - 视频：下载后原样回传（同上，CDN 超时放宽到 120 秒）
    - 语音：按微信服务端识别文本回复（无识别结果时忽略）

运行前准备：
    1. 安装依赖（任选其一）：
       - cd my-project && uv sync            # 使用 uv 环境
       - pip install httpx pydantic cryptography
    2. 手机微信需安装 ClawBot 插件（微信 >= 8.0.70）；
    3. 本机网络可访问 ilinkai.weixin.qq.com。

运行方式：
    python main.py               # 扫码登录 + 长轮询回声（Ctrl-C 退出）
    python main.py --once        # 只登录并轮询一次（快速验证收发链路）
    python main.py --reset       # 清除已保存的登录态，强制重新扫码
    python main.py --debug       # 开启 DEBUG 日志（查看完整请求细节）

登录成功后凭据保存在 Local/wechat_credentials.json（已被 .gitignore 忽略），
之后重复运行无需再次扫码；会话过期（errcode=-14）会自动重新扫码恢复。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Protocol

from src.transport.wechat import (
    AuthManager,
    CDNMedia,
    MediaUploader,
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


async def handle_message(
    sender: MessageSender, uploader: MediaUploader, message: WeixinMessage
) -> None:
    """回声机器人：文本原样回复，图片下载后原样回传。

    同时演示 typing 上下文管理器：处理期间微信侧显示“对方正在输入中”。
    """
    # 1. 图片 / 视频消息：下载后原样回传（AES-128-ECB + CDN 媒体管线）
    media_targets: list[tuple[MessageItemType, CDNMedia]] = []
    for item in message.item_list:
        if item.type == MessageItemType.IMAGE and item.image_item and item.image_item.media:
            media_targets.append((item.type, item.image_item.media))
        elif item.type == MessageItemType.VIDEO and item.video_item and item.video_item.media:
            media_targets.append((item.type, item.video_item.media))
    if media_targets:
        logger.info(
            "收到媒体消息: from=%s 共 %d 个（%s）",
            message.from_user_id,
            len(media_targets),
            "/".join(MEDIA_LABELS[kind] for kind, _ in media_targets),
        )
        for kind, media_ref in media_targets:
            label = MEDIA_LABELS[kind]
            try:
                data = await download_media(uploader.client, media_ref)
            except Exception:
                logger.exception("%s下载失败", label)
                await sender.send_text(
                    message.from_user_id,
                    f"{label}下载失败，暂时无法回传",
                    context_token=message.context_token,
                )
                continue
            async with sender.typing(message.from_user_id, message.context_token):
                if kind == MessageItemType.IMAGE:
                    uploaded = await uploader.upload_image(message.from_user_id, data)
                    await sender.send_image(
                        message.from_user_id, uploaded, context_token=message.context_token
                    )
                else:
                    uploaded = await uploader.upload_video(message.from_user_id, data)
                    await sender.send_video(
                        message.from_user_id, uploaded, context_token=message.context_token
                    )
            logger.info("%s已回传: %d 字节", label, len(data))
        return

    # 2. 文本（含语音消息的微信服务端识别结果）
    text = message.text.strip()
    if not text:
        for item in message.item_list:
            if (
                item.type == MessageItemType.VOICE
                and item.voice_item
                and item.voice_item.text.strip()
            ):
                text = item.voice_item.text.strip()
                logger.info("语音识别文本: %r", text)
                break
    if not text:
        kinds = [int(item.type) for item in message.item_list]
        logger.info("收到暂不支持的消息（item 类型=%s），已忽略", kinds)
        return

    logger.info(
        "收到消息: from=%s group=%s text=%r context_token=%s",
        message.from_user_id,
        message.group_id or "-",
        text,
        "有" if message.context_token else "无",
    )

    reply = f"Echo: {text}"
    async with sender.typing(message.from_user_id, message.context_token):
        await asyncio.sleep(0.5)  # 模拟业务处理耗时
        await sender.send_text(
            message.from_user_id, reply, context_token=message.context_token
        )
    logger.info("已回复: %r", reply)


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
        uploader = MediaUploader(client)

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

    try:
        asyncio.run(run(once=args.once, reset=args.reset))
    except KeyboardInterrupt:
        logger.info("收到 Ctrl-C，已退出")
    print("bye")


if __name__ == "__main__":
    main()
