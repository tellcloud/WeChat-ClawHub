"""微信认证：登录、Token 保存和认证状态管理。

职责（对应 README 目录结构中的 ``auth.py``）：

- 二维码扫码登录全流程：``get_bot_qrcode`` → 轮询 ``get_qrcode_status``
  （``wait`` → ``scaned`` → ``confirmed``）；
- 处理二维码过期（刷新，最多 3 次）与跨机房调度
  （``scaned_but_redirect`` → 切换 ``redirect_host``）；
- :class:`TokenStore` 将凭据与长轮询游标持久化到 JSON 文件，
  重启后可恢复，避免重复扫码；
- 会话过期（``errcode=-14``）时清理凭据，等待重新登录。
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Optional

from .client import QrcodeCallback, WeChatClient
from .models import Credentials, QRStatus

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """登录失败（二维码过期超限、用户取消等）。"""


class TokenStore:
    """凭据与长轮询游标的 JSON 持久化。

    文件结构::

        {
          "credentials": {...} | null,
          "get_updates_buf": "同步游标",
          "updated_at": "ISO8601"
        }

    默认写入 ``Local/`` 本地数据目录（已被 .gitignore 忽略，不会提交）。
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> dict:
        """读取状态文件；不存在或损坏时返回空状态。"""
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("凭据文件读取失败，将重新登录: %s (%s)", self.path, exc)
            return {}

    def save(self, *, credentials: Optional[Credentials] = None, get_updates_buf: Optional[str] = None) -> None:
        """原子写入状态文件（先写临时文件再替换，防止写一半损坏）。"""
        state = self.load()
        if credentials is not None:
            state["credentials"] = credentials.model_dump(mode="json")
        if get_updates_buf is not None:
            state["get_updates_buf"] = get_updates_buf
        state["updated_at"] = datetime.now(timezone.utc).isoformat()

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with open(fd, "w", encoding="utf-8") as fh:
                json.dump(state, fh, ensure_ascii=False, indent=2)
            Path(tmp_name).replace(self.path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    # -- 凭据 ---------------------------------------------------------------

    def load_credentials(self) -> Optional[Credentials]:
        """读取已保存的凭据，没有则返回 ``None``。"""
        data = self.load().get("credentials")
        if not data:
            return None
        try:
            return Credentials.model_validate(data)
        except Exception as exc:  # pydantic ValidationError 等
            logger.warning("凭据解析失败，将重新登录: %s", exc)
            return None

    def save_credentials(self, credentials: Credentials) -> None:
        """保存登录成功后的凭据。"""
        self.save(credentials=credentials)

    def clear_credentials(self) -> None:
        """清除凭据（会话过期 / 重新登录前调用）。"""
        state = self.load()
        state["credentials"] = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    # -- 长轮询游标 ----------------------------------------------------------

    def load_cursor(self) -> str:
        """读取 ``get_updates_buf`` 游标；首次为空串。"""
        return str(self.load().get("get_updates_buf", ""))

    def save_cursor(self, cursor: str) -> None:
        """保存 ``get_updates_buf`` 游标，防止重启后重复收消息。"""
        self.save(get_updates_buf=cursor)


class AuthManager:
    """扫码登录与认证状态管理。

    用法::

        async with WeChatClient() as client:
            auth = AuthManager(client)
            credentials = await auth.login(on_qrcode=show_qr)
    """

    def __init__(self, client: WeChatClient, store: Optional[TokenStore] = None) -> None:
        self.client = client
        self.store = store or TokenStore(client.config.token_store_path)

    # -- 状态 ---------------------------------------------------------------

    @property
    def is_authenticated(self) -> bool:
        """当前是否已认证。"""
        return self.client.credentials is not None

    def restore(self) -> bool:
        """从持久化文件恢复凭据。

        :returns: 恢复成功返回 ``True``（无需重新扫码）。
        """
        credentials = self.store.load_credentials()
        if credentials is None:
            return False
        self.client.credentials = credentials
        logger.info(
            "已从 %s 恢复登录态: bot=%s user=%s",
            self.store.path,
            credentials.ilink_bot_id,
            credentials.ilink_user_id,
        )
        return True

    def logout(self) -> None:
        """清除本地登录态。"""
        self.client.credentials = None
        self.store.clear_credentials()

    # -- 登录流程 ------------------------------------------------------------

    async def login(
        self,
        on_qrcode: Optional[QrcodeCallback] = None,
        *,
        poll_interval: Optional[float] = None,
        max_refresh: Optional[int] = None,
    ) -> Credentials:
        """执行完整扫码登录流程并持久化凭据。

        流程：获取二维码 →（回调展示）→ 每 ``poll_interval`` 秒轮询状态 →
        ``confirmed`` 保存凭据；``expired`` 刷新二维码（最多
        ``max_refresh`` 次）；``scaned_but_redirect`` 切换基址后继续轮询。

        :raises AuthenticationError: 二维码过期超过刷新次数，或用户取消登录。
        """
        interval = poll_interval or self.client.config.qr_poll_interval
        refresh_limit = max_refresh if max_refresh is not None else self.client.config.qr_max_refresh

        qrcode = await self._request_qrcode(on_qrcode)
        refresh_count = 0

        while True:
            status = await self.client.get_qrcode_status(qrcode)

            if status.status == QRStatus.CONFIRMED:
                credentials = status.to_credentials()
                if credentials is None:  # 理论上不会发生
                    raise AuthenticationError("登录确认响应缺少凭据字段")
                self.client.credentials = credentials
                self.store.save_credentials(credentials)
                logger.info(
                    "扫码登录成功: bot=%s user=%s baseurl=%s",
                    credentials.ilink_bot_id,
                    credentials.ilink_user_id,
                    credentials.baseurl,
                )
                return credentials

            if status.status == QRStatus.SCANED_BUT_REDIRECT and status.redirect_host:
                # 跨机房调度：切换基址后继续轮询同一个二维码
                self.client.redirect_to(status.redirect_host)
                await asyncio.sleep(interval)
                continue

            if status.status == QRStatus.EXPIRED:
                refresh_count += 1
                if refresh_count > refresh_limit:
                    raise AuthenticationError(
                        f"二维码过期且刷新次数超过上限（{refresh_limit} 次），请重新发起登录"
                    )
                logger.info("二维码已过期，刷新（第 %d/%d 次）", refresh_count, refresh_limit)
                qrcode = await self._request_qrcode(on_qrcode)
                continue

            # wait / scaned：继续等待
            await asyncio.sleep(interval)

    async def _request_qrcode(self, on_qrcode: Optional[QrcodeCallback]) -> str:
        """获取二维码并回调展示，返回二维码会话标识符。"""
        qr = await self.client.get_bot_qrcode()
        logger.info("已获取登录二维码: qrcode=%s", qr.qrcode)
        if on_qrcode is not None:
            await on_qrcode(qr.qrcode, qr.qrcode_img_content)
        return qr.qrcode


def save_qrcode_image(qrcode_img_content: str, path: str | Path) -> Optional[Path]:
    """把二维码保存为 PNG 文件，方便用户在终端外查看。

    :param qrcode_img_content: ``data:image/png;base64,...`` 或图片 URL。
    :param path: 保存路径（URL 形式时忽略，直接返回 ``None``）。
    :returns: 保存成功返回路径，否则 ``None``。
    """
    if qrcode_img_content.startswith("data:"):
        header, _, payload = qrcode_img_content.partition(",")
        if "base64" not in header:
            logger.warning("不支持的二维码图片编码: %s", header)
            return None
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(payload))
        logger.info("二维码图片已保存: %s", target)
        return target
    if qrcode_img_content.startswith("http"):
        logger.info("二维码图片 URL: %s", qrcode_img_content)
        return None
    logger.warning("无法识别的二维码内容: %.50s", qrcode_img_content)
    return None
