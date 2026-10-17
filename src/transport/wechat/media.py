"""微信媒体（图片等）上传 / 下载管线。

iLink 协议的媒体阶段：所有媒体文件在上传微信 CDN 前使用 **AES-128-ECB +
PKCS7** 加密，回复时通过 ``sendmessage`` 的 ``image_item`` 引用 CDNMedia。

上传流程::

    读取原图 → md5/字节数
      → POST /ilink/bot/getuploadurl（filekey、aeskey、rawsize、filesize...）
      → AES-128-ECB 加密
      → POST {cdn}/upload?encrypted_query_param=...&filekey=...（密文）
      → 响应头 x-encrypted-param 即下载引用参数
      → sendmessage: item_list=[{"type": 2, "image_item": {"media": {...}}}]

下载流程::

    优先 media.full_url；否则 {cdn}/download?encrypted_query_param=...
      → GET 密文 → AES-128-ECB 解密 → PKCS7 去填充 → 原文件
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Optional

import httpx
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from .client import WeChatClient
from .models import CDNMedia, GetUploadUrlRequest, UploadMediaType

logger = logging.getLogger(__name__)

#: AES 块大小（字节）
AES_BLOCK_SIZE = 16

#: 上传成功后 CDN 返回的下载参数响应头
UPLOAD_PARAM_HEADER = "x-encrypted-param"

#: CDN 上传 / 下载超时：视频等媒体文件较大，需远大于普通 API 的 15 秒
CDN_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


class MediaError(Exception):
    """媒体上传 / 下载失败。"""


@dataclass
class UploadedMedia:
    """一次媒体上传的结果，用于组装 ``sendmessage`` 的图片 / 视频消息。

    - :attr:`media`：sendmessage 引用的 CDNMedia（``aes_key`` 为
      ``base64(hex 字符串)`` 编码，与官方插件一致）；
    - :attr:`cipher_size`：PKCS7 填充后的密文字节数，对应 ``image_item.mid_size``，
      微信客户端需要它来正确拉取和渲染图片。
    """

    media: CDNMedia
    filekey: str
    aes_key_hex: str
    raw_size: int
    cipher_size: int


# ---------------------------------------------------------------------------
# AES-128-ECB 加解密
# ---------------------------------------------------------------------------


def _pkcs7_pad(data: bytes) -> bytes:
    pad_len = AES_BLOCK_SIZE - (len(data) % AES_BLOCK_SIZE)
    return data + bytes([pad_len]) * pad_len


def _pkcs7_unpad(data: bytes) -> bytes:
    if not data or len(data) % AES_BLOCK_SIZE != 0:
        raise MediaError(f"密文长度非法: {len(data)}")
    pad_len = data[-1]
    if not 1 <= pad_len <= AES_BLOCK_SIZE:
        raise MediaError(f"PKCS7 填充字节非法: {pad_len}")
    return data[:-pad_len]


def encrypt_aes128_ecb(data: bytes, key: bytes) -> bytes:
    """AES-128-ECB + PKCS7 加密（微信 CDN 上传前处理）。"""
    if len(key) != 16:
        raise MediaError(f"AES-128 密钥长度必须为 16 字节，当前 {len(key)}")
    encryptor = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
    return encryptor.update(_pkcs7_pad(data)) + encryptor.finalize()


def decrypt_aes128_ecb(data: bytes, key: bytes) -> bytes:
    """AES-128-ECB + PKCS7 解密（微信 CDN 下载后处理）。"""
    if len(key) != 16:
        raise MediaError(f"AES-128 密钥长度必须为 16 字节，当前 {len(key)}")
    decryptor = Cipher(algorithms.AES(key), modes.ECB()).decryptor()
    return _pkcs7_unpad(decryptor.update(data) + decryptor.finalize())


def decode_aes_key(aes_key: str) -> bytes:
    """解析 ``CDNMedia.aes_key``，兼容两种编码。

    - 格式 A：``base64(原始 16 字节)``；
    - 格式 B：``base64(32 字符十六进制字符串)``，需再 ``fromhex``。
    """
    try:
        raw = base64.b64decode(aes_key)
    except Exception as exc:
        raise MediaError(f"aes_key base64 解码失败: {exc}") from exc
    if len(raw) == 16:
        return raw
    if len(raw) == 32:
        try:
            return bytes.fromhex(raw.decode("ascii"))
        except ValueError as exc:
            raise MediaError(f"aes_key 不是合法 hex: {exc}") from exc
    raise MediaError(f"无法识别的 aes_key 长度: {len(raw)}")


def encrypted_size(raw_size: int) -> int:
    """PKCS7 填充后的密文长度：``ceil((raw_size + 1) / 16) * 16``。"""
    return ((raw_size // AES_BLOCK_SIZE) + 1) * AES_BLOCK_SIZE


# ---------------------------------------------------------------------------
# 上传 / 下载
# ---------------------------------------------------------------------------


class MediaUploader:
    """媒体上传器：图片等文件经 AES 加密后写入微信 CDN。

    用法::

        uploader = MediaUploader(client)
        uploaded = await uploader.upload_image(user_id, image_bytes)
        await sender.send_image(user_id, uploaded, context_token=ctx)
    """

    def __init__(self, client: WeChatClient) -> None:
        self.client = client

    async def upload_image(
        self,
        to_user_id: str,
        image_bytes: bytes,
        *,
        no_need_thumb: bool = True,
    ) -> UploadedMedia:
        """上传图片，返回可用于 ``send_image`` 的 :class:`UploadedMedia`。"""
        return await self.upload(
            to_user_id, image_bytes, media_type=UploadMediaType.IMAGE, no_need_thumb=no_need_thumb
        )

    async def upload_video(
        self,
        to_user_id: str,
        video_bytes: bytes,
        *,
        no_need_thumb: bool = True,
    ) -> UploadedMedia:
        """上传视频，返回可用于 ``send_video`` 的 :class:`UploadedMedia`。"""
        return await self.upload(
            to_user_id, video_bytes, media_type=UploadMediaType.VIDEO, no_need_thumb=no_need_thumb
        )

    async def upload(
        self,
        to_user_id: str,
        file_bytes: bytes,
        *,
        media_type: UploadMediaType,
        no_need_thumb: bool = True,
    ) -> UploadedMedia:
        """通用媒体上传（图片 / 视频 / 文件 / 语音）。"""
        key = os.urandom(16)  # 随机 16 字节 AES 密钥
        filekey = os.urandom(16).hex()  # 16 字节随机 hex
        cipher = encrypt_aes128_ecb(file_bytes, key)

        request = GetUploadUrlRequest(
            filekey=filekey,
            media_type=media_type,
            to_user_id=to_user_id,
            rawsize=len(file_bytes),
            rawfilemd5=hashlib.md5(file_bytes).hexdigest(),
            filesize=len(cipher),
            no_need_thumb=no_need_thumb,
            aeskey=key.hex(),
        )
        response = await self.client.get_upload_url(request)

        # v2.1.1+ 优先使用服务端返回的完整上传 URL
        upload_url = response.upload_full_url
        if not upload_url:
            upload_param = response.upload_param
            if not upload_param:
                raise MediaError("getuploadurl 响应缺少 upload_param / upload_full_url")
            upload_url = (
                f"{self.client.config.cdn_base_url.rstrip('/')}/upload"
                f"?encrypted_query_param={upload_param}&filekey={filekey}"
            )

        # 上传密文，下载参数在响应头 x-encrypted-param 中返回
        try:
            cdn_response = await self.client.http.post(
                upload_url,
                content=cipher,
                headers={"Content-Type": "application/octet-stream"},
                timeout=CDN_TIMEOUT,
            )
        except httpx.HTTPError as exc:
            raise MediaError(f"CDN 上传请求失败: {exc}") from exc
        if cdn_response.status_code >= 400:
            raise MediaError(
                f"CDN 上传失败 HTTP {cdn_response.status_code}: "
                f"{cdn_response.headers.get('x-error-message', cdn_response.text[:200])}"
            )
        download_param = cdn_response.headers.get(UPLOAD_PARAM_HEADER, "")
        if not download_param:
            raise MediaError("CDN 上传响应缺少 x-encrypted-param 头")

        logger.info(
            "媒体上传完成: type=%s rawsize=%d filesize=%d",
            media_type.name,
            len(file_bytes),
            len(cipher),
        )
        # 构造 sendmessage 引用的 CDNMedia。
        # 注意 aes_key 使用 base64(hex 字符串) 编码（与官方 openclaw-weixin 一致），
        # 直接 base64(原始 16 字节) 会导致部分微信客户端无法解密、图片打不开。
        return UploadedMedia(
            media=CDNMedia(
                encrypt_query_param=download_param,
                aes_key=base64.b64encode(key.hex().encode("ascii")).decode("ascii"),
                encrypt_type=1,
            ),
            filekey=filekey,
            aes_key_hex=key.hex(),
            raw_size=len(file_bytes),
            cipher_size=len(cipher),
        )


async def download_media(client: WeChatClient, media: CDNMedia) -> bytes:
    """下载并解密 CDN 媒体，返回原始字节。

    :param client: 已登录的 :class:`WeChatClient`（复用其连接池）。
    :param media: 入站消息中的 CDNMedia 引用。
    """
    if media.full_url:
        url = media.full_url
    elif media.encrypt_query_param:
        url = (
            f"{client.config.cdn_base_url.rstrip('/')}/download"
            f"?encrypted_query_param={media.encrypt_query_param}"
        )
    else:
        raise MediaError("CDNMedia 既无 full_url 也无 encrypt_query_param，无法下载")

    try:
        response = await client.http.get(url, timeout=CDN_TIMEOUT)
    except httpx.HTTPError as exc:
        raise MediaError(f"CDN 下载请求失败: {exc}") from exc
    if response.status_code >= 400:
        raise MediaError(f"CDN 下载失败 HTTP {response.status_code}: {response.text[:200]}")

    key = decode_aes_key(media.aes_key)
    plaintext = decrypt_aes128_ecb(response.content, key)
    logger.info("媒体下载完成: %d 字节（密文 %d 字节）", len(plaintext), len(response.content))
    return plaintext
