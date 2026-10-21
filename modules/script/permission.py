"""Script 模块权限：控制脚本的文件、网络与进程等权限。

对应 README 目录结构中的 ``modules/script/permission.py``。

权限模型（三层，逐层收紧）：

1. **档位声明**：每个 :class:`~modules.script.models.ScriptSpec` 用
   :class:`PermissionLevel` 声明自身需要的权限（只读 / 标准 / 完全）；
2. **策略校验**：:class:`PermissionChecker` 在启动前做预检——档位不得
   超过 :class:`PermissionPolicy` 上限，且脚本源码不得 import 其档位
   不允许的模块（网络 / 子进程 / 原生扩展）；
3. **运行期 containment**：由 :mod:`modules.script.sandbox` 以独立临时
   工作目录、净化环境变量、超时强杀、输出上限等方式兜底。

已知边界（如实说明，不夸大隔离能力）：

- 静态扫描是**尽力而为**的预检，无法防住混淆、动态 import 等绕过手段；
- 网络 / 子进程的**硬隔离**依赖 OS 级沙箱（网络命名空间 / Windows Job
  Object），当前版本未实现，仅做到「声明 + 预检 + 环境净化」；
- 真正不可信的业务脚本应在容器 / 独立低权账户中运行，本模块面向
  运营者自行注册的脚本（可信来源）。
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:  # 仅类型标注依赖（models 运行期依赖本模块的 PermissionLevel）
    from .models import ScriptSpec

#: 静态扫描禁则：档位不允许时，源码出现这些特征即拒绝启动。
#: 说明：正则对原始源码匹配，注释 / 字符串中的提及可能误伤（尽力而为）。
_NETWORK_RE = re.compile(
    r"\b(?:import\s+(?:socket|urllib|http|ftplib|telnetlib|smtplib|poplib|"
    r"requests|httpx|aiohttp|websocket|paramiko)|"
    r"from\s+(?:socket|urllib|http|ftplib|requests|httpx|aiohttp)\s+import)"
)
_SUBPROCESS_RE = re.compile(
    r"\b(?:import\s+(?:subprocess|multiprocessing)|"
    r"os\.(?:system|popen|exec|spawn|fork|posix_spawn)|"
    r"subprocess\.(?:run|Popen|call|check_call|check_output))"
)
_NATIVE_RE = re.compile(r"\b(?:import\s+ctypes|import\s+cffi)")


class PermissionLevel(StrEnum):
    """脚本权限档位（严格递增）。"""

    READONLY = "readonly"  #: 只读：禁写文件、禁网络、禁子进程
    STANDARD = "standard"  #: 标准：可写沙箱目录，禁网络、禁子进程
    FULL = "full"  #: 完全：不额外限制（仅受资源上限约束）


#: 档位严格递增顺序（用于「不超过策略上限」比较）
_LEVEL_ORDER: dict[PermissionLevel, int] = {
    PermissionLevel.READONLY: 0,
    PermissionLevel.STANDARD: 1,
    PermissionLevel.FULL: 2,
}


class PermissionSet(BaseModel):
    """解析后的权限集合（档位的展开形式）。"""

    allow_network: bool = Field(default=False, description="允许网络访问")
    allow_subprocess: bool = Field(default=False, description="派生子进程")
    allow_write: bool = Field(default=False, description="允许写文件")
    write_scope: str = Field(
        default="none", description="写入范围：none / sandbox / anywhere"
    )

    @classmethod
    def from_level(cls, level: PermissionLevel) -> PermissionSet:
        """把档位展开为具体权限集合。"""
        if level is PermissionLevel.READONLY:
            return cls(allow_write=False, write_scope="none")
        if level is PermissionLevel.STANDARD:
            return cls(allow_write=True, write_scope="sandbox")
        return cls(
            allow_network=True,
            allow_subprocess=True,
            allow_write=True,
            write_scope="anywhere",
        )


class PermissionDecision(BaseModel):
    """权限预检结论。"""

    allowed: bool
    level: PermissionLevel = PermissionLevel.STANDARD
    violations: list[str] = Field(
        default_factory=list, description="违规项（空 = 通过）"
    )
    reason: str = Field(default="", description="人类可读结论")


class PermissionPolicy(BaseModel):
    """权限策略：运营者侧的全局约束。"""

    #: 允许注册的最高档位（默认 STANDARD，FULL 需运营者显式放开）
    max_level: PermissionLevel = Field(
        default=PermissionLevel.STANDARD, description="策略允许的最高档位"
    )
    #: 是否启用源码静态扫描
    static_scan: bool = Field(default=True, description="静态源码扫描开关")


class PermissionChecker:
    """权限预检器：启动脚本前的第一道关口。

    用法::

        checker = PermissionChecker(
            PermissionPolicy(max_level=PermissionLevel.STANDARD)
        )
        decision = checker.check(spec)
        if not decision.allowed:
            ...  # 以 DENIED 结束，reason 反馈给运营者

    :param policy: 全局权限策略，默认
        :class:`PermissionPolicy`（最高 STANDARD + 开静态扫描）。
    """

    def __init__(self, policy: PermissionPolicy | None = None) -> None:
        self.policy = policy or PermissionPolicy()

    def resolve(self, spec: ScriptSpec) -> PermissionSet:
        """把脚本声明的档位展开为权限集合。"""
        return PermissionSet.from_level(spec.permission)
    def check(self, spec: ScriptSpec) -> PermissionDecision:
        """对一条脚本定义做权限预检（不执行脚本）。

        内置 ``handler`` 指令是运营者在宿主进程内的代码，不过子进程沙箱，
        免检（档位仅作声明）；落盘脚本才走完整预检。
        """
        if spec.is_builtin:
            return PermissionDecision(
                allowed=True,
                level=spec.permission,
                reason="内置指令（进程内处理器），免沙箱预检",
            )

        level = spec.permission
        violations: list[str] = []

        # 1. 档位不得超过策略上限
        if _LEVEL_ORDER[level] > _LEVEL_ORDER[self.policy.max_level]:
            violations.append(
                f"权限档位 {level.value} 超过策略上限 {self.policy.max_level.value}"
            )

        # 2. 源码静态扫描（仅落盘脚本；内置 handler 是运营者代码，免扫）
        if self.policy.static_scan and spec.entry is not None:
            violations.extend(self._scan_source(spec))

        if violations:
            return PermissionDecision(
                allowed=False,
                level=level,
                violations=violations,
                reason="；".join(violations),
            )
        return PermissionDecision(
            allowed=True, level=level, reason="权限校验通过"
        )

    # -- 内部 ----------------------------------------------------------------

    def _scan_source(self, spec: ScriptSpec) -> list[str]:
        """扫描脚本源码，返回违规描述列表（无违返返回空列表）。"""
        assert spec.entry is not None  # 调用方保证
        entry: Path = spec.entry
        if not entry.is_file():
            return [f"脚本文件不存在: {entry}"]

        try:
            source = entry.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return [f"无法读取脚本源码: {exc}"]

        resolved = self.resolve(spec)
        violations: list[str] = []

        if not resolved.allow_network and _NETWORK_RE.search(source):
            violations.append(
                f"源码疑似使用网络库（档位 {spec.permission.value} 禁止联网），"
                "如需联网请提升 permission 档位"
            )
        if not resolved.allow_subprocess and _SUBPROCESS_RE.search(source):
            violations.append(
                f"源码疑似派生子进程（档位 {spec.permission.value} 禁止），"
                "如需子进程请提升 permission 档位"
            )
        if not resolved.allow_network and _NATIVE_RE.search(source):
            violations.append(
                f"源码疑似使用 ctypes/cffi 原生扩展（档位 {spec.permission.value} "
                "禁止），如需请提升 permission 档位"
            )
        return violations
