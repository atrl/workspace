#!/usr/bin/env python3
"""统一通知模块：支持 Bark (iOS)、macOS 原生通知、Discord、控制台降级。

环境变量:
    BARK_KEY            - Bark 设备 key（优先）
    BARK_DEVICE_KEY     - Bark 设备 key（兼容 OpenClaw 默认）
    DISCORD_BOT_TOKEN   - Discord bot token（可选，启用 Discord 通知）
    DISCORD_CHANNEL_ID  - Discord 频道 ID（可选，默认 tradingagent）

用法:
    from production.notify import get_notifier
    notifier = get_notifier()
    notifier.send("标题", "内容", level="info")

    # 同时启用 Discord
    notifier = get_notifier(include_discord=True)
"""
from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class Notifier(ABC):
    """通知抽象基类。"""

    @abstractmethod
    def send(
        self,
        title: str,
        body: str,
        level: str = "info",
        sound: Optional[str] = None,
        group: Optional[str] = None,
    ) -> bool:
        """发送通知。

        Args:
            title: 通知标题
            body: 通知内容
            level: info / warning / error / critical
            sound: Bark 铃声名称（仅 Bark 有效）
            group: 通知分组（仅 Bark 有效）
        """


class ConsoleNotifier(Notifier):
    """仅打印到 stdout，作为最后降级。"""

    def send(
        self,
        title: str,
        body: str,
        level: str = "info",
        sound: Optional[str] = None,
        group: Optional[str] = None,
    ) -> bool:
        print(f"[{level.upper()}] {title}\n{body}")
        return True


class MacOSNotifier(Notifier):
    """macOS 原生通知（osascript）。"""

    def send(
        self,
        title: str,
        body: str,
        level: str = "info",
        sound: Optional[str] = None,
        group: Optional[str] = None,
    ) -> bool:
        try:
            script = f'display notification "{body.replace(chr(34), chr(92)+chr(34))}" with title "{title.replace(chr(34), chr(92)+chr(34))}"'
            subprocess.run(["osascript", "-e", script], check=True, capture_output=True)
            return True
        except Exception as exc:
            print(f"[macOS notify failed] {exc}", file=sys.stderr)
            return False


class BarkNotifier(Notifier):
    """Bark iOS 推送通知。

    API 文档: https://github.com/Finb/Bark
    """

    def __init__(self, device_key: str, base_url: str = "https://api.day.app"):
        self.device_key = device_key
        self.base_url = base_url.rstrip("/")

    def send(
        self,
        title: str,
        body: str,
        level: str = "info",
        sound: Optional[str] = None,
        group: Optional[str] = None,
    ) -> bool:
        url = f"{self.base_url}/{self.device_key}"
        payload: dict = {
            "title": title,
            "body": body,
            "group": group or "soxl_paper",
        }
        if sound:
            payload["sound"] = sound
        if level == "critical":
            payload["level"] = "critical"
        elif level == "timeSensitive":
            payload["level"] = "timeSensitive"
        elif level == "passive":
            payload["level"] = "passive"

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
                return resp.getcode() == 200
        except Exception as exc:
            print(f"[bark notify failed] {exc}", file=sys.stderr)
            return False


class DiscordNotifier(Notifier):
    """Discord 频道消息通知（通过 Bot）。"""

    LEVEL_EMOJI = {
        "info": "ℹ️",
        "warning": "⚠️",
        "error": "🚨",
        "critical": "🔴",
    }

    def __init__(self, token: str, channel_id: str):
        self.token = token
        self.channel_id = channel_id
        self._base_url = "https://discord.com/api/v10"

    def send(
        self,
        title: str,
        body: str,
        level: str = "info",
        sound: Optional[str] = None,
        group: Optional[str] = None,
    ) -> bool:
        emoji = self.LEVEL_EMOJI.get(level, "ℹ️")
        content = f"{emoji} **{title}**\n{body}"
        # Truncate if too long (Discord limit ~2000)
        if len(content) > 1990:
            content = content[:1987] + "..."

        url = f"{self._base_url}/channels/{self.channel_id}/messages"
        payload = {"content": content}
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bot {self.token}",
                    "Content-Type": "application/json",
                    "User-Agent": "DiscordBot (https://github.com/atrl/workspace, 1.0)",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                resp.read()
                return resp.getcode() in (200, 201)
        except Exception as exc:
            print(f"[discord notify failed] {exc}", file=sys.stderr)
            return False


class MultiNotifier(Notifier):
    """组合多个通知器，依次发送。"""

    def __init__(self, notifiers: list[Notifier]):
        self.notifiers = notifiers

    def send(
        self,
        title: str,
        body: str,
        level: str = "info",
        sound: Optional[str] = None,
        group: Optional[str] = None,
    ) -> bool:
        results = []
        for notifier in self.notifiers:
            try:
                ok = notifier.send(title, body, level, sound, group)
                results.append(ok)
            except Exception as exc:
                print(f"[multi-notify sub-failed] {exc}", file=sys.stderr)
                results.append(False)
        return any(results)


def _bark_key() -> Optional[str]:
    return os.environ.get("BARK_KEY") or os.environ.get("BARK_DEVICE_KEY")


def _discord_token() -> Optional[str]:
    """从环境变量或 .env.discord 读取 Discord bot token。"""
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if token:
        return token
    env_file = Path(__file__).resolve().parents[1] / ".env.discord"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("DISCORD_BOT_TOKEN="):
                return line.split("=", 1)[1].strip()
    return None


def _discord_channel_id() -> str:
    """返回 Discord 频道 ID，默认 tradingagent (bot 有权限)。"""
    return os.environ.get("DISCORD_CHANNEL_ID", "1475287062553956565")


def get_notifier(prefer_bark: bool = True, include_discord: bool = False) -> Notifier:
    """按优先级返回可用通知器。

    Priority:
        1. Bark (若 BARK_KEY/BARK_DEVICE_KEY 存在且 prefer_bark=True)
        2. macOS native (若运行在 macOS)
        3. Console (降级)

    Args:
        prefer_bark: 优先使用 Bark
        include_discord: 同时发送 Discord 消息（如果 token 可用）
    """
    notifiers: list[Notifier] = []

    key = _bark_key()
    if prefer_bark and key:
        notifiers.append(BarkNotifier(key))
    elif sys.platform == "darwin":
        notifiers.append(MacOSNotifier())
    else:
        notifiers.append(ConsoleNotifier())

    if include_discord:
        token = _discord_token()
        if token:
            notifiers.append(DiscordNotifier(token, _discord_channel_id()))

    if len(notifiers) == 1:
        return notifiers[0]
    return MultiNotifier(notifiers)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Test notification channels")
    parser.add_argument("--title", default="SOXL Paper Trading Test")
    parser.add_argument("--body", default="Notification channel is working.")
    parser.add_argument("--level", default="info", choices=["info", "warning", "error", "critical"])
    parser.add_argument("--sound", default="", help="Bark sound name")
    parser.add_argument("--no-bark", action="store_true", help="Force skip bark")
    parser.add_argument("--discord", action="store_true", help="Also send to Discord")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notifier = get_notifier(prefer_bark=not args.no_bark, include_discord=args.discord)
    notifier.send(
        title=args.title,
        body=args.body,
        level=args.level,
        sound=args.sound or None,
    )


if __name__ == "__main__":
    main()
