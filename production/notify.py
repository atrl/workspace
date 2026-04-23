#!/usr/bin/env python3
"""统一通知模块：支持 Bark (iOS)、macOS 原生通知、控制台降级。

环境变量:
    BARK_KEY            - Bark 设备 key（优先）
    BARK_DEVICE_KEY     - Bark 设备 key（兼容 OpenClaw 默认）

用法:
    from production.notify import get_notifier
    notifier = get_notifier()
    notifier.send("标题", "内容", level="info")
"""
from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
import json
from abc import ABC, abstractmethod
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


def _bark_key() -> Optional[str]:
    return os.environ.get("BARK_KEY") or os.environ.get("BARK_DEVICE_KEY")


def get_notifier(prefer_bark: bool = True) -> Notifier:
    """按优先级返回可用通知器。

    Priority:
        1. Bark (若 BARK_KEY/BARK_DEVICE_KEY 存在且 prefer_bark=True)
        2. macOS native (若运行在 macOS)
        3. Console (降级)
    """
    key = _bark_key()
    if prefer_bark and key:
        return BarkNotifier(key)
    if sys.platform == "darwin":
        return MacOSNotifier()
    return ConsoleNotifier()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Test notification channels")
    parser.add_argument("--title", default="SOXL Paper Trading Test")
    parser.add_argument("--body", default="Notification channel is working.")
    parser.add_argument("--level", default="info", choices=["info", "warning", "error", "critical"])
    parser.add_argument("--sound", default="", help="Bark sound name")
    parser.add_argument("--no-bark", action="store_true", help="Force skip bark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notifier = get_notifier(prefer_bark=not args.no_bark)
    notifier.send(
        title=args.title,
        body=args.body,
        level=args.level,
        sound=args.sound or None,
    )


if __name__ == "__main__":
    main()
