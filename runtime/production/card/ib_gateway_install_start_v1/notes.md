# IB Gateway Install And Startup Proof

Date: 2026-05-24

Goal:

- Install and start IB Gateway for the repo-owned IBKR live broker path.
- Keep scheduled simulated trading on Futu; this card only addresses IBKR
  Gateway readiness.

Source:

- Official IBKR Gateway download page:
  `https://www.interactivebrokers.com/en/trading/ib-gateway-download.php`
- Apple Silicon stable standalone DMG:
  `https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-macos-arm.dmg`
- Downloaded file:
  `/tmp/ibkr_gateway_install/ibgateway-stable-standalone-macos-arm.dmg`
- SHA256:
  `50cd44800e812343c20d43e6e9cced977186c7972ff091fece8f2e3fe2e8c79f`

Install:

```text
hdiutil attach -nobrowse /tmp/ibkr_gateway_install/ibgateway-stable-standalone-macos-arm.dmg
# mounted /Volumes/IB Gateway 10.45

"/Volumes/IB Gateway 10.45/IB Gateway 10.45 Installer.app/Contents/MacOS/JavaApplicationStub" \
  -q -dir "/Applications/IB Gateway 10.45"
# exit 0
# 安装目录设定至 /Applications/IB Gateway 10.45 。
# 文件解压中...
# 正在完成安装...
```

Installed app:

```text
/Applications/IB Gateway 10.45/.install4j/.installSuccess
/Applications/IB Gateway 10.45/IB Gateway 10.45.app
/Applications/IB Gateway 10.45/IB Gateway 10.45 Uninstaller.app
du -sh /Applications/IB Gateway 10.45
# 493M
```

Startup:

```text
open "/Applications/IB Gateway 10.45/IB Gateway 10.45.app"

ps -p 13873 -o pid,ppid,stat,etime,command
# PID  PPID STAT ELAPSED COMMAND
# 13873     1 S      01:28 /Applications/IB Gateway 10.45/IB Gateway 10.45.app/Contents/MacOS/JavaApplicationStub
```

UI state:

- Window title: `IBKR网关`
- API type was switched to `IB API`.
- Trading mode remained on the login screen's current selection.
- No username, password, or 2FA was entered by the agent.

Connectivity check before login:

```text
for p in 7496 7497 4001 4002; do nc -z 127.0.0.1 "$p" ...; done
# 7496 closed
# 7497 closed
# 4001 closed
# 4002 closed

PYTHONPATH=. qlib_venv/bin/python production/check_ibkr_gateway.py --host 127.0.0.1 --port 4002 --timeout 2 --port-only
# FAIL: IBKR TWS/Gateway is NOT reachable at 127.0.0.1:4002

PYTHONPATH=. qlib_venv/bin/python production/check_ibkr_gateway.py --host 127.0.0.1 --port 7497 --timeout 2 --port-only
# FAIL: IBKR TWS/Gateway is NOT reachable at 127.0.0.1:7497
```

Decision:

- IB Gateway is installed and running locally.
- The remaining blocker is interactive IBKR login and any required 2FA/API
  socket enablement inside Gateway.
- Do not treat IBKR as live-ready until `production/check_ibkr_gateway.py`
  passes against the logged-in Gateway port.
- Do not switch the scheduled Futu simulated-trading path to IBKR.
