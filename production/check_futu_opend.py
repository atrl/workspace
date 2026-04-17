#!/usr/bin/env python3
"""Quick diagnostic: verify Futu OpenD is reachable before trading."""

import argparse
import socket
import sys


def check_port(host: str, port: int, timeout: float = 5.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Check Futu OpenD connectivity")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11111)
    args = parser.parse_args()

    if check_port(args.host, args.port):
        print(f"OK: OpenD is reachable at {args.host}:{args.port}")
        sys.exit(0)

    print(f"FAIL: OpenD is NOT reachable at {args.host}:{args.port}")
    print("""
Futu OpenD must be installed and running before using the Futu broker adapter.

Installation

1. macOS (automated)
   ```bash
   bash scripts/install_futu_opend_macos.sh
   ```
   Then edit `~/futu_opend/Futu_OpenD_10.2.6208_Mac/Futu_OpenD_10.2.6208_Mac/FutuOpenD.xml`
   with your Futu credentials and start OpenD:
   ```bash
   ~/futu_opend/Futu_OpenD_10.2.6208_Mac/Futu_OpenD_10.2.6208_Mac/FutuOpenD.app/Contents/MacOS/FutuOpenD
   ```

2. macOS (manual) / Windows / Linux
   - Download the command-line package from the official OpenAPI page.
   - Known working macOS direct URL pattern:
     `https://softwaredownload.futunn.com/Futu_OpenD_${VERSION}_Mac.tar.gz`
     (the `.zip` and `.dmg` variants currently return 404)
   - Windows: run `FutuOpenD.exe`
   - Linux: run `./FutuOpenD`

3. Login
   - Launch OpenD and log in with your Futu ID / phone number and password.
   - First-time login requires scanning a QR code with the 富途牛牛 app.
   - Complete the API questionnaire when prompted.
   - Ensure US stock simulated trading is enabled in your account.

4. Verify
   - Re-run this script after starting OpenD:
       qlib_venv/bin/python production/check_futu_opend.py
""")
    sys.exit(1)


if __name__ == "__main__":
    main()
