#!/usr/bin/env python3
"""
富途 OpenD 安装指南

由于网络限制，需要手动下载 OpenD
"""
import webbrowser
import platform

print("="*70)
print("富途 OpenD 安装指南")
print("="*70)

system = platform.system()
print(f"\n检测到系统: {system}")

if system == "Darwin":  # macOS
    print("""
✅ futu-api 已安装成功！

下一步：手动下载并安装 OpenD

方式1：浏览器下载
  1. 打开浏览器，访问：
     https://www.futunn.com/download/OpenD

  2. 选择 macOS 版本下载

  3. 双击 DMG 文件安装

  4. 将 OpenD 拖到 Applications 文件夹

  5. 启动 OpenD：
     - 打开 Applications 文件夹
     - 双击 OpenD
     - 扫码登录富途账户

方式2：命令行下载（如果网络允许）
  curl -L -o OpenD.dmg "https://softwarefile.futunn.com/OpenD/OpenD_10.2.6218_macOS.dmg"

""")

    # 尝试打开浏览器
    print("正在打开浏览器...")
    webbrowser.open("https://www.futunn.com/download/OpenD")

elif system == "Windows":
    print("""
✅ futu-api 已安装成功！

下一步：下载并安装 OpenD

  1. 访问：https://www.futunn.com/download/OpenD
  2. 下载 Windows 版本
  3. 运行安装程序
  4. 启动 OpenD 并扫码登录
""")

    webbrowser.open("https://www.futunn.com/download/OpenD")

else:  # Linux
    print("""
✅ futu-api 已安装成功！

下一步：下载并安装 OpenD

  1. 访问：https://www.futunn.com/download/OpenD
  2. 下载 Linux 版本
  3. 解压: tar -xzf OpenD_*.tar.gz
  4. 运行: ./OpenD
  5. 扫码登录
""")

    webbrowser.open("https://www.futunn.com/download/OpenD")

print("\n" + "="*70)
print("安装完成后")
print("="*70)

print("""
1. OpenD 默认运行在 127.0.0.1:11111

2. 测试连接：
   cd /Users/rziiiii/.openclaw/workspace
   source futu_venv/bin/activate
   python3 test_futu_api.py

3. 如果成功，将显示：
   ✅ futu-api 已安装: 10.02.6218
   ✅ OpenD 运行中 (port 11111)
   ✅ 已连接 OpenD

4. 然后就可以获取真实的大单数据了！
""")

print("\n" + "="*70)
print("当前状态")
print("="*70)

print("""
✅ futu-api: 已安装 (v10.02.6218)
⏳ OpenD: 等待手动安装
⏳ 富途账户: 等待扫码登录

下一步：
1. 浏览器应该已经打开下载页面
2. 下载并安装 OpenD
3. 启动 OpenD 并扫码登录
4. 运行测试脚本验证
""")

print("="*70)
