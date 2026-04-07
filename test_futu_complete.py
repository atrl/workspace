#!/usr/bin/env python3
"""
富途 API 完整测试 - SOXL 大单数据

测试步骤：
1. 检查 futu-api 是否安装
2. 检查 OpenD 是否运行
3. 连接 OpenD
4. 获取 SOXL 数据
"""
import sys
sys.path.insert(0, '/Users/rziiiii/.openclaw/workspace/futu_venv/lib/python3.14/site-packages')

import socket

print("="*70)
print("富途 API 完整测试 - SOXL 大单数据")
print("="*70)

# 1. 检查 futu-api
print("\n1. 检查 futu-api...")
try:
    import futu
    print(f"   ✅ futu-api 已安装: {futu.__version__}")
    FUTU_AVAILABLE = True
except ImportError as e:
    print(f"   ❌ futu-api 未安装: {e}")
    FUTU_AVAILABLE = False

# 2. 检查 OpenD
print("\n2. 检查 OpenD 网关...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = sock.connect_ex(('127.0.0.1', 11111))
if result == 0:
    print("   ✅ OpenD 运行中 (port 11111)")
    OPEND_RUNNING = True
else:
    print("   ❌ OpenD 未运行")
    print("\n   安装步骤：")
    print("   1. 访问：https://www.futunn.com/download/OpenD")
    print("   2. 下载 macOS 版本")
    print("   3. 安装并启动 OpenD")
    print("   4. 扫码登录富途账户")
    print("   5. 重新运行此脚本")
    OPEND_RUNNING = False
sock.close()

# 3. 如果条件满足，尝试获取数据
if FUTU_AVAILABLE and OPEND_RUNNING:
    print("\n" + "="*70)
    print("3. 获取 SOXL 真实数据")
    print("="*70)

    try:
        from futu import *

        # 连接 OpenD
        quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
        print("   ✅ 已连接 OpenD")

        # SOXL 在美股，代码为 US.SOXL
        symbol = 'US.SOXL'

        # 3.1 获取市场快照
        print(f"\n   3.1 获取 {symbol} 市场快照...")
        ret, data = quote_ctx.get_market_snapshot([symbol])
        if ret == RET_OK:
            print(f"   ✅ 成功获取市场快照")
            if len(data) > 0:
                row = data.iloc[0]
                print(f"      价格: ${row['last_price']:.2f}")
                print(f"      成交量: {row['volume']:,.0f}")
                print(f"      涨跌幅: {row['change_rate']:.2%}")
        else:
            print(f"   ❌ 获取失败: {data}")

        # 3.2 获取资金流向（大单数据）
        print(f"\n   3.2 获取 {symbol} 资金流向（大单分类）...")
        ret, data = quote_ctx.get_capital_flow(symbol, start='2026-03-01')
        if ret == RET_OK:
            print(f"   ✅ 成功获取资金流向")
            if len(data) > 0:
                print(f"      数据列: {list(data.columns)}")
                print(f"\n      最近5天资金流向:")
                for idx, row in data.tail().iterrows():
                    print(f"      {idx}:")
                    print(f"        主力净流入: {row.get('main_net_inflow', 'N/A')}")
                    print(f"        大单净流入: {row.get('huge_net_inflow', 'N/A')}")
        else:
            print(f"   ❌ 获取失败: {data}")

        # 3.3 获取逐笔成交（实时大单）
        print(f"\n   3.3 获取 {symbol} 逐笔成交（识别大单）...")
        ret, data = quote_ctx.get_rt_data(symbol)
        if ret == RET_OK:
            print(f"   ✅ 成功获取逐笔成交")
            if len(data) > 0:
                print(f"      总成交笔数: {len(data)}")

                # 识别大单（成交量 > 阈值）
                threshold = 10000  # 1万股
                large_orders = data[data['volume'] > threshold]
                print(f"      大单笔数（>{threshold}股）: {len(large_orders)}")

                if len(large_orders) > 0:
                    print(f"\n      最大5笔大单:")
                    top_orders = large_orders.nlargest(5, 'volume')
                    for idx, row in top_orders.iterrows():
                        direction = "买入" if row['ticker_direction'] == 1 else "卖出"
                        print(f"        {direction} {row['volume']:,.0f}股 @ ${row['price']:.2f}")
        else:
            print(f"   ❌ 获取失败: {data}")

        # 3.4 获取经纪商队列（识别机构）
        print(f"\n   3.4 获取 {symbol} 经纪商队列...")
        ret, data = quote_ctx.get_broker_queue(symbol)
        if ret == RET_OK:
            print(f"   ✅ 成功获取经纪商队列")
            if len(data) > 0:
                print(f"      买盘经纪商: {len(data[data['side'] == 'BID'])}家")
                print(f"      卖盘经纪商: {len(data[data['side'] == 'ASK'])}家")
        else:
            print(f"   ❌ 获取失败: {data}")

        # 关闭连接
        quote_ctx.close()
        print(f"\n   ✅ 已断开 OpenD 连接")

        print("\n" + "="*70)
        print("✅ 测试完成 - 成功获取真实大单数据")
        print("="*70)

    except Exception as e:
        print(f"\n   ❌ 错误: {e}")
        import traceback
        traceback.print_exc()

else:
    print("\n" + "="*70)
    print("❌ 无法测试 - 前置条件不满足")
    print("="*70)

    if not FUTU_AVAILABLE:
        print("\n   需要安装 futu-api:")
        print("   python3 -m venv futu_venv")
        print("   source futu_venv/bin/activate")
        print("   pip install futu-api")

    if not OPEND_RUNNING:
        print("\n   需要启动 OpenD:")
        print("   1. 下载: https://www.futunn.com/download/OpenD")
        print("   2. 安装并启动 OpenD")
        print("   3. 扫码登录富途账户")

print("\n" + "="*70)
