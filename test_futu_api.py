#!/usr/bin/env python3
"""
富途 API 测试 - SOXL 大单数据

功能：
1. 检查 futu-api 是否安装
2. 检查 OpenD 是否运行
3. 尝试获取 SOXL 数据
4. 获取资金流向和大单数据

说明：
- 这是探索性环境检查脚本，不是生产执行入口
- 正式执行层应通过 `src/soxl_quant/runtime.py` 的 broker adapter 接口接入
"""
import sys
import socket

print("="*70)
print("富途 API 测试 - SOXL 大单数据")
print("="*70)

# 1. 检查 futu-api
print("\n1. 检查 futu-api...")
try:
    import futu
    print(f"   ✅ futu-api 已安装: {futu.__version__}")
    FUTU_AVAILABLE = True
except ImportError:
    print(f"   ❌ futu-api 未安装")
    print(f"   安装命令: pip install futu-api")
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
    print("   下载地址: https://www.futunn.com/download/OpenD")
    OPEND_RUNNING = False
sock.close()

# 3. 如果条件满足，尝试获取数据
if FUTU_AVAILABLE and OPEND_RUNNING:
    print("\n" + "="*70)
    print("3. 获取 SOXL 数据")
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
            print(f"   数据列: {list(data.columns)}")
            if len(data) > 0:
                print(f"   价格: ${data['last_price'].iloc[0]:.2f}")
                print(f"   成交量: {data['volume'].iloc[0]:,.0f}")
        else:
            print(f"   ❌ 获取失败: {data}")

        # 3.2 获取资金流向
        print(f"\n   3.2 获取 {symbol} 资金流向...")
        ret, data = quote_ctx.get_capital_flow(symbol, start='2026-01-01')
        if ret == RET_OK:
            print(f"   ✅ 成功获取资金流向")
            print(f"   数据列: {list(data.columns)}")
            if len(data) > 0:
                print(f"   最近5天:")
                print(data.tail())
        else:
            print(f"   ❌ 获取失败: {data}")

        # 3.3 获取逐笔成交（实时）
        print(f"\n   3.3 获取 {symbol} 逐笔成交...")
        ret, data = quote_ctx.get_rt_data(symbol)
        if ret == RET_OK:
            print(f"   ✅ 成功获取逐笔成交")
            print(f"   数据列: {list(data.columns)}")
            if len(data) > 0:
                # 识别大单（成交量 > 10000）
                large_orders = data[data['volume'] > 10000]
                print(f"   总成交笔数: {len(data)}")
                print(f"   大单笔数（>10000股）: {len(large_orders)}")
                if len(large_orders) > 0:
                    print(f"   大单示例:")
                    print(large_orders.head())
        else:
            print(f"   ❌ 获取失败: {data}")

        # 3.4 获取经纪商队列
        print(f"\n   3.4 获取 {symbol} 经纪商队列...")
        ret, data = quote_ctx.get_broker_queue(symbol)
        if ret == RET_OK:
            print(f"   ✅ 成功获取经纪商队列")
            print(f"   数据列: {list(data.columns)}")
            if len(data) > 0:
                print(f"   前5个经纪商:")
                print(data.head())
        else:
            print(f"   ❌ 获取失败: {data}")

        # 关闭连接
        quote_ctx.close()
        print(f"\n   ✅ 已断开 OpenD 连接")

    except Exception as e:
        print(f"\n   ❌ 错误: {e}")
        import traceback
        traceback.print_exc()

else:
    print("\n" + "="*70)
    print("3. 无法获取数据（前置条件不满足）")
    print("="*70)

    if not FUTU_AVAILABLE:
        print("\n   需要安装 futu-api:")
        print("   pip install futu-api")

    if not OPEND_RUNNING:
        print("\n   需要启动 OpenD:")
        print("   1. 下载: https://www.futunn.com/download/OpenD")
        print("   2. 解压并运行: ./OpenD")
        print("   3. 扫码登录富途账户")

print("\n" + "="*70)
print("富途 API 可用功能")
print("="*70)

if FUTU_AVAILABLE:
    from futu import *
    print("""
富途 API 支持的大单相关数据：

1. 资金流向（get_capital_flow）
   - 主力资金流入/流出
   - 大单、中单、小单分类
   - 净流入/流出金额

2. 逐笔成交（get_rt_data）
   - 每笔交易的详细信息
   - 成交价格、成交量
   - 买卖方向

3. 经纪商队列（get_broker_queue）
   - 买卖盘经纪商
   - 识别机构席位

4. 订单簿（get_order_book）
   - 买卖盘深度
   - 大单挂单

5. 历史K线（get_history_kline）
   - 历史价格和成交量
   - 多种周期（1分、5分、日K等）

这些是真实的逐笔数据，不是估算！
""")
else:
    print("""
需要安装富途 API 才能获取真实大单数据：

1. pip install futu-api
2. 下载并启动 OpenD
3. 扫码登录富途账户
""")

print("="*70)
