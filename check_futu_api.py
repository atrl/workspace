#!/usr/bin/env python3
"""
检查富途 API 和 SOXL 大单数据
"""
import sys

print("="*70)
print("富途 API 检查")
print("="*70)

# 1. 检查 futu-api 是否安装
print("\n1. 检查 futu-api 库...")
try:
    import futu
    print(f"   ✅ futu-api 已安装: {futu.__version__}")
except ImportError:
    print(f"   ❌ futu-api 未安装")
    print(f"   安装命令: pip install futu-api")
    print(f"   注意: 还需要安装富途 OpenD 网关")

# 2. 检查 OpenD 是否运行
print("\n2. 检查富途 OpenD 网关...")
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = sock.connect_ex(('127.0.0.1', 11111))
if result == 0:
    print("   ✅ OpenD 运行中 (port 11111)")
else:
    print("   ❌ OpenD 未运行")
    print("   下载地址: https://www.futunn.com/download/OpenD")
sock.close()

# 3. 替代方案
print("\n" + "="*70)
print("替代方案")
print("="*70)

print("""
方案1: 富途 API（需要账户和 OpenD）
  优势: 实时大单数据、资金流向
  劣势: 需要富途账户、OpenD 网关

方案2: Yahoo Finance（免费）
  数据: 价格、成交量
  劣势: 无大单分类

方案3: 东方财富/同花顺 API
  数据: 资金流向、大单净量
  劣势: 需要爬虫或第三方库

方案4: TradingView/Trading212
  数据: 实时数据
  劣势: 付费
""")

# 4. 如果有 futu-api，尝试获取数据
if 'futu' in sys.modules:
    print("\n" + "="*70)
    print("尝试获取 SOXL 数据")
    print("="*70)

    try:
        from futu import *

        # 连接 OpenD
        quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

        # SOXL 在美股
        # 富途股票代码格式: US.SOXL
        ret, data = quote_ctx.get_market_snapshot(['US.SOXL'])

        if ret == RET_OK:
            print("\n✅ SOXL 市场快照:")
            print(data)

            # 获取资金流向（如果有）
            # ret, flow_data = quote_ctx.get_capital_flow('US.SOXL')
            # if ret == RET_OK:
            #     print("\n✅ 资金流向:")
            #     print(flow_data)
        else:
            print(f"\n❌ 获取失败: {data}")

        quote_ctx.close()

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*70)
print("建议")
print("="*70)
print("""
如果需要 SOXL 大单数据：

1. 使用富途 API:
   - 安装: pip install futu-api
   - 下载 OpenD: https://www.futunn.com/download/OpenD
   - 文档: https://openapi.futunn.com/futu-api-doc/

2. 使用第三方数据源:
   - Alpha Vantage (免费额度)
   - IEX Cloud
   - Polygon.io

3. 自建特征:
   - 基于 OHLCV 计算成交量异常
   - 使用 VWAP 判断大单
   - 结合价格变动率估算资金强度
""")
