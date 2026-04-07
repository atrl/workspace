# Futu API Skill - 富途 OpenAPI

这个 skill 用于通过富途 API 获取真实的大单数据、资金流向等信息。

## 安装步骤

### 1. 安装 futu-api

```bash
pip install futu-api
```

### 2. 下载并安装 OpenD

- 下载地址：https://www.futunn.com/download/OpenD
- 支持 Windows / macOS / Linux

### 3. 启动 OpenD

```bash
# macOS/Linux
./OpenD

# Windows
OpenD.exe
```

OpenD 默认运行在 `127.0.0.1:11111`

### 4. 配置

首次启动需要扫码登录富途账户。

## 功能

### 可用接口

1. **行情数据**
   - 实时行情
   - 历史K线
   - 市场快照

2. **资金流向**
   - 资金流向数据
   - 经纪商席位

3. **逐笔成交**
   - 逐笔交易数据
   - 大单识别

4. **订单簿**
   - 买卖盘
   - 经纪商队列

## 示例代码

### 获取 SOXL 行情

```python
from futu import *

# 连接 OpenD
quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

# 获取 SOXL 市场快照（美股代码：US.SOXL）
ret, data = quote_ctx.get_market_snapshot(['US.SOXL'])
if ret == RET_OK:
    print(data)

# 获取历史K线
ret, data = quote_ctx.get_history_kline('US.SOXL')
if ret == RET_OK:
    print(data)

quote_ctx.close()
```

### 获取资金流向

```python
from futu import *

quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

# 获取资金流向
ret, data = quote_ctx.get_capital_flow('US.SOXL')
if ret == RET_OK:
    print(data)

quote_ctx.close()
```

### 获取逐笔成交（大单识别）

```python
from futu import *

quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

# 获取逐笔成交
ret, data = quote_ctx.get_rt_data('US.SOXL')
if ret == RET_OK:
    # 识别大单（例如：成交量 > 阈值）
    large_orders = data[data['volume'] > 10000]
    print(large_orders)

quote_ctx.close()
```

## 支持的市场

- 港股：HK.00700（腾讯）、HK.09988（阿里）等
- 美股：US.AAPL、US.TSLA、US.SOXL 等
- A股：SH.600000、SZ.000001 等

## 注意事项

1. 需要富途账户
2. OpenD 必须保持运行
3. 部分数据需要订阅权限
4. 注意 API 调用频率限制

## 官方文档

- Futu API 文档：https://openapi.futunn.com/futu-api-doc/
- GitHub：https://github.com/FutunnOpen/py-futu-api
