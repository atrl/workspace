# Futu API Skill - 富途 OpenAPI

这个 skill 用于通过富途 API 获取真实的大单数据、资金流向等信息。

## 安装步骤

### 1. 安装 futu-api

```bash
pip install futu-api
```

### 2. 下载并安装 OpenD

#### macOS（自动安装）

```bash
bash scripts/install_futu_opend_macos.sh
```

安装完成后，编辑配置文件填入富途账号密码：

```bash
~/futu_opend/Futu_OpenD_10.2.6208_Mac/Futu_OpenD_10.2.6208_Mac/FutuOpenD.xml
```

然后启动 OpenD：

```bash
~/futu_opend/Futu_OpenD_10.2.6208_Mac/Futu_OpenD_10.2.6208_Mac/FutuOpenD.app/Contents/MacOS/FutuOpenD
```

#### macOS（手动）/ Windows / Linux

- 官方 macOS 直接下载链接格式（`.tar.gz` 可用，`.zip` / `.dmg` 目前 404）：
  ```
  https://softwaredownload.futunn.com/Futu_OpenD_${VERSION}_Mac.tar.gz
  ```
- Windows：运行 `FutuOpenD.exe`
- Linux：运行 `./FutuOpenD`

OpenD 默认运行在 `127.0.0.1:11111`。

首次启动需要扫码登录富途账户，并完成 API 问卷评估。

### 3. 连接检测

在运行策略前，建议先检测 OpenD 是否可达：

```bash
qlib_venv/bin/python production/check_futu_opend.py
```

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
5. macOS 可通过 `scripts/install_futu_opend_macos.sh` 自动安装

## 官方文档

- Futu API 文档：https://openapi.futunn.com/futu-api-doc/
- GitHub：https://github.com/FutunnOpen/py-futu-api
