class BacktestConfig:
    def __init__(self, params=None):
        """初始化回测配置"""
        self.params = self.get_default_params()
        if params:
            self.params.update(params)
            
        # 验证必要参数
        missing_params = []
        for key in self.get_default_params().keys():
            if key not in self.params:
                missing_params.append(key)
        
        if missing_params:
            raise ValueError(f"缺失的参数: {', '.join(missing_params)}")
            
    def get_default_params(self):
        """获取默认参数"""
        return {
            'USE_LONGBRIDGE': True,  # 控制是否使用长桥API获取数据
            'STOCK_CODE': "SOXL",  # 股票代码
            'START_DATE': "2020-01-01",  # 回测开始时间
            'END_DATE': None,  # 回测结束时间，默认为当前日期
            'INITIAL_CAPITAL': 100000,  # 初始资金
            
            # VIX相关参数
            'VIX_LOW': 10,
            'VIX_HIGH': 30,
            'FEAR_MIN': -100,
            'FEAR_MAX': 100,
            'SIGMOID_STEEPNESS': 0.5,
            # 波动率参数
            'VOLATILITY_MEDIAN': 1.0,  # 波动率中位数
            
            # 仓位控制
            'MAX_POSITION': 0.9,        # 最大仓位
            'MIN_POSITION': 0.5,        # 最小仓位（卖出时保留的仓位）
            'MIN_TRADE_POSITION': 0.05, # 最小交易仓位（买入或卖出的最小仓位）
            'BUY_STEP_INITIAL': 0.3,    # 初始建仓时的买入步长（仓位小于最小仓位时）
            'BUY_STEP_NORMAL': 0.1,     # 正常加仓时的买入步长（仓位大于最小仓位时）
            'SELL_AMOUNT': 0.1,         # 每次卖出的仓位
            'POSITION_UPDATE_DAYS': 20,  # 仓位更新频率（每隔多少天更新一次）
            'LOW_POSITION_THRESHOLD_MULTIPLIER': 1,  # 低持仓时的跌幅阈值调整系数
            
            # 买入条件
            'BUY_SCORE_THRESHOLD': -50,  # 买入恐慌值阈值
            'MAX_PANIC_THRESHOLD': -300,  # 最大恐慌阈值，超过此值暂停买入
            'DAILY_RETURN_THRESHOLD': -0.15,  # 买入所需的最小日跌幅（-8%）
            'CUMULATIVE_RETURN_THRESHOLD': -0.30,  # 买入所需的最小累积跌幅（-15%）
            'CUMULATIVE_DAYS': 30,       # 计算累积跌幅的天数
            'DAILY_BUY_COOLDOWN': 3,
            
            # 卖出条件
            'SELL_SCORE_THRESHOLD': 85,   # 卖出恐慌值阈值
            'SELL_PROFIT_TARGET': 0.20,   # 整体仓位获利目标
            'LAST_BUY_PROFIT_TARGET': 0.30,  # 最近一次买入的获利目标
            'STOP_LOSS_THRESHOLD': -1,  # 止损阈值（-20%）
            'STOP_LOSS_POSITION': 0.6,    # 止损时卖出仓位比例
            'TAKE_PROFIT_THRESHOLD': 0.60,  # 止盈阈值（60%）
            'TAKE_PROFIT_POSITION': 0.9,   # 止盈时卖出仓位比例
        }
        
    def __getitem__(self, key):
        """允许使用字典方式访问参数"""
        return self.params[key]
        
    def get(self, key, default=None):
        """获取参数值，如果不存在返回默认值"""
        return self.params.get(key, default) 