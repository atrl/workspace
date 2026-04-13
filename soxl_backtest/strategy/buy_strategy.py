from .base_strategy import BaseStrategy

class BuyStrategy(BaseStrategy):
    """买入策略"""
    
    def __init__(self, config, position_manager):
        super().__init__(config, position_manager)
        self.last_buy_was_cumulative = False
        self.cumulative_buy_cooldown = 0
        self.last_buy_conditions = []
        self.last_buy_date = None
        self.no_buy_reason = None
    
    def should_initial_build_position(self, current_data, df, current_idx):
        """判断是否应该进行首次建仓"""
        self.last_buy_conditions = []
        
        panic_score = self.get_panic_score(current_data)
        
        # 如果恐慌值超过最大阈值，暂停买入
        if panic_score <= self.config['MAX_PANIC_THRESHOLD']:
            self.no_buy_reason = f"恐慌指数({panic_score:.2f})低于最大阈值({self.config['MAX_PANIC_THRESHOLD']:.2f})"
            return False
            
        # 检查基本条件
        base_conditions = (
            panic_score <= self.config['BUY_SCORE_THRESHOLD']+10 and
            panic_score > self.config['MAX_PANIC_THRESHOLD']
        )
        
        if not base_conditions:
            self.no_buy_reason = f"恐慌指数({panic_score:.2f})不满足基本条件(≤{self.config['BUY_SCORE_THRESHOLD']+10:.2f}且>{self.config['MAX_PANIC_THRESHOLD']:.2f})"
            return False
            
        # 如果在冷却期内，不买入
        if self.cumulative_buy_cooldown > 0:
            self.no_buy_reason = f"在累积跌幅买入冷却期内，剩余{self.cumulative_buy_cooldown}天"
            return False
            
        # 计算当日跌幅
        prev_price = df.iloc[current_idx - 1]['price']
        daily_return = (current_data['price'] - prev_price) / prev_price
        
        # 计算最大累积跌幅
        max_cumulative_return = self.calculate_max_cumulative_return(df, current_idx, self.last_buy_date)
        
        # 获取当前波动率系数
        vol_coef = current_data['VOL_COEF']
            
        # 检查基本条件
        base_conditions = (
            panic_score <= self.config['BUY_SCORE_THRESHOLD']+10 and
            panic_score > self.config['MAX_PANIC_THRESHOLD']
        )
        
        if not base_conditions:
            self.no_buy_reason = f"恐慌指数({panic_score:.2f})不满足基本条件(≤{self.config['BUY_SCORE_THRESHOLD']+10:.2f}且>{self.config['MAX_PANIC_THRESHOLD']:.2f})"
            return False
        
        # 建仓时使用更严格的条件，当波动率中位数不为0时才应用波动率系数
        if self.config['VOLATILITY_MEDIAN'] != 0:
            daily_threshold = self.config['DAILY_RETURN_THRESHOLD'] * self.config['LOW_POSITION_THRESHOLD_MULTIPLIER'] * vol_coef
            cumulative_threshold = self.config['CUMULATIVE_RETURN_THRESHOLD'] * self.config['LOW_POSITION_THRESHOLD_MULTIPLIER'] * vol_coef
        else:
            daily_threshold = self.config['DAILY_RETURN_THRESHOLD'] * self.config['LOW_POSITION_THRESHOLD_MULTIPLIER']
            cumulative_threshold = self.config['CUMULATIVE_RETURN_THRESHOLD'] * self.config['LOW_POSITION_THRESHOLD_MULTIPLIER']
        
        # 检查买入条件
        daily_condition = daily_return <= daily_threshold
        cumulative_condition = max_cumulative_return <= cumulative_threshold
        
        # 记录触发的条件
        if daily_condition:
            self.last_buy_conditions.append({
                'type': 'daily_drop',
                'value': daily_return,
                'threshold': daily_threshold
            })
        if cumulative_condition:
            self.last_buy_conditions.append({
                'type': 'cumulative_drop',
                'value': max_cumulative_return,
                'threshold': cumulative_threshold,
                'cooldown_days': self.actual_cumulative_days
            })
        
        # 记录这次买入是否是通过累积跌幅条件
        if cumulative_condition and not daily_condition:
            self.last_buy_was_cumulative = True
            # 使用实际的累积天数作为冷却时间
            self.cumulative_buy_cooldown = self.actual_cumulative_days
        else:
            self.last_buy_was_cumulative = False
        
        return daily_condition or cumulative_condition

    def should_buy_overvalued(self, current_data, df, current_idx):
        """高估时的买入判断逻辑"""
        self.last_buy_conditions = []
        
        # 如果已经达到最大仓位，不买入
        if self.position_manager.position >= self.config['MAX_POSITION']:
            self.no_buy_reason = "已达到最大仓位"
            return False
            
        # 获取当前恐慌指数
        panic_score = self.get_panic_score(current_data)
        
        # 如果恐慌指数超过最大阈值，暂停买入
        if panic_score <= self.config['MAX_PANIC_THRESHOLD']:
            self.no_buy_reason = f"恐慌指数({panic_score:.2f})低于最大阈值({self.config['MAX_PANIC_THRESHOLD']:.2f})"
            return False
            
        # 如果在冷却期内，不买入
        if self.cumulative_buy_cooldown > 0:
            self.no_buy_reason = f"在累积跌幅买入冷却期内，剩余{self.cumulative_buy_cooldown}天"
            return False
            
        # 计算当前可用的买入仓位
        available_position = self.config['MAX_POSITION'] - self.position_manager.position
        
        # 如果可用仓位小于最小交易仓位，不买入
        if available_position < self.config['MIN_TRADE_POSITION']:
            self.no_buy_reason = f"可用仓位({available_position:.2%})小于最小交易仓位({self.config['MIN_TRADE_POSITION']:.2%})"
            return False
            
        # 如果恐慌指数低于阈值
        if panic_score <= self.config['BUY_SCORE_THRESHOLD']:
            # 计算日收益率
            if current_idx > 0:
                daily_return = (current_data['price'] - df.iloc[current_idx - 1]['price']) / df.iloc[current_idx - 1]['price']
            else:
                daily_return = 0
                
            # 计算最大累积跌幅
            max_cumulative_return = self.calculate_max_cumulative_return(df, current_idx, self.last_buy_date)
            
            # 获取当前波动率系数
            vol_coef = current_data['VOL_COEF']
            
            # 记录恐慌指数条件
            self.last_buy_conditions.append({
                'type': 'panic_score',
                'value': panic_score,
                'threshold': self.config['BUY_SCORE_THRESHOLD']
            })
            
            # 判断日跌幅条件，当波动率中位数不为0时才应用波动率系数
            if self.config['VOLATILITY_MEDIAN'] != 0:
                daily_condition = daily_return <= self.config['DAILY_RETURN_THRESHOLD'] * vol_coef
            else:
                daily_condition = daily_return <= self.config['DAILY_RETURN_THRESHOLD']
            if daily_condition:
                self.last_buy_conditions.append({
                    'type': 'daily_drop',
                    'value': daily_return,
                    'threshold': self.config['DAILY_RETURN_THRESHOLD'] * vol_coef
                })
                self.last_buy_was_cumulative = False
                self.cumulative_buy_cooldown = 1  # 日跌幅买入后冷却时间为1天
                self.last_buy_date = df.iloc[current_idx]['date']
                return True
            
            # 如果日跌幅不满足，再判断累积跌幅条件，当波动率中位数不为0时才应用波动率系数
            if self.config['VOLATILITY_MEDIAN'] != 0:
                cumulative_condition = max_cumulative_return <= self.config['CUMULATIVE_RETURN_THRESHOLD'] * vol_coef
            else:
                cumulative_condition = max_cumulative_return <= self.config['CUMULATIVE_RETURN_THRESHOLD']
            if cumulative_condition:
                self.last_buy_conditions.append({
                    'type': 'cumulative_drop',
                    'value': max_cumulative_return,
                    'threshold': self.config['CUMULATIVE_RETURN_THRESHOLD'] * vol_coef,
                    'cooldown_days': self.actual_cumulative_days
                })
                self.last_buy_was_cumulative = True
                self.cumulative_buy_cooldown = self.actual_cumulative_days  # 累积跌幅买入使用实际累积天数作为冷却时间
                self.last_buy_date = df.iloc[current_idx]['date']
                return True
            
            # 如果恐慌指数满足但跌幅条件都不满足
            self.no_buy_reason = f"恐慌指数满足({panic_score:.2f})，但日跌幅({daily_return:.2%})和累积跌幅({max_cumulative_return:.2%})均不满足条件"
                
        else:
            # 恐慌指数不满足
            self.no_buy_reason = f"恐慌指数({panic_score:.2f})不满足买入阈值({self.config['BUY_SCORE_THRESHOLD']:.2f})"
        return False

    def should_buy_undervalued(self, current_data, df, current_idx):
        """低估时的买入判断逻辑"""
        self.last_buy_conditions = []
        
        # 如果已经达到最大仓位，不买入
        if self.position_manager.position >= self.config['MAX_POSITION']:
            self.no_buy_reason = "已达到最大仓位"
            return False
            
        # 获取当前恐慌指数
        panic_score = self.get_panic_score(current_data)
        
        # 如果恐慌指数超过最大阈值，暂停买入
        if panic_score <= self.config['MAX_PANIC_THRESHOLD']:
            self.no_buy_reason = f"恐慌指数({panic_score:.2f})低于最大阈值({self.config['MAX_PANIC_THRESHOLD']:.2f})"
            return False
            
        # 如果在冷却期内，不买入
        if self.cumulative_buy_cooldown > 0:
            self.no_buy_reason = f"在累积跌幅买入冷却期内，剩余{self.cumulative_buy_cooldown}天"
            return False
            
        # 计算当前可用的买入仓位
        available_position = self.config['MAX_POSITION'] - self.position_manager.position
        
        # 如果可用仓位小于最小交易仓位，不买入
        if available_position < self.config['MIN_TRADE_POSITION']:
            self.no_buy_reason = f"可用仓位({available_position:.2%})小于最小交易仓位({self.config['MIN_TRADE_POSITION']:.2%})"
            return False
            
        # 如果恐慌指数低于阈值
        if panic_score <= self.config['BUY_SCORE_THRESHOLD']:
            # 计算日收益率
            if current_idx > 0:
                daily_return = (current_data['price'] - df.iloc[current_idx - 1]['price']) / df.iloc[current_idx - 1]['price']
            else:
                daily_return = 0
                
            # 计算最大累积跌幅
            max_cumulative_return = self.calculate_max_cumulative_return(df, current_idx, self.last_buy_date)
            
            # 获取当前波动率系数
            vol_coef = current_data['VOL_COEF']
            
            # 记录恐慌指数条件
            self.last_buy_conditions.append({
                'type': 'panic_score',
                'value': panic_score,
                'threshold': self.config['BUY_SCORE_THRESHOLD']
            })
            
            # 判断日跌幅条件，当波动率中位数不为0时才应用波动率系数
            if self.config['VOLATILITY_MEDIAN'] != 0:
                daily_condition = daily_return <= self.config['DAILY_RETURN_THRESHOLD'] * vol_coef
            else:
                daily_condition = daily_return <= self.config['DAILY_RETURN_THRESHOLD']
            if daily_condition:
                self.last_buy_conditions.append({
                    'type': 'daily_drop',
                    'value': daily_return,
                    'threshold': self.config['DAILY_RETURN_THRESHOLD'] * vol_coef
                })
                self.last_buy_was_cumulative = False
                self.cumulative_buy_cooldown = 1  # 日跌幅买入后冷却时间为1天
                self.last_buy_date = df.iloc[current_idx]['date']
                return True
            
            # 如果日跌幅不满足，再判断累积跌幅条件，当波动率中位数不为0时才应用波动率系数
            if self.config['VOLATILITY_MEDIAN'] != 0:
                cumulative_condition = max_cumulative_return <= self.config['CUMULATIVE_RETURN_THRESHOLD'] * vol_coef
            else:
                cumulative_condition = max_cumulative_return <= self.config['CUMULATIVE_RETURN_THRESHOLD']
            if cumulative_condition:
                self.last_buy_conditions.append({
                    'type': 'cumulative_drop',
                    'value': max_cumulative_return,
                    'threshold': self.config['CUMULATIVE_RETURN_THRESHOLD'] * vol_coef,
                    'cooldown_days': self.actual_cumulative_days
                })
                self.last_buy_was_cumulative = True
                self.cumulative_buy_cooldown = self.actual_cumulative_days  # 累积跌幅买入使用实际累积天数作为冷却时间
                self.last_buy_date = df.iloc[current_idx]['date']
                return True
            
            # 如果恐慌指数满足但跌幅条件都不满足
            self.no_buy_reason = f"恐慌指数满足({panic_score:.2f})，但日跌幅({daily_return:.2%})和累积跌幅({max_cumulative_return:.2%})均不满足条件"
                
        else:
            # 恐慌指数不满足
            self.no_buy_reason = f"恐慌指数({panic_score:.2f})不满足买入阈值({self.config['BUY_SCORE_THRESHOLD']:.2f})"
        return False
        
    def get_last_buy_conditions(self):
        """获取最近一次买入的触发条件"""
        return self.last_buy_conditions

    def should_buy(self, current_data, df, current_idx):
        """判断是否应该买入"""
        # 重置未买入原因
        self.no_buy_reason = None
        
        # 如果还没有完成初始建仓，使用初始建仓逻辑
        if not self.position_manager.initial_position_built:
            result = self.should_initial_build_position(current_data, df, current_idx)
            return result
            
        # 判断是高估还是低估（这里暂时用一个简单的判断，后续可以修改）
        is_overvalued = current_data['price'] > current_data['MA20']
        
        if is_overvalued:
            result = self.should_buy_overvalued(current_data, df, current_idx)
            return result
        else:
            result = self.should_buy_undervalued(current_data, df, current_idx)
            return result
            
    def get_no_buy_reason(self):
        """获取未买入的原因"""
        return self.no_buy_reason