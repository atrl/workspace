from .base_strategy import BaseStrategy

class SellStrategy(BaseStrategy):
    """卖出策略"""
    
    def __init__(self, config, position_manager):
        super().__init__(config, position_manager)
        self.last_sell_conditions = []
        self.no_sell_reason = None
        
    def get_no_sell_reason(self):
        """获取未卖出的原因"""
        return self.no_sell_reason
    
    def should_sell(self, current_data):
        """判断是否应该卖出"""
        self.last_sell_conditions = []
        self.no_sell_reason = None
        panic_score = self.get_panic_score(current_data)
        profit = (current_data['price'] - self.position_manager.avg_price) / self.position_manager.avg_price if self.position_manager.avg_price > 0 else 0
        
        # 获取当前波动率系数
        vol_coef = current_data['VOL_COEF']
        
        # 计算可卖出的最大仓位
        max_sell_position = self.position_manager.position - self.config['MIN_POSITION']
        
        # 如果可卖出仓位小于最小交易仓位，不卖出
        if max_sell_position < self.config['MIN_TRADE_POSITION']:
            self.no_sell_reason = f"可卖出仓位({max_sell_position:.2%})小于最小交易仓位({self.config['MIN_TRADE_POSITION']:.2%})"
            return False
        
        # 止损
        if profit <= self.config['STOP_LOSS_THRESHOLD']:
            self.last_sell_conditions.append({
                'type': 'stop_loss',
                'value': profit,
                'threshold': self.config['STOP_LOSS_THRESHOLD']
            })
            return 'stop_loss'
            
        # 获取最后一次买入的持仓信息
        last_buy = self.position_manager.get_last_position()
        if last_buy:
            last_buy_profit = (current_data['price'] - last_buy['price']) / last_buy['price']
        else:
            last_buy_profit = 0
            
        # 大幅止盈条件（卖出95%仓位），应用波动率系数
        take_profit_condition = (
            panic_score >= self.config['SELL_SCORE_THRESHOLD'] and 
            profit*self.position_manager.position >= self.config['TAKE_PROFIT_THRESHOLD']  and 
            self.position_manager.position >= self.config['MIN_POSITION']
        )
        
        if take_profit_condition:
            if (self.position_manager.position * self.config['TAKE_PROFIT_POSITION']) >= self.config['MIN_TRADE_POSITION']:
                self.last_sell_conditions.extend([
                    {
                        'type': 'panic_score',
                        'value': panic_score,
                        'threshold': self.config['SELL_SCORE_THRESHOLD']
                    },
                    {
                        'type': 'take_profit',
                        'value': profit*self.position_manager.position,
                        'threshold': self.config['TAKE_PROFIT_THRESHOLD'] 
                    }
                ])
                return 'take_profit'
            else:
                self.no_sell_reason = f"止盈条件满足，但卖出仓位({self.position_manager.position * self.config['TAKE_PROFIT_POSITION']:.2%})小于最小交易仓位({self.config['MIN_TRADE_POSITION']:.2%})"
        elif panic_score < self.config['SELL_SCORE_THRESHOLD']:
            self.no_sell_reason = f"恐慌指数({panic_score:.2f})低于卖出阈值({self.config['SELL_SCORE_THRESHOLD']:.2f})"
        elif profit*self.position_manager.position < self.config['TAKE_PROFIT_THRESHOLD']:
            self.no_sell_reason = f"总体收益({profit*self.position_manager.position:.2%})低于止盈阈值({self.config['TAKE_PROFIT_THRESHOLD']:.2%})"
            
        # 获利了结或恐慌指数过高（两个条件之一满足即可），应用波动率系数
        normal_sell_condition_1 = (
            profit*self.position_manager.position >= self.config['SELL_PROFIT_TARGET'] and
            panic_score >= self.config['SELL_SCORE_THRESHOLD'] and 
            profit > 0
        )
        
        # 当波动率中位数不为0时才应用波动率系数
        if self.config['VOLATILITY_MEDIAN'] != 0:
            normal_sell_condition_2 = (
                panic_score >= self.config['SELL_SCORE_THRESHOLD'] and
                last_buy_profit >= self.config['LAST_BUY_PROFIT_TARGET'] * vol_coef
            )
        else:
            normal_sell_condition_2 = (
                panic_score >= self.config['SELL_SCORE_THRESHOLD'] and
                last_buy_profit >= self.config['LAST_BUY_PROFIT_TARGET']
            )
        
        if normal_sell_condition_1 or normal_sell_condition_2:
            if self.config['SELL_AMOUNT'] >= self.config['MIN_TRADE_POSITION']:
                if normal_sell_condition_1:
                    self.last_sell_conditions.extend([
                        {
                            'type': 'profit_target',
                            'value': profit*self.position_manager.position,
                            'threshold': self.config['SELL_PROFIT_TARGET']
                        },
                        {
                            'type': 'panic_score',
                            'value': panic_score,
                            'threshold': self.config['SELL_SCORE_THRESHOLD']
                        }
                    ])
                if normal_sell_condition_2:
                    self.last_sell_conditions.extend([
                        {
                            'type': 'last_buy_profit',
                            'value': last_buy_profit,
                            'threshold': self.config['LAST_BUY_PROFIT_TARGET'] * vol_coef
                        },
                        {
                            'type': 'panic_score',
                            'value': panic_score,
                            'threshold': self.config['SELL_SCORE_THRESHOLD']
                        }
                    ])
                return 'normal_sell'
            else:
                self.no_sell_reason = f"卖出条件满足，但卖出仓位({self.config['SELL_AMOUNT']:.2%})小于最小交易仓位({self.config['MIN_TRADE_POSITION']:.2%})"
        elif not normal_sell_condition_1 and not normal_sell_condition_2:
            if panic_score < self.config['SELL_SCORE_THRESHOLD']:
                self.no_sell_reason = f"恐慌指数({panic_score:.2f})低于卖出阈值({self.config['SELL_SCORE_THRESHOLD']:.2f})"
            elif profit*self.position_manager.position < self.config['SELL_PROFIT_TARGET']:
                self.no_sell_reason = f"总体收益({profit*self.position_manager.position:.2%})低于获利目标({self.config['SELL_PROFIT_TARGET']:.2%})"
            elif last_buy_profit < self.config['LAST_BUY_PROFIT_TARGET'] * vol_coef:
                self.no_sell_reason = f"最近买入收益({last_buy_profit:.2%})低于目标({self.config['LAST_BUY_PROFIT_TARGET'] * vol_coef:.2%})"
        
        return False
        
    def get_last_sell_conditions(self):
        """获取最近一次卖出的触发条件"""
        return self.last_sell_conditions