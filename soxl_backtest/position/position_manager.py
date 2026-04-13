class PositionManager:
    """仓位管理类"""
    
    def __init__(self, config):
        """初始化仓位管理器"""
        self.config = config
        self.position = 0
        self.shares = 0
        self.capital = config['INITIAL_CAPITAL']
        self.position_stack = []
        self.avg_price = 0
        self.initial_position_built = False
        
    def update_position_stack(self, price, shares, is_buy=True):
        """更新持仓栈"""
        if is_buy:
            self.position_stack.append({
                'price': price,
                'shares': shares
            })
        else:
            remaining_shares = shares
            while remaining_shares > 0 and self.position_stack:
                pos = self.position_stack[0]
                shares_sold = min(remaining_shares, pos['shares'])
                if shares_sold == pos['shares']:
                    self.position_stack.pop(0)
                else:
                    pos['shares'] -= shares_sold
                remaining_shares -= shares_sold
                
        if self.position_stack:
            total_cost = sum(pos['price'] * pos['shares'] for pos in self.position_stack)
            total_shares = sum(pos['shares'] for pos in self.position_stack)
            self.avg_price = total_cost / total_shares
            self.shares = total_shares
        else:
            self.avg_price = 0
            self.shares = 0
            
    def update_avg_price(self):
        """更新平均买入价格"""
        if not self.position_stack:
            self.avg_price = 0
            self.shares = 0
            return
            
        total_cost = sum(pos['price'] * pos['shares'] for pos in self.position_stack)
        total_shares = sum(pos['shares'] for pos in self.position_stack)
        self.avg_price = total_cost / total_shares if total_shares > 0 else 0
        self.shares = total_shares
    
    def add_position(self, price, shares):
        """添加新的持仓"""
        self.position_stack.append({
            'price': price,
            'shares': shares
        })
        self.update_avg_price()
    
    def reduce_position(self, shares_to_sell):
        """按FIFO原则减少持仓"""
        remaining_shares = shares_to_sell
        sold_value = 0
        
        while remaining_shares > 0 and self.position_stack:
            pos = self.position_stack[-1]  # 获取最后买入的持仓
            shares_sold = min(remaining_shares, pos['shares'])
            sold_value += shares_sold * pos['price']
            
            if shares_sold == pos['shares']:
                self.position_stack.pop()  # 如果全部卖出，移除这笔持仓
            else:
                pos['shares'] -= shares_sold  # 否则减少持仓数量
                
            remaining_shares -= shares_sold
            
        self.update_avg_price()
        return sold_value / shares_to_sell if shares_to_sell > 0 else 0  # 返回平均卖出价格
        
    def update_position(self, current_price):
        """更新当前仓位"""
        total_value = self.capital + self.shares * current_price
        self.position = (self.shares * current_price) / total_value if total_value > 0 else 0
        self.position = round(self.position, 4)
        return total_value
        
    def get_last_position(self):
        """获取最后一次买入的持仓信息"""
        return self.position_stack[-1] if self.position_stack else None 