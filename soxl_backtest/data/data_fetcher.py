from abc import ABC, abstractmethod
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json
try:
    from longport.openapi import QuoteContext, Config, Period, AdjustType
except Exception:
    QuoteContext = Config = Period = AdjustType = None
try:
    from market_data_manager import MarketDataManager
except Exception:
    class MarketDataManager:
        def get_market_data(self, *args, **kwargs):
            raise NotImplementedError()
        def get_vix_data(self, *args, **kwargs):
            raise NotImplementedError()

class DataFetcher(ABC):
    """数据获取基类"""
    
    @abstractmethod
    def fetch_data(self, config):
        """获取数据的抽象方法"""
        pass

class LongbridgeDataFetcher(DataFetcher):
    """使用Longbridge API获取数据"""
    
    def fetch_data(self, config):
        """使用长桥API获取股票数据"""
        try:
            # 计算开始日期（额外获取30天数据用于计算指标）
            start_date = pd.to_datetime(config['START_DATE'])
            actual_start = start_date - timedelta(days=30)
            end_date = pd.to_datetime(config['END_DATE']) if config['END_DATE'] else datetime.now()

            print(f"开始获取数据(Longbridge)...")
            print(f"股票代码: {config['STOCK_CODE']}")
            print(f"数据时间范围: {actual_start.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")

            # 使用MarketDataManager获取股票数据
            market_data_manager = MarketDataManager()
            stock_df = market_data_manager.get_market_data(
                symbol=config['STOCK_CODE'],
                start_date=actual_start.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            # 创建股票DataFrame
            df = pd.DataFrame({
                'date': pd.to_datetime(stock_df['date']),
                'price': stock_df['close']
            })
            
            # 从数据库获取VIX数据
            # market_data_manager = MarketDataManager()
            vix_df = market_data_manager.get_vix_data(
                start_date=actual_start.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            # 确保两个DataFrame的date列都是datetime格式
            df['date'] = pd.to_datetime(df['date'])
            vix_df['date'] = pd.to_datetime(vix_df['date'])
            
            # 合并数据
            df = pd.merge(df, vix_df[['date', 'vix']], on='date', how='left')
            
            # 将date列转换为datetime
            df['date'] = pd.to_datetime(df['date'])

            print(f"\n过滤后剩余 {len(df)} 条数据")
            print("数据获取完成")
            
            # 检查是否有缺失值
            if df['vix'].isna().any():
                print("\n警告：VIX数据中存在缺失值")
                print(f"缺失值数量：{df['vix'].isna().sum()}")
                # 使用前向填充处理缺失值
                df['vix'].fillna(method='ffill', inplace=True)
                # 如果开头有缺失值，使用后向填充
                df['vix'].fillna(method='bfill', inplace=True)
            
            return df
            
        except Exception as e:
            import traceback
            print(f"使用长桥API获取数据失败: {str(e)}")
            print("错误调用栈:")
            print(traceback.format_exc())
            print("切换到使用Yahoo Finance获取数据...")
            return YahooDataFetcher().fetch_data(config)

class YahooDataFetcher(DataFetcher):
    """使用Yahoo Finance获取数据"""
    
    def fetch_data(self, config):
        """从本地CSV读取股票和VIX数据（兼容离线回测）"""
        from pathlib import Path
        start_date = pd.to_datetime(config['START_DATE'])
        actual_start = start_date - timedelta(days=30)
        end_date = pd.to_datetime(config['END_DATE']) if config['END_DATE'] else datetime.now()

        root = Path(__file__).resolve().parents[2]
        stock_path = root / "data" / "symbols" / f"{config['STOCK_CODE']}.csv"
        vix_path = root / "data" / "soxl_soxx_vix_full.csv"

        stock_df = pd.read_csv(stock_path, parse_dates=['date'])
        stock_df = stock_df.rename(columns={'close': 'price'})
        stock_df = stock_df[(stock_df['date'] >= actual_start) & (stock_df['date'] <= end_date)].copy()

        if vix_path.exists():
            vix_df = pd.read_csv(vix_path, parse_dates=['date'])[['date', 'vix']]
            df = pd.merge(stock_df[['date', 'price']], vix_df, on='date', how='left')
        else:
            df = stock_df[['date', 'price']].copy()
            df['vix'] = 20.0

        df['vix'] = df['vix'].ffill().bfill()
        print(f"[local] 读取 {len(df)} 条数据")
        return df