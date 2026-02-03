import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    API_KEY = os.getenv('BINANCE_API_KEY', 'your_api_key')
    API_SECRET = os.getenv('BINANCE_API_SECRET', 'your_api_secret')
    SYMBOL = os.getenv('TRADING_SYMBOL', 'BTC/USDT')
    INTERVAL = os.getenv('TRADING_INTERVAL', '5m')
    LOOKBACK = int(os.getenv('LOOKBACK', '30'))
    TRADE_FRACTION = float(os.getenv('TRADE_FRACTION', '0.05'))
    MAX_ACTIVE_TRADES = int(os.getenv('MAX_ACTIVE_TRADES', '5'))
    STOP_LOSS_PERCENTAGE = float(os.getenv('STOP_LOSS_PERCENTAGE', '0.01')) # 1%
    TAKE_PROFIT_PERCENTAGE = float(os.getenv('TAKE_PROFIT_PERCENTAGE', '0.01')) # 1%
    MOCK_MODE = os.getenv('MOCK_MODE', 'True').lower() == 'true'
