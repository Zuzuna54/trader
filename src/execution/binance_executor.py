import logging
from src.config import Config

class BinanceExecutor:
    def __init__(self, mock=True):
        self.mock = mock
        self.balance_usdt = 1000.0
        self.balance_btc = 0.0
        self.active_trades = []

    def execute_signal(self, signal, current_price):
        if signal == "BUY":
            self.buy(current_price)
        elif signal == "SELL":
            self.sell(current_price)

    def buy(self, price):
        if self.balance_usdt >= 10:
            amount_to_spend = self.balance_usdt * Config.TRADE_FRACTION
            if amount_to_spend < 10: amount_to_spend = 10

            btc_bought = amount_to_spend / price
            self.balance_usdt -= amount_to_spend
            self.balance_btc += btc_bought

            self.active_trades.append({
                'entry_price': price,
                'amount': btc_bought,
                'stop_loss': price * (1 - Config.STOP_LOSS_PERCENTAGE),
                'take_profit': price * (1 + Config.TAKE_PROFIT_PERCENTAGE)
            })
            logging.info(f"MOCK BUY: {btc_bought:.6f} BTC at {price}. Balance: {self.balance_usdt:.2f} USDT")

    def sell(self, price, trade=None):
        if trade:
            amount_to_sell = trade['amount']
            self.balance_btc -= amount_to_sell
            self.balance_usdt += amount_to_sell * price
            logging.info(f"MOCK SELL (TP/SL): {amount_to_sell:.6f} BTC at {price}. Balance: {self.balance_usdt:.2f} USDT")
        elif self.balance_btc > 0:
            # Sell everything
            amount_to_sell = self.balance_btc
            self.balance_btc = 0
            self.balance_usdt += amount_to_sell * price
            logging.info(f"MOCK SELL (Signal): {amount_to_sell:.6f} BTC at {price}. Balance: {self.balance_usdt:.2f} USDT")

    def check_risk_management(self, current_price):
        for trade in self.active_trades[:]:
            if current_price <= trade['stop_loss']:
                logging.info("Stop Loss Triggered")
                self.sell(current_price, trade)
                self.active_trades.remove(trade)
            elif current_price >= trade['take_profit']:
                logging.info("Take Profit Triggered")
                self.sell(current_price, trade)
                self.active_trades.remove(trade)
