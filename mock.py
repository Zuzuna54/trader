import pandas as pd
import numpy as np

# Function to calculate RSI
def rsi(prices, interval=14):
    deltas = np.diff(prices)
    seed = deltas[:interval+1]
    up = seed[seed >= 0].sum() / interval
    down = -seed[seed < 0].sum() / interval
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:interval] = 100. - 100. / (1. + rs)

    for i in range(interval, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (interval - 1) + upval) / interval
        down = (down * (interval - 1) + downval) / interval
        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)
    return rsi

# Function to calculate MACD
def macd(prices, short_interval=6, long_interval=13, signal_interval=4):
    prices_series = pd.Series(prices)
    short_ema = prices_series.ewm(span=short_interval).mean()
    long_ema = prices_series.ewm(span=long_interval).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_interval).mean()
    return macd_line, signal_line

# Function to calculate Bollinger Bands
def bollinger_bands(prices, interval=20, std_dev_factor=2):
    prices_series = pd.Series(prices)
    sma = prices_series.rolling(window=interval).mean()
    std_dev = prices_series.rolling(window=interval).std()
    upper_band = sma + (std_dev * std_dev_factor)
    lower_band = sma - (std_dev * std_dev_factor)
    return upper_band, lower_band

# Function to execute a trade
def execute_trade(mock_balance, current_price, predicted_price, trade_fraction, active_trades, total_trades_made, rising_trend, dropping_tred, predicted_price_array, retrains_made, max_active_trades, stop_loss_percentage, historical_prices):
    
    total_btc_in_trades = 0
    total_btc_in_trades_value = 0
    predicted_price_avg = sum(predicted_price_array)/len(predicted_price_array)
    
    # Calculate RSI
    rsi_values = rsi(np.array(historical_prices))
    current_rsi = rsi_values[-1]

    # Calculate MACD
    historical_prices_series = pd.Series(historical_prices)
    macd_line, signal_line = macd(historical_prices_series)
    macd_cross_above = macd_line[-2] < signal_line[-2] and macd_line[-1] > signal_line[-1]
    macd_cross_below = macd_line[-2] > signal_line[-2] and macd_line[-1] < signal_line[-1]

    # Calculate Bollinger Bands
    upper_band, lower_band = bollinger_bands(historical_prices_series)
    price_cross_above_lower_band = historical_prices_series[-2] < lower_band[-2] and historical_prices_series[-1] > lower_band[-1]
    price_cross_below_upper_band = historical_prices_series[-2] > upper_band[-2] and historical_prices_series[-1] < upper_band[-1]


    # Buy only if model has been retrained more than 1 times
    if retrains_made > 1:
        # Buy if rising trend, RSI is below 30 (oversold), and MACD crosses above the signal line
        if rising_trend and mock_balance["USDT"] >= 10 and current_rsi < 30 and macd_cross_above:
            # Buy via rising trend
            trade_amount = mock_balance["USDT"] * trade_fraction if mock_balance["USDT"] * trade_fraction > 10 else 10
            mock_balance["USDT"] -= trade_amount # Update the mock balance
            mock_balance["BTC"] += trade_amount / current_price # Update the mock balance
            print(f"Buying BTC: {trade_amount / current_price} at {current_price} via rising trend and RSI")
            # Add the trade to active trades with a stop loss price
            active_trades.append({"entry_price": current_price, "btc_amount": trade_amount / current_price})

         # Buy if predicted price is higher than current price and price crosses above the lower Bollinger Band
        if predicted_price_avg > current_price and mock_balance["USDT"] >= 10 and price_cross_above_lower_band:
            # Buy via predicted price
            trade_amount = mock_balance["USDT"] * trade_fraction if mock_balance["USDT"] * trade_fraction > 10 else 10
            mock_balance["USDT"] -= trade_amount # Update the mock balance
            mock_balance["BTC"] += trade_amount / current_price # Update the mock balance
            print(f"Buying BTC: {trade_amount / current_price} at {current_price} via predicted price and Bollinger Bands crossing below price") 
            # Add the trade to active trades with a stop loss price
            active_trades.append({"entry_price": current_price, "btc_amount": trade_amount / current_price})

        # Update active trades and sell if necessary
        for trade in active_trades:

            total_btc_in_trades += trade["btc_amount"]
            total_btc_in_trades_value += trade["btc_amount"] * current_price

            # Sell at 1% profit, if RSI is above 70 (overbought), or if MACD crosses below the signal line
            if (mock_balance["BTC"] * current_price > 10 and current_price > (trade["entry_price"] * 0.01) + trade["entry_price"]) or current_rsi > 70 or macd_cross_below:
                total_trades_made += 1
                trade_amount = trade["btc_amount"]
                mock_balance["BTC"] -= trade_amount
                mock_balance["USDT"] += trade_amount * current_price
                print(f"Selling BTC: {trade_amount} at {current_price} with entry price of {trade['entry_price']} due to profit or RSI or MACD")
                # Remove the trade from active trades
                active_trades.remove(trade)

            # Sell active trades based on stop-loss and predicted price and if the price is dropping
            if current_price <= trade["entry_price"] * stop_loss_percentage:
                sell_amount = trade["btc_amount"]
                mock_balance["BTC"] -= sell_amount
                mock_balance["USDT"] += sell_amount * current_price
                print(f"Sold {sell_amount} BTC at ${current_price} due to stop-loss")
                active_trades.remove(trade)
                total_trades_made += 1
            elif predicted_price_avg - 50 < current_price:
                sell_amount = trade["btc_amount"]
                mock_balance["BTC"] -= sell_amount
                mock_balance["USDT"] += sell_amount * current_price
                print(f"Sold {sell_amount} BTC at ${current_price} due to predicted price")
                active_trades.remove(trade)
                total_trades_made += 1
            elif dropping_tred:
                sell_amount = trade["btc_amount"]
                mock_balance["BTC"] -= sell_amount
                mock_balance["USDT"] += sell_amount * current_price
                print(f"Sold {sell_amount} BTC at ${current_price} due to dropping trend")
                active_trades.remove(trade)
                total_trades_made += 1
            elif price_cross_below_upper_band:
                sell_amount = trade["btc_amount"]
                mock_balance["BTC"] -= sell_amount
                mock_balance["USDT"] += sell_amount * current_price
                print(f"Sold {sell_amount} BTC at ${current_price} due to crossing below the upper Bollinger Band")
                active_trades.remove(trade)
                total_trades_made += 1

    # Print the mock balance and trade history
    print(f"Mock balance: {mock_balance}")
    print(f"Total BTC value in Balance in USDT: {mock_balance['BTC'] * current_price}")
    print(f"Total number of open trades: {len(active_trades)}")
    print(f"Total number of trades made: {total_trades_made}")
    print(f"Hihest Entry Price: {max([trade['entry_price'] for trade in active_trades])}")
    print(f"Lowest Entry Price: {min([trade['entry_price'] for trade in active_trades])}")
    print(f"Total BTC in open trades value: {total_btc_in_trades_value}")