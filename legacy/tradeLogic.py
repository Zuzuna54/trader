

# Function to execute a trade
def execute_trade(mock_balance, current_price, predicted_price, trade_fraction, active_trades, total_trades_made, rising_trend, dropping_tred, predicted_price_array, retrains_made, max_active_trades, stop_loss_percentage):
    
    total_btc_in_trades = 0
    total_btc_in_trades_value = 0
    predicted_price_avg = sum(predicted_price_array)/len(predicted_price_array)
    

    # # Buy if if rising trend
    # if rising_trend and mock_balance["USDT"] >= 10:
    #     # Buy via rising trend
    #     trade_amount = mock_balance["USDT"] * trade_fraction if mock_balance["USDT"] * trade_fraction > 10 else 10
    #     mock_balance["USDT"] -= trade_amount # Update the mock balance
    #     mock_balance["BTC"] += trade_amount / current_price # Update the mock balance
    #     print(f"Buying BTC: {trade_amount / current_price} at {current_price} via rising trend")
    #     # Add the trade to active trades with a stop loss price
    #     active_trades.append({"entry_price": current_price, "btc_amount": trade_amount / current_price})

    # Buy if predicted price is higher than current price
    if predicted_price_avg - 10 > current_price and mock_balance["USDT"] >= 10:
        # Buy via predicted price
        trade_amount = mock_balance["USDT"] * trade_fraction if mock_balance["USDT"] * trade_fraction > 10 else 10
        mock_balance["USDT"] -= trade_amount # Update the mock balance
        mock_balance["BTC"] += trade_amount / current_price # Update the mock balance
        print(f"Buying BTC: {trade_amount / current_price} at {current_price} via predicted price")
        # Add the trade to active trades with a stop loss price
        active_trades.append({"entry_price": current_price, "btc_amount": trade_amount / current_price})

    # Update active trades and sell if necessary
    for trade in active_trades:

        # Sell active trades based on stop-loss and predicted price and if the price is rising
        if mock_balance["BTC"] * current_price > 10 and current_price > (trade["entry_price"] * 0.005) + trade["entry_price"]:
            # Sell at 1% profit
            total_trades_made += 1
            trade_amount = trade["btc_amount"]
            mock_balance["BTC"] -= trade_amount
            mock_balance["USDT"] += trade_amount * current_price
            print(f"Selling BTC: {trade_amount} at {current_price} with entry price of {trade['entry_price']}")
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
        
        total_btc_in_trades += trade["btc_amount"]
        total_btc_in_trades_value += trade["btc_amount"] * current_price

    # Print the mock balance and trade history
    print(f"Mock balance: {mock_balance}")
    print(f"Total BTC value in Balance in USDT: {mock_balance['BTC'] * current_price}")
    print(f"Total number of open trades: {len(active_trades)}")
    print(f"Total number of trades made: {total_trades_made}")
    print(f"Hihest Entry Price: {max([trade['entry_price'] for trade in active_trades])}")
    print(f"Lowest Entry Price: {min([trade['entry_price'] for trade in active_trades])}")
    print(f"Total BTC in open trades value: {total_btc_in_trades_value}")