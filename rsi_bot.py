import os
import time
import numpy as np
import pytz
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta

API_KEY    = os.environ.get("API")
SECRET_KEY = os.environ.get("SECRET")
PAPER      = os.environ.get("PAPER", "true").lower() == "true"

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
stock_data     = StockHistoricalDataClient(API_KEY, SECRET_KEY)
crypto_data    = CryptoHistoricalDataClient()

ASSETS = [
    {"symbol": "SPY",     "budget": 40, "type": "stock"},
    {"symbol": "QQQ",     "budget": 25, "type": "stock"},
    {"symbol": "GLD",     "budget": 15, "type": "stock"},
    {"symbol": "NVDA",    "budget": 10, "type": "stock"},
    {"symbol": "BTC/USD", "budget": 10, "type": "crypto"},
]

RSI_BUY    = 40
RSI_SELL   = 65
RSI_PERIOD = 14

def calculate_rsi(prices, period=14):
    deltas   = np.diff(prices)
    gains    = np.where(deltas > 0, deltas, 0)
    losses   = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_prices(symbol, asset_type, days=30):
    end   = datetime.now()
    start = end - timedelta(days=days)
    try:
        if asset_type == "crypto":
            bars = crypto_data.get_crypto_bars(
                CryptoBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Hour,
                    start=start,
                    end=end
                )
            ).df
        else:
            bars = stock_data.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Hour,
                    start=start,
                    end=end,
                    feed="iex"
                )
            ).df
        return bars["close"].values if not bars.empty else None
    except Exception as e:
        print(f"  Data error for {symbol}: {e}")
        return None

def get_position_value(symbol):
    try:
        pos = trading_client.get_open_position(symbol.replace("/", ""))
        return float(pos.market_value)
    except:
        return 0.0

def place_order(symbol, amount, side, asset_type):
    tif = TimeInForce.GTC if asset_type == "crypto" else TimeInForce.DAY
    return trading_client.submit_order(MarketOrderRequest(
        symbol=symbol,
        notional=round(amount, 2),
        side=side,
        time_in_force=tif
    ))

def run():
    print(f"\n{'='*45}")
    print(f"RSI Bot -- {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*45}")

    for asset in ASSETS:
        symbol     = asset["symbol"]
        budget     = asset["budget"]
        asset_type = asset["type"]

        print(f"\n{symbol}:")
        prices = get_prices(symbol, asset_type)

        if prices is None or len(prices) < RSI_PERIOD + 1:
            print(f"  Not enough data -- skipping")
            continue

        rsi            = calculate_rsi(prices, RSI_PERIOD)
        position_value = get_position_value(symbol)

        print(f"  RSI: {rsi:.1f} | Position: ${position_value:.2f}")

        if rsi < RSI_BUY:
            multiplier = 1 + (RSI_BUY - rsi) / RSI_BUY
            buy_amount = min(budget * multiplier, budget * 2)
            try:
                result = place_order(symbol, buy_amount, OrderSide.BUY, asset_type)
                print(f"  BUY  ${buy_amount:.2f} -- RSI oversold at {rsi:.1f} ({result.status})")
            except Exception as e:
                print(f"  BUY failed: {e}")

        elif rsi > RSI_SELL and position_value > 0:
            sell_amount = position_value * 0.5
            if sell_amount >= 1:
                try:
                    result = place_order(symbol, sell_amount, OrderSide.SELL, asset_type)
                    print(f"  SELL ${sell_amount:.2f} -- RSI overbought at {rsi:.1f} ({result.status})")
                except Exception as e:
                    print(f"  SELL failed: {e}")
            else:
                print(f"  Position too small to sell -- holding")
        else:
            print(f"  HOLD -- RSI neutral ({rsi:.1f})")

    print(f"\nDone.")

while True:
    now = datetime.now(pytz.timezone("US/Eastern"))
    is_weekday     = now.weekday() < 5
    is_market_hours = (now.hour == 9 and now.minute >= 30) or (10 <= now.hour < 16)

    if is_weekday and is_market_hours:
        run()
        print("Sleeping 5 minutes...")
        time.sleep(300)
    elif is_weekday and now.hour < 9:
        print(f"Market not open yet -- {now.strftime('%H:%M')} ET")
        time.sleep(600)
    elif is_weekday and now.hour >= 16:
        print(f"Market closed for today -- {now.strftime('%H:%M')} ET")
        time.sleep(3600)
    else:
        print("Weekend -- sleeping 1 hour")
        time.sleep(3600)
