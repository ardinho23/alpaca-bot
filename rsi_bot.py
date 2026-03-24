"""
=============================================================
  Alpaca RSI Bot v6 — Full Strategy
=============================================================

ARCHITECTURE
------------
Single-file runnable script with modular functions:

  fetch_data()         — pulls OHLCV bars from Alpaca
  compute_indicators() — Wilder RSI, 100MA + slope, ATR, SPY 200 daily MA
  check_regime()       — SPY 200-day MA regime filter with exposure scaling
  size_position()      — RSI-tiered + ATR-adjusted buy sizing vs real NAV
  risk_checks()        — caps, cash reserve, cooldowns, drawdown brake, daily limits
  check_stops()        — ATR trailing stop loss per position
  execute_trade()      — places market order via Alpaca API
  update_state()       — persists cooldowns, counters, equity peak, stops to JSON

CHANGES IN v6
-------------
1. Wilder RSI     — proper EMA-style smoothed RSI, not simple mean
2. MA slope filter — blocks buys when 100MA is pointing downward
3. ATR trailing stop — exits position if price drops X * ATR from entry
4. Dynamic NAV sizing — all sizing based on real account equity, not fixed constant
5. Regime exposure scaling — bear regime halves all base_pct, not just disables buys

STATE PERSISTENCE
-----------------
State saved to state.json. Railway restarts reload cooldowns,
daily counters, equity peak, regime state and stop levels.

ENVIRONMENT VARIABLES
---------------------
API            Alpaca key ID
SECRET         Alpaca secret key
PAPER          true / false (default true)
TELEGRAM       Telegram bot token (optional)
CHAT           Telegram chat ID (optional)
DRYRUN         true / false (default false)
"""

import os
import json
import time
import logging
import requests
import numpy as np
import pytz
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

# ─── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("rsi_bot")

# ─── Environment ────────────────────────────────────────────
API_KEY        = os.environ.get("API")
SECRET_KEY     = os.environ.get("SECRET")
PAPER          = os.environ.get("PAPER",  "true").lower() == "true"
DRYRUN         = os.environ.get("DRYRUN", "false").lower() == "true"
TELEGRAM_TOKEN = os.environ.get("TELEGRAM")
TELEGRAM_CHAT  = os.environ.get("CHAT")

# ─── Clients ────────────────────────────────────────────────
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
stock_data     = StockHistoricalDataClient(API_KEY, SECRET_KEY)
crypto_data    = CryptoHistoricalDataClient()

# ─── Universe & Weights ─────────────────────────────────────
# base_pct = target weight in bull regime
# bear regime halves all base_pct automatically
# max_pct  = absolute ceiling regardless of regime
ASSETS = [
    {"symbol": "SPY",     "base_pct": 0.12, "max_pct": 0.18, "type": "stock"},
    {"symbol": "QQQ",     "base_pct": 0.12, "max_pct": 0.18, "type": "stock"},
    {"symbol": "GLD",     "base_pct": 0.08, "max_pct": 0.14, "type": "stock"},
    {"symbol": "SLV",     "base_pct": 0.06, "max_pct": 0.12, "type": "stock"},
    {"symbol": "COPX",    "base_pct": 0.06, "max_pct": 0.12, "type": "stock"},
    {"symbol": "XLE",     "base_pct": 0.06, "max_pct": 0.12, "type": "stock"},
    {"symbol": "IWM",     "base_pct": 0.08, "max_pct": 0.14, "type": "stock"},
    {"symbol": "NVDA",    "base_pct": 0.08, "max_pct": 0.14, "type": "stock"},
    {"symbol": "MSFT",    "base_pct": 0.08, "max_pct": 0.14, "type": "stock"},
    {"symbol": "BTC/USD", "base_pct": 0.10, "max_pct": 0.16, "type": "crypto"},
    {"symbol": "ETH/USD", "base_pct": 0.08, "max_pct": 0.14, "type": "crypto"},
    {"symbol": "TLT",     "base_pct": 0.08, "max_pct": 0.14, "type": "stock"},
]

# ─── Strategy Parameters ────────────────────────────────────
RSI_PERIOD        = 14
MA_PERIOD         = 100
ATR_PERIOD        = 14
REGIME_MA_PERIOD  = 200
RSI_BUY_STRONG    = 25    # 1.5x unit
RSI_BUY_MID       = 30    # 1.0x unit
RSI_BUY_WEAK      = 35    # 0.5x unit
RSI_SELL_MILD     = 65    # sell 25%
RSI_SELL_STRONG   = 72    # sell 50%
RSI_SELL_BEARISH  = 60    # sell 50% when regime bearish
MIN_CASH_RESERVE  = 0.20  # keep 20% of NAV in cash
COOLDOWN_BUY_H    = 6
COOLDOWN_SELL_H   = 1
MAX_BUYS_PER_ASSET= 3
MAX_TRADES_DAY    = 10
DRAWDOWN_BRAKE_1  = 0.05  # halve buy sizes
DRAWDOWN_BRAKE_2  = 0.08  # stop all buys
DRAWDOWN_RESUME   = 0.03
MIN_NOTIONAL      = 1.0
ATR_STOP_MULT     = 2.5   # stop = entry - 2.5 * ATR at time of buy
MA_SLOPE_LOOKBACK = 5     # bars to measure MA slope direction

# ─── State File ─────────────────────────────────────────────
STATE_FILE = "state.json"

def load_state():
    default = {
        "cooldowns":      {},
        "daily_trades":   {},
        "daily_date":     "",
        "equity_peak":    0.0,
        "regime_bearish": False,
        "stop_levels":    {},  # symbol -> stop price
        "entry_prices":   {},  # symbol -> avg entry price
    }
    try:
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
            for k, v in default.items():
                if k not in data:
                    data[k] = v
            return data
    except:
        return default

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

# ─── Telegram ───────────────────────────────────────────────
def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT, "text": message}, timeout=10)
    except Exception as e:
        log.warning(f"Telegram error: {e}")

# ─── Account helpers ────────────────────────────────────────
def get_account_info():
    account = trading_client.get_account()
    return float(account.cash), float(account.equity)

def get_position_value(symbol):
    try:
        pos = trading_client.get_open_position(symbol.replace("/", ""))
        return float(pos.market_value)
    except:
        return 0.0

def get_all_positions():
    try:
        return {p.symbol: float(p.market_value) for p in trading_client.get_all_positions()}
    except:
        return {}

# ─── fetch_data() ────────────────────────────────────────────
def fetch_data(symbol, asset_type, timeframe=TimeFrame.Hour, days=30):
    end   = datetime.now(pytz.utc)
    start = end - timedelta(days=days)
    try:
        if asset_type == "crypto":
            bars = crypto_data.get_crypto_bars(
                CryptoBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=timeframe,
                    start=start,
                    end=end
                )
            ).df
        else:
            bars = stock_data.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=timeframe,
                    start=start,
                    end=end,
                    feed="iex"
                )
            ).df
        if bars.empty:
            return None
        if hasattr(bars.index, "levels"):
            bars = bars.reset_index(level=0, drop=True)
        return bars
    except Exception as e:
        log.warning(f"fetch_data error for {symbol}: {e}")
        return None

# ─── compute_indicators() ────────────────────────────────────

def compute_wilder_rsi(closes, period=14):
    """
    Proper Wilder smoothed RSI using EMA-style updates.
    Requires at least 2*period bars for reliable seeding.
    """
    if len(closes) < period * 2:
        return None

    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Seed with simple average over first period
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Wilder smoothing: EMA with alpha = 1/period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

def compute_ma(closes, period):
    if len(closes) < period:
        return None
    return float(np.mean(closes[-period:]))

def compute_ma_slope(closes, period, lookback=5):
    """
    Returns True if MA is sloping upward over last `lookback` bars.
    Blocks buys on downward-sloping MAs (falling knife filter).
    """
    if len(closes) < period + lookback:
        return None
    ma_now  = np.mean(closes[-(period):])
    ma_prev = np.mean(closes[-(period + lookback):-lookback])
    return ma_now > ma_prev

def compute_atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i]  - closes[i-1])
        )
        trs.append(tr)
    return float(np.mean(trs[-period:]))

def compute_indicators(symbol, asset_type):
    bars = fetch_data(symbol, asset_type, timeframe=TimeFrame.Hour, days=60)
    if bars is None or len(bars) < MA_PERIOD + RSI_PERIOD * 2:
        log.warning(f"{symbol}: not enough data for indicators")
        return None

    closes  = bars["close"].values
    highs   = bars["high"].values
    lows    = bars["low"].values
    price   = closes[-1]

    rsi      = compute_wilder_rsi(closes, RSI_PERIOD)
    ma100    = compute_ma(closes, MA_PERIOD)
    ma_up    = compute_ma_slope(closes, MA_PERIOD, MA_SLOPE_LOOKBACK)
    atr      = compute_atr(highs, lows, closes, ATR_PERIOD)
    atr_pct  = (atr / price * 100) if atr and price > 0 else None

    return {
        "price":   round(price, 4),
        "rsi":     rsi,
        "ma100":   round(ma100, 4) if ma100 else None,
        "ma_up":   ma_up,
        "atr":     round(atr, 4)   if atr   else None,
        "atr_pct": round(atr_pct, 2) if atr_pct else None,
    }

# ─── check_regime() ──────────────────────────────────────────
def check_regime(state):
    """
    Checks SPY vs 200-day MA.
    Bull regime  → full base_pct allocations
    Bear regime  → all base_pct halved (exposure scaling, not just buy disabling)
    """
    bars = fetch_data("SPY", "stock", timeframe=TimeFrame.Day, days=220)
    if bars is None or len(bars) < REGIME_MA_PERIOD:
        log.warning("Regime: not enough SPY daily data — keeping previous state")
        return state

    closes   = bars["close"].values
    ma200    = compute_ma(closes, REGIME_MA_PERIOD)
    spy_last = closes[-1]
    bearish  = spy_last < ma200

    if bearish and not state["regime_bearish"]:
        msg = (f"⚠️ REGIME BEARISH | SPY ${spy_last:.2f} below 200d MA ${ma200:.2f} | "
               f"All allocations halved — de-risking mode active")
        log.warning(msg)
        send_telegram(msg)
    elif not bearish and state["regime_bearish"]:
        msg = (f"✅ REGIME BULLISH | SPY ${spy_last:.2f} back above 200d MA ${ma200:.2f} | "
               f"Full allocations restored")
        log.info(msg)
        send_telegram(msg)

    state["regime_bearish"] = bearish
    return state

def effective_base_pct(asset, regime_bearish):
    """
    In bear regime, halve the base allocation.
    This reduces exposure across the board, not just disables buys.
    """
    pct = asset["base_pct"]
    if regime_bearish:
        pct = pct * 0.5
    return pct

# ─── check_stops() ───────────────────────────────────────────
def check_stops(symbol, price, asset_type, state, cash):
    """
    ATR trailing stop loss.
    If price drops below stop level set at entry, sell full position.
    Stop is set at: entry_price - ATR_STOP_MULT * ATR at time of buy.
    """
    stop = state["stop_levels"].get(symbol)
    if not stop:
        return False, state, cash

    if price <= stop:
        position_value = get_position_value(symbol)
        if position_value <= 0:
            # Position already closed, clean up state
            state["stop_levels"].pop(symbol, None)
            state["entry_prices"].pop(symbol, None)
            return False, state, cash

        msg = (f"🛑 STOP LOSS {symbol} | Price ${price} hit stop ${stop:.4f} | "
               f"Selling full position ${position_value:.2f}")
        log.warning(msg)
        send_telegram(msg)

        result = execute_trade(symbol, position_value, OrderSide.SELL, asset_type, msg)
        if result:
            cash += position_value
            state["stop_levels"].pop(symbol, None)
            state["entry_prices"].pop(symbol, None)
            save_state(state)
        return True, state, cash

    return False, state, cash

# ─── size_position() ─────────────────────────────────────────
def size_position(rsi, atr_pct, base_alloc, reason_parts):
    """
    Buy unit = 10% of base allocation (based on real NAV).
    Tiered by RSI. Adjusted by ATR volatility.
    """
    unit = base_alloc * 0.10

    if rsi < RSI_BUY_STRONG:
        multiplier = 1.5
        reason_parts.append(f"RSI {rsi} < {RSI_BUY_STRONG} → 1.5x unit")
    elif rsi < RSI_BUY_MID:
        multiplier = 1.0
        reason_parts.append(f"RSI {rsi} < {RSI_BUY_MID} → 1.0x unit")
    else:
        multiplier = 0.5
        reason_parts.append(f"RSI {rsi} < {RSI_BUY_WEAK} → 0.5x unit")

    size = unit * multiplier

    if atr_pct is not None:
        if atr_pct > 5.0:
            size *= 0.25
            reason_parts.append(f"ATR% {atr_pct}% > 5% → size -75%")
        elif atr_pct > 3.0:
            size *= 0.50
            reason_parts.append(f"ATR% {atr_pct}% > 3% → size -50%")
        elif atr_pct < 1.5:
            size *= 1.20
            reason_parts.append(f"ATR% {atr_pct}% < 1.5% → size +20%")

    return round(size, 2)

# ─── risk_checks() ───────────────────────────────────────────
def risk_checks(symbol, buy_amount, asset, equity, cash, state, reason_parts):
    today = datetime.now().strftime("%Y-%m-%d")

    if state["daily_date"] != today:
        state["daily_trades"] = {}
        state["daily_date"]   = today
        log.info("Daily counters reset")

    total_trades = sum(state["daily_trades"].values())
    if total_trades >= MAX_TRADES_DAY:
        reason_parts.append(f"SKIP: max {MAX_TRADES_DAY} total trades today")
        return False, state

    asset_trades = state["daily_trades"].get(symbol, 0)
    if asset_trades >= MAX_BUYS_PER_ASSET:
        reason_parts.append(f"SKIP: max {MAX_BUYS_PER_ASSET} buys for {symbol} today")
        return False, state

    cooldown_key = f"buy_{symbol}"
    last_trade   = state["cooldowns"].get(cooldown_key)
    if last_trade:
        elapsed = (datetime.now() - datetime.fromisoformat(last_trade)).total_seconds() / 3600
        if elapsed < COOLDOWN_BUY_H:
            remaining = round(COOLDOWN_BUY_H - elapsed, 1)
            reason_parts.append(f"SKIP: {symbol} cooldown {remaining}h remaining")
            return False, state

    # Max position cap vs real equity
    max_value   = equity * asset["max_pct"]
    current_pos = get_position_value(symbol)
    if current_pos + buy_amount > max_value:
        reason_parts.append(
            f"SKIP: {symbol} would exceed cap ${max_value:.2f} "
            f"(current ${current_pos:.2f})"
        )
        return False, state

    # Minimum cash reserve vs real equity
    min_cash = equity * MIN_CASH_RESERVE
    if cash - buy_amount < min_cash:
        reason_parts.append(
            f"SKIP: would breach min cash reserve "
            f"${min_cash:.2f} (cash ${cash:.2f})"
        )
        return False, state

    # Drawdown brake vs equity peak
    peak     = state.get("equity_peak", equity)
    drawdown = (peak - equity) / peak if peak > 0 else 0

    if drawdown > DRAWDOWN_BRAKE_2:
        reason_parts.append(
            f"SKIP: drawdown {drawdown*100:.1f}% > "
            f"{DRAWDOWN_BRAKE_2*100}% — all buys stopped"
        )
        return False, state
    elif drawdown > DRAWDOWN_BRAKE_1:
        reason_parts.append(
            f"WARNING: drawdown {drawdown*100:.1f}% — buy size halved"
        )
        return "halve", state

    return True, state

# ─── execute_trade() ─────────────────────────────────────────
def execute_trade(symbol, amount, side, asset_type, reason):
    if amount < MIN_NOTIONAL:
        log.info(f"SKIP {symbol}: ${amount:.2f} below min notional ${MIN_NOTIONAL}")
        return None

    action = "BUY" if side == OrderSide.BUY else "SELL"
    emoji  = "🟢" if side == OrderSide.BUY else "🔴"

    if DRYRUN:
        msg = f"[DRY RUN] {action} {symbol} | ${amount:.2f} | {reason}"
        log.info(msg)
        send_telegram(f"{emoji} DRY RUN | {msg}")
        return "dry_run"

    try:
        tif    = TimeInForce.GTC if asset_type == "crypto" else TimeInForce.DAY
        result = trading_client.submit_order(MarketOrderRequest(
            symbol=symbol,
            notional=round(amount, 2),
            side=side,
            time_in_force=tif
        ))
        msg = f"{action} {symbol} | ${amount:.2f} | {reason} | status: {result.status}"
        log.info(f"✅ {msg}")
        send_telegram(f"{emoji} {msg}")
        return result
    except Exception as e:
        err = f"Order failed for {symbol}: {e}"
        log.error(f"❌ {err}")
        send_telegram(f"🔴 ERROR | {err}")
        return None

# ─── update_state() ──────────────────────────────────────────
def update_state(state, symbol, side, equity, price=None, atr=None):
    today = datetime.now().strftime("%Y-%m-%d")
    now   = datetime.now().isoformat()

    if equity > state.get("equity_peak", 0):
        state["equity_peak"] = equity

    if state["daily_date"] != today:
        state["daily_trades"] = {}
        state["daily_date"]   = today

    if side == OrderSide.BUY:
        state["daily_trades"][symbol] = state["daily_trades"].get(symbol, 0) + 1
        # Set ATR trailing stop
        if price is not None and atr is not None:
            stop = round(price - ATR_STOP_MULT * atr, 4)
            state["stop_levels"][symbol]  = stop
            state["entry_prices"][symbol] = price
            log.info(f"Stop set for {symbol}: ${stop:.4f} "
                     f"(entry ${price:.4f} - {ATR_STOP_MULT}x ATR ${atr:.4f})")
    else:
        # On sell, remove stop
        state["stop_levels"].pop(symbol, None)
        state["entry_prices"].pop(symbol, None)

    cooldown_key = f"{'buy' if side == OrderSide.BUY else 'sell'}_{symbol}"
    state["cooldowns"][cooldown_key] = now

    save_state(state)
    return state

# ─── Main run loop ───────────────────────────────────────────
def run():
    state       = load_state()
    cash, equity = get_account_info()

    # Update equity peak
    if equity > state.get("equity_peak", 0):
        state["equity_peak"] = equity
        save_state(state)

    peak     = state["equity_peak"]
    drawdown = (peak - equity) / peak if peak > 0 else 0

    log.info(f"{'='*55}")
    log.info(f"RSI Bot v6 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info(
        f"Equity: ${equity:.2f} | Cash: ${cash:.2f} | "
        f"Peak: ${peak:.2f} | Drawdown: {drawdown*100:.1f}%"
    )
    log.info(f"Regime: {'BEARISH 🔴 (allocations halved)' if state['regime_bearish'] else 'BULLISH 🟢 (full allocations)'}")
    log.info(f"{'='*55}")

    state = check_regime(state)

    for asset in ASSETS:
        symbol     = asset["symbol"]
        asset_type = asset["type"]

        # Effective allocation — halved in bear regime
        eff_pct    = effective_base_pct(asset, state["regime_bearish"])
        base_alloc = equity * eff_pct

        log.info(f"\n--- {symbol} ---")
        ind = compute_indicators(symbol, asset_type)

        if ind is None:
            log.warning(f"{symbol}: skipping — insufficient data")
            continue

        price   = ind["price"]
        rsi     = ind["rsi"]
        ma100   = ind["ma100"]
        ma_up   = ind["ma_up"]
        atr     = ind["atr"]
        atr_pct = ind["atr_pct"]

        ma_slope_str = "rising" if ma_up else "falling" if ma_up is not None else "unknown"
        log.info(
            f"{symbol} | Price: ${price} | RSI: {rsi} | "
            f"100MA: ${ma100} ({ma_slope_str}) | ATR%: {atr_pct}%"
        )

        # ── STOP LOSS CHECK ─────────────────────────────────
        stopped, state, cash = check_stops(symbol, price, asset_type, state, cash)
        if stopped:
            continue

        position_value = get_position_value(symbol)

        # ── SELL LOGIC ──────────────────────────────────────
        sell_pct    = 0
        sell_reason = []

        if state["regime_bearish"] and rsi is not None and rsi > RSI_SELL_BEARISH:
            sell_pct = 0.50
            sell_reason.append(f"bearish regime + RSI {rsi} > {RSI_SELL_BEARISH}")
        elif rsi is not None and rsi > RSI_SELL_STRONG:
            sell_pct = 0.50
            sell_reason.append(f"RSI {rsi} > {RSI_SELL_STRONG} (strongly overbought)")
        elif rsi is not None and rsi > RSI_SELL_MILD:
            sell_pct = 0.25
            sell_reason.append(f"RSI {rsi} > {RSI_SELL_MILD} (overbought)")
        elif ma100 is not None and price < ma100:
            sell_pct = 0.25
            sell_reason.append(f"price ${price} below 100MA ${ma100}")

        if sell_pct > 0 and position_value > 0:
            sell_amount = round(position_value * sell_pct, 2)
            reason_str  = " | ".join(sell_reason)
            full_reason = (
                f"{reason_str} | Position: ${position_value:.2f} | "
                f"Selling {sell_pct*100:.0f}% = ${sell_amount:.2f} | "
                f"Cash after: ~${cash+sell_amount:.2f}"
            )

            sell_key  = f"sell_{symbol}"
            last_sell = state["cooldowns"].get(sell_key)
            if last_sell:
                elapsed = (datetime.now() - datetime.fromisoformat(last_sell)).total_seconds() / 3600
                if elapsed < COOLDOWN_SELL_H:
                    log.info(
                        f"SKIP sell {symbol}: cooldown "
                        f"{round(COOLDOWN_SELL_H-elapsed,1)}h remaining"
                    )
                    continue

            result = execute_trade(symbol, sell_amount, OrderSide.SELL, asset_type, full_reason)
            if result:
                cash  += sell_amount
                state  = update_state(state, symbol, OrderSide.SELL, equity)
            continue

        # ── BUY LOGIC ───────────────────────────────────────
        if rsi is None or rsi >= RSI_BUY_WEAK:
            log.info(f"HOLD {symbol} | RSI {rsi} neutral — no action")
            continue

        # MA slope filter — block buys on falling MA (falling knife)
        if ma_up is False:
            log.info(
                f"SKIP buy {symbol} | 100MA is falling — "
                f"no buy into downtrend (RSI {rsi})"
            )
            continue

        if ma100 is not None and price < ma100:
            log.info(
                f"SKIP buy {symbol} | Price ${price} below 100MA ${ma100}"
            )
            continue

        reason_parts = [f"RSI {rsi} (oversold)"]
        size = size_position(rsi, atr_pct, base_alloc, reason_parts)

        check, state = risk_checks(
            symbol, size, asset, equity, cash, state, reason_parts
        )

        if check == "halve":
            size = round(size * 0.50, 2)
            reason_parts.append("size halved — drawdown brake")
        elif not check:
            log.info(f"SKIP buy {symbol} | {' | '.join(reason_parts)}")
            continue

        full_reason = (
            f"Buying {symbol}: {' | '.join(reason_parts)} | "
            f"Alloc: {eff_pct*100:.0f}% of NAV ${equity:.2f} | "
            f"Order: ${size:.2f} | Cash before: ${cash:.2f} | "
            f"Cash after: ~${cash-size:.2f}"
        )

        result = execute_trade(symbol, size, OrderSide.BUY, asset_type, full_reason)
        if result:
            cash  -= size
            state  = update_state(
                state, symbol, OrderSide.BUY, equity,
                price=price, atr=atr
            )

    log.info(f"\nCycle complete. Next check in 30 minutes.")
    log.info(f"{'='*55}")
    save_state(state)

# ─── Entry point ─────────────────────────────────────────────
ET = pytz.timezone("US/Eastern")

log.info(f"Bot starting | PAPER={PAPER} | DRYRUN={DRYRUN}")
if DRYRUN:
    log.info("DRY RUN MODE — no real orders will be placed")

while True:
    now             = datetime.now(ET)
    is_weekday      = now.weekday() < 5
    is_market_hours = (now.hour == 9 and now.minute >= 30) or (10 <= now.hour < 16)

    if is_weekday and is_market_hours:
        run()
        log.info("Sleeping 30 minutes...")
        time.sleep(1800)
    elif is_weekday and now.hour < 9:
        log.info(f"Market not open yet — {now.strftime('%H:%M')} ET — sleeping 10 min")
        time.sleep(600)
    elif is_weekday and now.hour >= 16:
        log.info(f"Stock market closed — {now.strftime('%H:%M')} ET — crypto-only cycle")
        run()
        log.info("Sleeping 1 hour...")
        time.sleep(3600)
    else:
        log.info(f"Weekend — {now.strftime('%A %H:%M')} ET — crypto-only cycle")
        run()
        log.info("Sleeping 1 hour...")
        time.sleep(3600)
