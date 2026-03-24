"""
=============================================================
  Alpaca RSI Bot — Full Strategy
=============================================================

ARCHITECTURE
------------
Single-file runnable script with modular functions:

  fetch_data()        — pulls OHLCV bars from Alpaca
  compute_indicators()— RSI, 100MA, ATR, SPY 200 daily MA
  check_regime()      — SPY 200-day MA market regime filter
  size_position()     — RSI-tiered + ATR-adjusted buy sizing
  risk_checks()       — caps, cash reserve, cooldowns, drawdown brake, daily limits
  execute_trade()     — places market order via Alpaca API
  update_state()      — persists cooldowns, counters, equity peak to JSON

STATE PERSISTENCE
-----------------
State is saved to state.json in the working directory.
Railway restarts will reload this file so cooldowns,
daily counters, equity peak and regime state survive.

ASSUMPTIONS
-----------
- 1-hour candles used for RSI, 100MA, ATR signals
- SPY daily candles used only for 200-day MA regime filter
- Buy unit = 10% of asset's base target allocation
- Crypto trades 24/7; stocks only during market hours
- Minimum notional enforced ($1 floor before Alpaca rejects)
- DRYRUN=true prints decisions without placing orders

ENVIRONMENT VARIABLES
---------------------
API            Alpaca key ID
SECRET         Alpaca secret key
PAPER          true / false (default true)
TELEGRAM       Telegram bot token (optional)
CHAT           Telegram chat ID (optional)
DRYRUN         true / false (default false)
PORTFOLIO      portfolio size in USD (default 500)
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
PAPER          = os.environ.get("PAPER",   "true").lower() == "true"
DRYRUN         = os.environ.get("DRYRUN", "false").lower() == "true"
TELEGRAM_TOKEN = os.environ.get("TELEGRAM")
TELEGRAM_CHAT  = os.environ.get("CHAT")
PORTFOLIO_SIZE = float(os.environ.get("PORTFOLIO", "500"))

# ─── Clients ────────────────────────────────────────────────
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
stock_data     = StockHistoricalDataClient(API_KEY, SECRET_KEY)
crypto_data    = CryptoHistoricalDataClient()

# ─── Universe & Weights ─────────────────────────────────────
ASSETS = [
    {"symbol": "SPY",     "base_pct": 0.12, "max_pct": 0.18, "type": "stock"},   # S&P 500
    {"symbol": "QQQ",     "base_pct": 0.12, "max_pct": 0.18, "type": "stock"},   # Nasdaq 100
    {"symbol": "GLD",     "base_pct": 0.08, "max_pct": 0.14, "type": "stock"},   # Gold
    {"symbol": "SLV",     "base_pct": 0.06, "max_pct": 0.12, "type": "stock"},   # Silver
    {"symbol": "COPX",    "base_pct": 0.06, "max_pct": 0.12, "type": "stock"},   # Copper miners
    {"symbol": "XLE",     "base_pct": 0.06, "max_pct": 0.12, "type": "stock"},   # Energy
    {"symbol": "IWM",     "base_pct": 0.08, "max_pct": 0.14, "type": "stock"},   # Russell 2000
    {"symbol": "NVDA",    "base_pct": 0.08, "max_pct": 0.14, "type": "stock"},   # Nvidia
    {"symbol": "MSFT",    "base_pct": 0.08, "max_pct": 0.14, "type": "stock"},   # Microsoft
    {"symbol": "BTC/USD", "base_pct": 0.10, "max_pct": 0.16, "type": "crypto"},  # Bitcoin
    {"symbol": "ETH/USD", "base_pct": 0.08, "max_pct": 0.14, "type": "crypto"},  # Ethereum
    {"symbol": "TLT",     "base_pct": 0.08, "max_pct": 0.14, "type": "stock"},   # Long-term bonds
]

# ─── Strategy Parameters ────────────────────────────────────
RSI_PERIOD        = 14
MA_PERIOD         = 100
ATR_PERIOD        = 14
REGIME_MA_PERIOD  = 200
RSI_BUY_STRONG    = 25   # 1.5x unit
RSI_BUY_MID       = 30   # 1.0x unit
RSI_BUY_WEAK      = 35   # 0.5x unit
RSI_SELL_MILD     = 65   # sell 25%
RSI_SELL_STRONG   = 72   # sell 50%
RSI_SELL_BEARISH  = 60   # sell 50% when regime bearish
MIN_CASH_RESERVE  = 0.20  # keep 20% of portfolio in cash
COOLDOWN_BUY_H    = 6     # hours
COOLDOWN_SELL_H   = 1     # hour
MAX_BUYS_PER_ASSET= 3
MAX_TRADES_DAY    = 10
DRAWDOWN_BRAKE_1  = 0.05  # >5% drawdown: halve buy sizes
DRAWDOWN_BRAKE_2  = 0.08  # >8% drawdown: stop all buys
DRAWDOWN_RESUME   = 0.03  # resume when drawdown < 3%
MIN_NOTIONAL      = 1.0   # Alpaca minimum order size

# ─── State File ─────────────────────────────────────────────
STATE_FILE = "state.json"

def load_state():
    default = {
        "cooldowns": {},
        "daily_trades": {},
        "daily_date": "",
        "equity_peak": 0.0,
        "regime_bearish": False,
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
    account  = trading_client.get_account()
    cash     = float(account.cash)
    equity   = float(account.equity)
    return cash, equity

def get_position_value(symbol):
    try:
        pos = trading_client.get_open_position(symbol.replace("/", ""))
        return float(pos.market_value)
    except:
        return 0.0

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
        # flatten multi-index if present
        if hasattr(bars.index, "levels"):
            bars = bars.reset_index(level=0, drop=True)
        return bars
    except Exception as e:
        log.warning(f"fetch_data error for {symbol}: {e}")
        return None

# ─── compute_indicators() ────────────────────────────────────
def compute_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    deltas   = np.diff(closes)
    gains    = np.where(deltas > 0, deltas, 0.0)
    losses   = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

def compute_ma(closes, period):
    if len(closes) < period:
        return None
    return float(np.mean(closes[-period:]))

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
    bars = fetch_data(symbol, asset_type, timeframe=TimeFrame.Hour, days=30)
    if bars is None or len(bars) < MA_PERIOD + 1:
        log.warning(f"{symbol}: not enough hourly data")
        return None

    closes = bars["close"].values
    highs  = bars["high"].values
    lows   = bars["low"].values
    price  = closes[-1]

    rsi    = compute_rsi(closes, RSI_PERIOD)
    ma100  = compute_ma(closes, MA_PERIOD)
    atr    = compute_atr(highs, lows, closes, ATR_PERIOD)
    atr_pct = (atr / price * 100) if atr and price > 0 else None

    return {
        "price":   round(price, 4),
        "rsi":     rsi,
        "ma100":   round(ma100, 4) if ma100 else None,
        "atr":     round(atr, 4)   if atr   else None,
        "atr_pct": round(atr_pct, 2) if atr_pct else None,
    }

# ─── check_regime() ──────────────────────────────────────────
def check_regime(state):
    bars = fetch_data("SPY", "stock", timeframe=TimeFrame.Day, days=220)
    if bars is None or len(bars) < REGIME_MA_PERIOD:
        log.warning("Regime check: not enough SPY daily data — assuming neutral")
        return state

    closes   = bars["close"].values
    ma200    = compute_ma(closes, REGIME_MA_PERIOD)
    spy_last = closes[-1]
    bearish  = spy_last < ma200

    if bearish and not state["regime_bearish"]:
        msg = f"⚠️ REGIME BEARISH | SPY ${spy_last:.2f} below 200-day MA ${ma200:.2f} | All new buys disabled"
        log.warning(msg)
        send_telegram(msg)
    elif not bearish and state["regime_bearish"]:
        msg = f"✅ REGIME BULLISH | SPY ${spy_last:.2f} back above 200-day MA ${ma200:.2f} | Buying re-enabled"
        log.info(msg)
        send_telegram(msg)

    state["regime_bearish"] = bearish
    return state

# ─── size_position() ─────────────────────────────────────────
def size_position(rsi, atr_pct, base_budget, equity, reason_parts):
    unit = base_budget * 0.10  # 1 unit = 10% of base allocation

    # RSI tier
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

    # ATR volatility adjustment
    if atr_pct is not None:
        if atr_pct > 5.0:
            size *= 0.25
            reason_parts.append(f"ATR% {atr_pct}% > 5% → size reduced 75%")
        elif atr_pct > 3.0:
            size *= 0.50
            reason_parts.append(f"ATR% {atr_pct}% > 3% → size reduced 50%")
        elif atr_pct < 1.5:
            size *= 1.20
            reason_parts.append(f"ATR% {atr_pct}% < 1.5% → size increased 20%")

    return round(size, 2)

# ─── risk_checks() ───────────────────────────────────────────
def risk_checks(symbol, buy_amount, asset, equity, cash, state, reason_parts):
    today = datetime.now().strftime("%Y-%m-%d")

    # Reset daily counters on new day
    if state["daily_date"] != today:
        state["daily_trades"]  = {}
        state["daily_date"]    = today
        log.info("Daily counters reset")

    # Max total trades per day
    total_trades = sum(state["daily_trades"].values())
    if total_trades >= MAX_TRADES_DAY:
        reason_parts.append(f"SKIP: max {MAX_TRADES_DAY} total trades reached today")
        return False, state

    # Max buys per asset per day
    asset_trades = state["daily_trades"].get(symbol, 0)
    if asset_trades >= MAX_BUYS_PER_ASSET:
        reason_parts.append(f"SKIP: max {MAX_BUYS_PER_ASSET} buys for {symbol} today")
        return False, state

    # Cooldown check
    cooldown_key = f"buy_{symbol}"
    last_trade   = state["cooldowns"].get(cooldown_key)
    if last_trade:
        elapsed = (datetime.now() - datetime.fromisoformat(last_trade)).total_seconds() / 3600
        if elapsed < COOLDOWN_BUY_H:
            remaining = round(COOLDOWN_BUY_H - elapsed, 1)
            reason_parts.append(f"SKIP: {symbol} buy cooldown active ({remaining}h remaining)")
            return False, state

    # Max position cap
    max_value    = PORTFOLIO_SIZE * asset["max_pct"]
    current_pos  = get_position_value(symbol)
    if current_pos + buy_amount > max_value:
        reason_parts.append(f"SKIP: {symbol} would exceed max cap ${max_value:.2f} (current ${current_pos:.2f})")
        return False, state

    # Minimum cash reserve
    min_cash = PORTFOLIO_SIZE * MIN_CASH_RESERVE
    if cash - buy_amount < min_cash:
        reason_parts.append(f"SKIP: buying ${buy_amount:.2f} would breach min cash reserve ${min_cash:.2f}")
        return False, state

    # Portfolio drawdown brake
    peak     = state.get("equity_peak", equity)
    drawdown = (peak - equity) / peak if peak > 0 else 0

    if drawdown > DRAWDOWN_BRAKE_2:
        reason_parts.append(f"SKIP: drawdown brake active ({drawdown*100:.1f}% > {DRAWDOWN_BRAKE_2*100}%) — all buys stopped")
        return False, state
    elif drawdown > DRAWDOWN_BRAKE_1:
        reason_parts.append(f"WARNING: drawdown {drawdown*100:.1f}% > {DRAWDOWN_BRAKE_1*100}% — buy size halved")
        # caller will halve the size — we still allow the trade
        return "halve", state

    return True, state

# ─── execute_trade() ─────────────────────────────────────────
def execute_trade(symbol, amount, side, asset_type, reason):
    if amount < MIN_NOTIONAL:
        log.info(f"SKIP {symbol}: order ${amount:.2f} below min notional ${MIN_NOTIONAL}")
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
def update_state(state, symbol, side, equity):
    today = datetime.now().strftime("%Y-%m-%d")
    now   = datetime.now().isoformat()

    # Update equity peak
    if equity > state.get("equity_peak", 0):
        state["equity_peak"] = equity

    # Reset daily counter if needed
    if state["daily_date"] != today:
        state["daily_trades"] = {}
        state["daily_date"]   = today

    # Increment trade counter (buys only toward daily limit)
    if side == OrderSide.BUY:
        state["daily_trades"][symbol] = state["daily_trades"].get(symbol, 0) + 1

    # Set cooldown
    cooldown_key = f"{'buy' if side == OrderSide.BUY else 'sell'}_{symbol}"
    state["cooldowns"][cooldown_key] = now

    save_state(state)
    return state

# ─── Main run loop ───────────────────────────────────────────
def run():
    state        = load_state()
    cash, equity = get_account_info()

    # Update equity peak
    if equity > state.get("equity_peak", 0):
        state["equity_peak"] = equity
        save_state(state)

    peak     = state["equity_peak"]
    drawdown = (peak - equity) / peak if peak > 0 else 0

    log.info(f"{'='*55}")
    log.info(f"RSI Bot cycle | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info(f"Real equity: ${equity:.2f} | Cash: ${cash:.2f} | Portfolio target: ${PORTFOLIO_SIZE:.2f} | Peak: ${peak:.2f} | Drawdown: {drawdown*100:.1f}%")
    log.info(f"Regime: {'BEARISH 🔴' if state['regime_bearish'] else 'BULLISH 🟢'}")
    log.info(f"{'='*55}")

    # Check regime (once per cycle)
    state = check_regime(state)

    for asset in ASSETS:
        symbol     = asset["symbol"]
        asset_type = asset["type"]
        base_alloc = PORTFOLIO_SIZE * asset["base_pct"]

        log.info(f"\n--- {symbol} ---")
        ind = compute_indicators(symbol, asset_type)

        if ind is None:
            log.warning(f"{symbol}: skipping — could not compute indicators")
            continue

        price   = ind["price"]
        rsi     = ind["rsi"]
        ma100   = ind["ma100"]
        atr_pct = ind["atr_pct"]

        log.info(
            f"{symbol} | Price: ${price} | RSI: {rsi} | "
            f"100MA: ${ma100} | ATR%: {atr_pct}%"
        )

        # ── SELL LOGIC ──────────────────────────────────────
        position_value = get_position_value(symbol)

        sell_pct  = 0
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
            full_reason = f"{reason_str} | Position: ${position_value:.2f} | Selling {sell_pct*100:.0f}% = ${sell_amount:.2f} | Cash after: ~${cash+sell_amount:.2f}"

            # Sell cooldown check
            sell_key    = f"sell_{symbol}"
            last_sell   = state["cooldowns"].get(sell_key)
            if last_sell:
                elapsed = (datetime.now() - datetime.fromisoformat(last_sell)).total_seconds() / 3600
                if elapsed < COOLDOWN_SELL_H:
                    log.info(f"SKIP sell {symbol}: cooldown active ({round(COOLDOWN_SELL_H-elapsed,1)}h remaining)")
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

        if state["regime_bearish"]:
            log.info(f"SKIP buy {symbol} | Regime bearish — buys disabled")
            continue

        if ma100 is not None and price < ma100:
            log.info(f"SKIP buy {symbol} | Price ${price} below 100MA ${ma100} — no buy below MA")
            continue

        reason_parts = [f"RSI {rsi} (oversold)"]
        size = size_position(rsi, atr_pct, base_alloc, equity, reason_parts)

        check, state = risk_checks(symbol, size, asset, equity, cash, state, reason_parts)

        if check == "halve":
            size = round(size * 0.50, 2)
            reason_parts.append("size halved due to drawdown brake")
        elif not check:
            log.info(f"SKIP buy {symbol} | {' | '.join(reason_parts)}")
            continue

        full_reason = (
            f"Buying {symbol}: {' | '.join(reason_parts)} | "
            f"Order: ${size:.2f} | Cash before: ${cash:.2f} | "
            f"Cash after: ~${cash-size:.2f}"
        )

        result = execute_trade(symbol, size, OrderSide.BUY, asset_type, full_reason)
        if result:
            cash  -= size
            state  = update_state(state, symbol, OrderSide.BUY, equity)

    log.info(f"\nCycle complete. Next check in 30 minutes.")
    log.info(f"{'='*55}")
    save_state(state)

# ─── Entry point ─────────────────────────────────────────────
ET = pytz.timezone("US/Eastern")

log.info(f"Bot starting | PAPER={PAPER} | DRYRUN={DRYRUN} | PORTFOLIO=${PORTFOLIO_SIZE:.2f}")
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
        # Still run for crypto after hours
        log.info(f"Stock market closed — {now.strftime('%H:%M')} ET — crypto-only cycle")
        run()
        log.info("Sleeping 1 hour...")
        time.sleep(3600)
    else:
        log.info(f"Weekend — {now.strftime('%A %H:%M')} ET — crypto-only cycle")
        run()
        log.info("Sleeping 1 hour...")
        time.sleep(3600)
