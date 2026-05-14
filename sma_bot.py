"""
EMA Crossover Day Trading Bot
Account: $100K in 6 ticker groups (100 virtual $1,000 slots)
Strategy: 9/21 EMA crossover + VWAP + RSI(14) + ATR(14) stops + volume filter
Windows: 9:45–11:30 AM ET | 1:30–3:00 PM ET | Close all 3:45 PM ET
"""

import os, json, logging, xml.etree.ElementTree as ETree
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

# 6 ticker groups — each slot = $1,000 of capital
# Groups are non-overlapping with tech_trader.py
TICKER_GROUPS = [
    {"ticker": "SPY",  "slots": 20},   # slots  1–20:  $20K
    {"ticker": "QQQ",  "slots": 20},   # slots 21–40:  $20K
    {"ticker": "AAPL", "slots": 15},   # slots 41–55:  $15K
    {"ticker": "AMD",  "slots": 15},   # slots 56–70:  $15K
    {"ticker": "NVDA", "slots": 15},   # slots 71–85:  $15K
    {"ticker": "MSFT", "slots": 15},   # slots 86–100: $15K
]

SLOT_SIZE          = 1000.0   # $ per slot
MAX_RISK_PER_SLOT  = 20.0     # 2% of $1,000 — risk per slot per trade
MAX_LOSS_PER_SLOT  = 50.0     # halt group if loss > slots × $50
TARGET_PER_SLOT    = 10.0     # keep trading until group hits slots × $10 profit
ATR_STOP_MULT      = 1.5      # stop_loss  = entry ± ATR × 1.5
ATR_TP_MULT        = 2.5      # take_profit = entry ± ATR × 2.5
EMA_FAST_PERIOD    = 9
EMA_SLOW_PERIOD    = 21
RSI_PERIOD         = 14
ATR_PERIOD         = 14
VOLUME_LOOKBACK    = 20

NY_TZ          = ZoneInfo("America/New_York")
TRADE_WINDOW_1 = ((9, 45), (11, 30))   # morning session
TRADE_WINDOW_2 = ((13, 30), (15, 0))   # afternoon session
CLOSE_ALL_TIME = (15, 45)

STATE_FILE = Path(__file__).parent / "state.json"


# ── State ─────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    today = datetime.now(NY_TZ).date().isoformat()
    if STATE_FILE.exists():
        s = json.loads(STATE_FILE.read_text())
        if s.get("date") == today:
            return s
    return {
        "date": today,
        "total_daily_pl": 0.0,
        "groups": {
            g["ticker"]: {"halted": False, "target_hit": False}
            for g in TICKER_GROUPS
        },
    }


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── Alpaca clients ────────────────────────────────────────────────────────────

def get_clients():
    key    = os.environ["ALPACA_API_KEY_TECH"]
    secret = os.environ["ALPACA_SECRET_KEY_TECH"]
    paper  = os.getenv("ALPACA_PAPER", "true").lower() == "true"
    return TradingClient(key, secret, paper=paper), StockHistoricalDataClient(key, secret)


def get_realized_pl(ticker: str) -> float:
    """Today's realized P&L for a ticker from filled buy/sell orders."""
    key    = os.environ["ALPACA_API_KEY_TECH"]
    secret = os.environ["ALPACA_SECRET_KEY_TECH"]
    today  = datetime.now(NY_TZ).date().isoformat()
    try:
        r = requests.get(
            f"https://paper-api.alpaca.markets/v2/orders"
            f"?status=closed&symbols={ticker}&after={today}T00:00:00Z&limit=100",
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            timeout=10,
        )
        orders = r.json() if r.status_code == 200 else []
        if not isinstance(orders, list):
            return 0.0
        filled = [o for o in orders if o.get("status") == "filled"]
        buy_val  = sum(float(o.get("filled_avg_price", 0)) * float(o.get("filled_qty", 0))
                       for o in filled if o["side"] == "buy")
        sell_val = sum(float(o.get("filled_avg_price", 0)) * float(o.get("filled_qty", 0))
                       for o in filled if o["side"] == "sell")
        return sell_val - buy_val
    except Exception:
        return 0.0


# ── Time helpers ──────────────────────────────────────────────────────────────

def now_et() -> datetime:
    return datetime.now(NY_TZ)


def in_trade_window() -> bool:
    t = now_et()

    def window(start, end) -> bool:
        s = t.replace(hour=start[0], minute=start[1], second=0, microsecond=0)
        e = t.replace(hour=end[0],   minute=end[1],   second=0, microsecond=0)
        return s <= t <= e

    return window(*TRADE_WINDOW_1) or window(*TRADE_WINDOW_2)


def is_close_all_time() -> bool:
    t = now_et()
    close = t.replace(hour=CLOSE_ALL_TIME[0], minute=CLOSE_ALL_TIME[1],
                      second=0, microsecond=0)
    return t >= close


# ── Indicators ────────────────────────────────────────────────────────────────

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    hi, lo, cl = df["high"], df["low"], df["close"]
    tr = pd.concat([
        hi - lo,
        (hi - cl.shift()).abs(),
        (lo - cl.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def calc_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).cumsum() / df["volume"].cumsum()


def get_signal(df: pd.DataFrame):
    """
    Returns (signal, price, cur_atr, stop_price, tp_price).
    signal is 'buy', 'sell', or 'none'.
    """
    min_bars = max(EMA_SLOW_PERIOD, ATR_PERIOD, RSI_PERIOD, VOLUME_LOOKBACK) + 3
    if len(df) < min_bars:
        return "none", 0.0, 0.0, 0.0, 0.0

    e9   = calc_ema(df["close"], EMA_FAST_PERIOD)
    e21  = calc_ema(df["close"], EMA_SLOW_PERIOD)
    cur_rsi  = calc_rsi(df["close"]).iloc[-1]
    cur_atr  = calc_atr(df).iloc[-1]
    cur_vwap = calc_vwap(df).iloc[-1]
    price    = df["close"].iloc[-1]
    cur_vol  = df["volume"].iloc[-1]
    avg_vol  = df["volume"].rolling(VOLUME_LOOKBACK).mean().iloc[-2]  # avoid look-ahead

    if pd.isna(cur_atr) or cur_atr <= 0:
        return "none", price, 0.0, 0.0, 0.0

    prev_above = e9.iloc[-2] > e21.iloc[-2]
    curr_above = e9.iloc[-1] > e21.iloc[-1]

    # LONG: EMA crossed up, price above VWAP, RSI 45–65, volume surge
    if (not prev_above and curr_above
            and price > cur_vwap
            and 45 <= cur_rsi <= 65
            and cur_vol > avg_vol):
        stop = round(price - cur_atr * ATR_STOP_MULT, 2)
        tp   = round(price + cur_atr * ATR_TP_MULT,   2)
        return "buy", price, cur_atr, stop, tp

    # SHORT: EMA crossed down, price below VWAP, RSI 35–55, volume surge
    if (prev_above and not curr_above
            and price < cur_vwap
            and 35 <= cur_rsi <= 55
            and cur_vol > avg_vol):
        stop = round(price + cur_atr * ATR_STOP_MULT, 2)
        tp   = round(price - cur_atr * ATR_TP_MULT,   2)
        return "sell", price, cur_atr, stop, tp

    return "none", price, cur_atr, 0.0, 0.0


# ── Execution ─────────────────────────────────────────────────────────────────

def get_position(trading, ticker: str) -> dict | None:
    try:
        p = trading.get_open_position(ticker)
        return {
            "qty":           float(p.qty),
            "avg_entry":     float(p.avg_entry_price),
            "unrealized_pl": float(p.unrealized_pl),
        }
    except Exception:
        return None


def close_position(trading, ticker: str, position: dict):
    try:
        qty  = abs(int(position["qty"]))
        side = OrderSide.SELL if position["qty"] > 0 else OrderSide.BUY
        trading.submit_order(MarketOrderRequest(
            symbol=ticker, qty=qty, side=side, time_in_force=TimeInForce.DAY,
        ))
        log.info("CLOSED %s x%d | P&L: $%.2f", ticker, qty, position["unrealized_pl"])
    except Exception as e:
        log.error("Failed to close %s: %s", ticker, e)


def enter_trade(trading, ticker: str, signal: str, price: float,
                cur_atr: float, stop: float, tp: float, slots: int) -> bool:
    stop_dist = cur_atr * ATR_STOP_MULT
    max_risk  = slots * MAX_RISK_PER_SLOT          # e.g. 20 slots × $20 = $400
    qty       = max(1, int(max_risk / stop_dist))
    side      = OrderSide.BUY if signal == "buy" else OrderSide.SELL

    log.info(
        "ENTRY %s %s x%d @ ~$%.2f | stop=$%.2f tp=$%.2f | atr=%.3f risk=$%.0f | slots=%d",
        signal.upper(), ticker, qty, price, stop, tp,
        cur_atr, qty * stop_dist, slots,
    )
    try:
        trading.submit_order(MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=tp),
            stop_loss=StopLossRequest(stop_price=stop),
        ))
        log.info("Bracket order submitted: %s", ticker)
        return True
    except Exception as e:
        log.error("Entry failed %s: %s", ticker, e)
        return False


def close_all_positions(trading):
    log.info("3:45 PM — closing all open positions.")
    try:
        for p in trading.get_all_positions():
            if not any(c.isdigit() for c in p.symbol):  # skip options
                close_position(trading, p.symbol, {
                    "qty":           float(p.qty),
                    "unrealized_pl": float(p.unrealized_pl),
                })
    except Exception as e:
        log.error("Error closing positions: %s", e)


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    state   = load_state()
    trading, data_client = get_clients()

    # Force close all at 3:45 PM
    if is_close_all_time():
        close_all_positions(trading)
        save_state(state)
        return

    # Update total daily P&L
    try:
        acct = trading.get_account()
        state["total_daily_pl"] = float(acct.equity) - float(acct.last_equity)
    except Exception:
        pass

    log.info("Daily P&L: $%.2f | ET: %s", state["total_daily_pl"], now_et().strftime("%H:%M"))

    if not in_trade_window():
        log.info("Outside trade windows (9:45–11:30 AM | 1:30–3:00 PM ET).")
        save_state(state)
        return

    # Fetch 5-min bars from market open to now for all tickers
    et_now    = now_et()
    day_start = et_now.replace(hour=9, minute=25, second=0, microsecond=0)
    tickers   = [g["ticker"] for g in TICKER_GROUPS]

    try:
        req = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=day_start.astimezone(timezone.utc),
            end=et_now.astimezone(timezone.utc),
            feed=DataFeed.IEX,
        )
        all_bars = data_client.get_stock_bars(req).df
    except Exception as e:
        log.error("Failed to fetch bars: %s", e)
        return

    # ── Process each ticker group ────────────────────────────────────────────
    for group in TICKER_GROUPS:
        ticker   = group["ticker"]
        slots    = group["slots"]
        gs       = state["groups"][ticker]
        target   = slots * TARGET_PER_SLOT     # e.g. 20 slots × $10 = $200
        max_loss = slots * MAX_LOSS_PER_SLOT   # e.g. 20 slots × $50 = $1,000

        if gs["halted"]:
            log.info("%s: halted for today (loss limit hit)", ticker)
            continue

        # Check realized P&L to see if daily target is already met
        realized_pl = get_realized_pl(ticker)
        position    = get_position(trading, ticker)
        unrealized  = float(position["unrealized_pl"]) if position else 0.0
        group_pl    = realized_pl + unrealized

        if gs["target_hit"] or group_pl >= target:
            gs["target_hit"] = True
            log.info(
                "%s: ✅ TARGET HIT — P&L=$%.2f / target=$%.2f. Done for today.",
                ticker, group_pl, target,
            )
            continue

        log.info(
            "%s: P&L=$%.2f / target=$%.2f | still working towards goal",
            ticker, group_pl, target,
        )

        # Open position — monitor loss limit, otherwise hold
        if position:
            if unrealized <= -max_loss:
                log.warning(
                    "%s: loss limit hit (unrealized=$%.2f / limit=$%.0f). Closing + halting.",
                    ticker, unrealized, max_loss,
                )
                close_position(trading, ticker, position)
                gs["halted"] = True
            else:
                log.info("%s: holding open position (unrealized=$%.2f)", ticker, unrealized)
            continue

        # No open position and target not met — look for next entry
        try:
            if isinstance(all_bars.index, pd.MultiIndex):
                df = all_bars.xs(ticker, level="symbol").copy()
            else:
                df = all_bars[all_bars.get("symbol") == ticker].copy()
            df = df.sort_index()
        except Exception as e:
            log.warning("No bars for %s: %s", ticker, e)
            continue

        if len(df) < 25:
            log.info("%s: only %d bars, need ≥25 — skipping", ticker, len(df))
            continue

        signal, price, cur_atr, stop, tp = get_signal(df)

        if signal == "none":
            log.info(
                "%s: no signal yet | price=%.2f atr=%.3f rsi=%.1f",
                ticker, price, cur_atr,
                calc_rsi(df["close"]).iloc[-1] if len(df) >= RSI_PERIOD + 1 else 0,
            )
            continue

        log.info(
            "%s: %s signal | price=$%.2f atr=%.3f stop=$%.2f tp=$%.2f | "
            "slots=%d target=$%.0f remaining=$%.2f",
            ticker, signal.upper(), price, cur_atr, stop, tp,
            slots, target, target - group_pl,
        )

        enter_trade(trading, ticker, signal, price, cur_atr, stop, tp, slots)

    save_state(state)
    log.info(
        "Cycle complete. Status: %s",
        {k: ("TARGET ✅" if v["target_hit"] else "HALTED ❌" if v["halted"] else "working...")
         for k, v in state["groups"].items()},
    )


if __name__ == "__main__":
    run()
