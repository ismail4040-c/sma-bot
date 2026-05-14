"""
SMA Crossover Day Trading Bot
Account: $1,000 | Goal: $10/day
Strategy: Scans 50+ stocks every 5 min, finds best SMA(9)/SMA(21) crossover + VWAP
News filter: MarketWatch RSS — skip trade if negative news on ticker
Hours: 9:30–11:30 AM ET only | Close all by 3:45 PM ET
Risk: $20 max per trade | $50 max daily loss | 3 trades max/day
"""

import os, json, logging, xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
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
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

# Tickers exclusive to the SMA bot — no overlap with the tech trader
# (Tech trader uses: AAPL, MSFT, NVDA, TSLA, AMZN, META, GOOGL, AMD,
#  PLTR, ARM, CLS, SMCI, NFLX, ORCL, ADBE, CRM, JPM, BAC, GS, MS,
#  XOM, CVX, OXY, COIN, MSTR, HOOD, DKNG, RBLX, SNAP, UBER, SHOP,
#  QCOM, INTC, MU, AVGO, RIVN, NIO, SPY, QQQ)
SCAN_UNIVERSE = [
    # ETFs not used by tech trader
    "IWM", "DIA", "GLD", "TLT", "XLF", "XLE", "XLK",
    # Large cap not used by tech trader
    "ORCL", "ADBE", "CRM", "PYPL", "SQ", "ABNB", "LYFT",
    # Finance not used by tech trader
    "WFC", "C", "AXP", "V", "MA",
    # Healthcare
    "JNJ", "PFE", "MRNA", "ABBV", "LLY",
    # Consumer
    "WMT", "COST", "TGT", "HD", "NKE",
    # Industrials / other
    "CAT", "BA", "GE", "F", "GM",
    # Semi not used by tech trader
    "TSM", "AMAT", "LRCX",
    # Media / telecom
    "DIS", "NFLX", "T", "VZ",
]
ET              = ZoneInfo("America/New_York")
MARKET_OPEN     = (9, 30)
TRADE_WINDOW_END = (11, 30)
CLOSE_ALL_TIME  = (15, 45)

MAX_RISK_PER_TRADE = 20.0   # $20 max loss per trade
MAX_DAILY_LOSS     = 50.0   # $50 stop trading for the day
MAX_TRADES_PER_DAY = 3
STOP_LOSS_PCT      = 0.005  # 0.5% stop loss
TAKE_PROFIT_PCT    = 0.010  # 1.0% take profit

STATE_FILE = Path(__file__).parent / "state.json"

NEGATIVE_KEYWORDS = [
    "crash", "plunge", "tumble", "collapse", "fall", "drop", "decline",
    "sell-off", "selloff", "warning", "risk", "recession", "loss",
    "downgrade", "miss", "disappoints", "fears", "concern", "trouble",
    "investigation", "lawsuit", "fraud", "halt", "suspend",
]


# ── State ─────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    today = datetime.now(ET).date().isoformat()
    if STATE_FILE.exists():
        s = json.loads(STATE_FILE.read_text())
        if s.get("date") == today:
            return s
    return {"date": today, "daily_pl": 0.0, "trades_today": 0}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── Alpaca clients ────────────────────────────────────────────────────────────

def get_clients():
    key    = os.environ["ALPACA_API_KEY_TECH"]
    secret = os.environ["ALPACA_SECRET_KEY_TECH"]
    paper  = os.getenv("ALPACA_PAPER", "true").lower() == "true"
    return TradingClient(key, secret, paper=paper), StockHistoricalDataClient(key, secret)


# ── Time helpers ──────────────────────────────────────────────────────────────

def now_et() -> datetime:
    return datetime.now(ET)


def in_trade_window() -> bool:
    t = now_et()
    start = t.replace(hour=MARKET_OPEN[0],      minute=MARKET_OPEN[1],      second=0)
    end   = t.replace(hour=TRADE_WINDOW_END[0], minute=TRADE_WINDOW_END[1], second=0)
    return start <= t <= end


def is_close_all_time() -> bool:
    t = now_et()
    close = t.replace(hour=CLOSE_ALL_TIME[0], minute=CLOSE_ALL_TIME[1], second=0)
    return t >= close


# ── News sentiment (MarketWatch RSS) ─────────────────────────────────────────

def has_negative_news(ticker: str) -> bool:
    """Returns True if recent MarketWatch news contains negative keywords for this ticker."""
    try:
        url = f"https://feeds.marketwatch.com/marketwatch/topstories/"
        r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(r.content)
        headlines = []
        for item in root.iter("item"):
            title = item.findtext("title") or ""
            desc  = item.findtext("description") or ""
            headlines.append((title + " " + desc).lower())

        ticker_lower = ticker.lower()
        relevant = [h for h in headlines if ticker_lower in h]

        for headline in relevant:
            for kw in NEGATIVE_KEYWORDS:
                if kw in headline:
                    log.info("NEWS FILTER blocked %s — negative headline: '%.80s'", ticker, headline)
                    return True
        return False
    except Exception as e:
        log.warning("Could not fetch news (%s), proceeding anyway", e)
        return False


# ── Indicators ────────────────────────────────────────────────────────────────

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cumvol  = df["volume"].cumsum()
    cumtpv  = (typical * df["volume"]).cumsum()
    return cumtpv / cumvol


def sma_crossover_signal(close: pd.Series) -> str:
    """
    Returns 'buy' if SMA9 just crossed above SMA21 on the last bar.
    Returns 'sell' if SMA9 just crossed below SMA21.
    Returns 'none' otherwise.
    """
    if len(close) < 22:
        return "none"
    sma9  = close.rolling(9).mean()
    sma21 = close.rolling(21).mean()

    # Check last two bars for crossover
    prev_above = sma9.iloc[-2] > sma21.iloc[-2]
    curr_above = sma9.iloc[-1] > sma21.iloc[-1]

    if not prev_above and curr_above:
        return "buy"
    if prev_above and not curr_above:
        return "sell"
    return "none"


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


def enter_trade(trading, ticker: str, price: float) -> bool:
    # Size: risk $20 max, stop is 0.5% away
    # qty = max_risk / (price * stop_pct)
    qty = max(1, int(MAX_RISK_PER_TRADE / (price * STOP_LOSS_PCT)))
    stop_price = round(price * (1 - STOP_LOSS_PCT), 2)
    tp_price   = round(price * (1 + TAKE_PROFIT_PCT), 2)
    cost       = qty * price

    log.info("ENTRY BUY %s x%d @ ~$%.2f | stop=$%.2f | tp=$%.2f | risk=$%.2f",
             ticker, qty, price, stop_price, tp_price, qty * price * STOP_LOSS_PCT)
    try:
        trading.submit_order(MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=tp_price),
            stop_loss=StopLossRequest(stop_price=stop_price),
        ))
        log.info("Bracket order submitted: %s", ticker)
        return True
    except Exception as e:
        log.error("Entry failed %s: %s", ticker, e)
        return False


# ── Close all positions at 3:45 PM ───────────────────────────────────────────

def close_all_positions(trading):
    log.info("3:45 PM — closing all open positions.")
    try:
        positions = trading.get_all_positions()
        for p in positions:
            if not any(c.isdigit() for c in p.symbol):  # stocks only
                close_position(trading, p.symbol, {
                    "qty": float(p.qty),
                    "unrealized_pl": float(p.unrealized_pl),
                })
    except Exception as e:
        log.error("Error closing positions: %s", e)


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    state   = load_state()
    trading, data_client = get_clients()

    # Always close all at 3:45 PM
    if is_close_all_time():
        close_all_positions(trading)
        save_state(state)
        return

    # Track daily P&L from account
    try:
        acct = trading.get_account()
        state["daily_pl"] = float(acct.equity) - float(acct.last_equity)
    except Exception:
        pass

    log.info("Daily P&L: $%.2f | Trades today: %d/%d",
             state["daily_pl"], state["trades_today"], MAX_TRADES_PER_DAY)

    # Stop trading if daily loss limit hit
    if state["daily_pl"] <= -MAX_DAILY_LOSS:
        log.warning("Daily loss limit hit ($%.2f). No more trades today.", state["daily_pl"])
        save_state(state)
        return

    # Stop trading if max trades reached
    if state["trades_today"] >= MAX_TRADES_PER_DAY:
        log.info("Max trades (%d) reached for today.", MAX_TRADES_PER_DAY)
        save_state(state)
        return

    # Only trade in the 9:30–11:30 AM window
    if not in_trade_window():
        log.info("Outside trade window (9:30–11:30 AM ET). Current ET: %s", now_et().strftime("%H:%M"))
        save_state(state)
        return

    # Fetch 5-min bars from market open to now
    et_now   = now_et()
    today_open = et_now.replace(hour=9, minute=25, second=0, microsecond=0)
    start    = today_open.astimezone(timezone.utc)
    end      = et_now.astimezone(timezone.utc)

    try:
        req = StockBarsRequest(
            symbol_or_symbols=SCAN_UNIVERSE,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )
        all_bars = data_client.get_stock_bars(req).df
    except Exception as e:
        log.error("Failed to fetch bars: %s", e)
        return

    # ── Score every ticker in the universe ──────────────────────────────────
    buy_signals   = []  # (ticker, price, sma_gap) — sorted by conviction
    sell_signals  = []

    for ticker in SCAN_UNIVERSE:
        try:
            if isinstance(all_bars.index, pd.MultiIndex):
                df = all_bars.xs(ticker, level="symbol").copy()
            else:
                df = all_bars[all_bars.get("symbol") == ticker].copy()
            df = df.sort_index()

            if len(df) < 22:
                continue

            price  = df["close"].iloc[-1]
            vwap   = calculate_vwap(df).iloc[-1]
            signal = sma_crossover_signal(df["close"])
            sma9   = df["close"].rolling(9).mean().iloc[-1]
            sma21  = df["close"].rolling(21).mean().iloc[-1]

            if signal == "buy" and price > vwap:
                # Conviction = how far SMA9 is above SMA21 (bigger gap = stronger momentum)
                gap = (sma9 - sma21) / sma21 * 100
                buy_signals.append((ticker, price, gap))
                log.info("BUY SIGNAL  %s @ $%.2f | SMA9=%.2f SMA21=%.2f VWAP=%.2f | gap=+%.3f%%",
                         ticker, price, sma9, sma21, vwap, gap)

            elif signal == "sell":
                sell_signals.append(ticker)
                log.info("SELL SIGNAL %s @ $%.2f | SMA crossed down", ticker, price)

        except Exception as e:
            log.warning("Skipping %s: %s", ticker, e)

    # ── Exit positions with sell signals ────────────────────────────────────
    for ticker in sell_signals:
        position = get_position(trading, ticker)
        if position:
            log.info("%s: SMA crossed down — closing position", ticker)
            close_position(trading, ticker, position)

    # ── Enter best buy signals (sorted by conviction) ────────────────────
    buy_signals.sort(key=lambda x: x[2], reverse=True)  # strongest gap first

    for ticker, price, gap in buy_signals:
        if state["trades_today"] >= MAX_TRADES_PER_DAY:
            log.info("Max %d trades reached for today.", MAX_TRADES_PER_DAY)
            break
        position = get_position(trading, ticker)
        if position:
            continue  # already in this trade
        if has_negative_news(ticker):
            log.info("%s: skipping — negative news", ticker)
            continue
        log.info("Best setup: %s (SMA gap=+%.3f%%) — entering", ticker, gap)
        if enter_trade(trading, ticker, price):
            state["trades_today"] += 1

    if not buy_signals and not sell_signals:
        log.info("No crossover signals found this cycle across %d tickers.", len(SCAN_UNIVERSE))

    save_state(state)
    log.info("Cycle complete.")


if __name__ == "__main__":
    run()
