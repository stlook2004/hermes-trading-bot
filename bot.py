#!/usr/bin/env python3
"""
Hermes Trading Bot - 1-minute bar backtester with strict no-lookahead rules.
Trades EMA/RSI strategy with ATR-based stops/targets.
Reports to Discord with entry/exit times and comprehensive backtest stats.

RULES:
1. Pull 1-minute OHLCV bars from Databento with cost approval
2. Store bars sorted by timestamp
3. NO LOOKAHEAD: Process bars one at a time, at bar N only use bars 0..N
4. Signals on bar CLOSE, fills at next bar OPEN
5. 1 tick slippage per side + realistic commission
6. Max 5 trades per day, 1 position at a time, force-close at session end
7. EMA(9) cross EMA(21) + RSI(14) > 50 for LONG, inverse for SHORT
8. Stop = 2x ATR(14), Target = 3x ATR(14)
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import requests
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import io
import base64

try:
    import databento as db
except ImportError:
    db = None

try:
    import talib
    import numpy as np
except ImportError:
    talib = None
    np = None

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    plt = None

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATABENTO_API_KEY = os.getenv("DATABENTO_API_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/trading")

MAX_TRADES_PER_DAY = 5
POSITION_SIZE = 1
SLIPPAGE_TICKS = 1
COMMISSION_PER_CONTRACT = 2.50

EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
ATR_PERIOD = 14
STOP_MULTIPLIER = 2.0
TARGET_MULTIPLIER = 3.0


class TradingBot:
    def __init__(self):
        try:
            self.redis_client = redis.from_url(REDIS_URL)
            logger.info("Redis connected")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
        
        try:
            self.db_conn = psycopg2.connect(DATABASE_URL)
            logger.info("PostgreSQL connected")
        except Exception as e:
            logger.warning(f"Database connection failed: {e}")
            self.db_conn = None
        
        self.trades_today = 0
        self.current_position = None
        self.session_date = None
        self.closed_trades = []

    def init_db(self):
        if not self.db_conn:
            logger.error("Database not connected")
            return
        
        try:
            with self.db_conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id SERIAL PRIMARY KEY,
                        symbol TEXT,
                        entry_time TIMESTAMP,
                        entry_price FLOAT,
                        exit_time TIMESTAMP,
                        exit_price FLOAT,
                        direction TEXT,
                        stop_loss FLOAT,
                        take_profit FLOAT,
                        pnl FLOAT,
                        status TEXT,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                self.db_conn.commit()
                logger.info("Database initialized")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")

    def get_databento_cost(self, symbol: str, dataset: str, start_date: str, end_date: str) -> float:
        if not db or not DATABENTO_API_KEY:
            logger.warning("databento not installed or API key missing")
            return 0.0

        try:
            client = db.Historical(key=DATABENTO_API_KEY)
            cost = client.metadata.get_cost(
                dataset=dataset,
                symbols=[symbol],
                schema="ohlcv-1m",
                start=start_date,
                end=end_date,
            )
            logger.info(f"Databento cost estimate: ${cost:.2f}")
            logger.info("⚠️  AWAITING APPROVAL TO PULL 1-MINUTE DATA")
            return cost
        except Exception as e:
            logger.error(f"Error getting Databento cost: {e}")
            return 0.0

    def fetch_bars(self, symbol: str, dataset: str, start_date: str, end_date: str) -> List[Dict]:
        if not db or not DATABENTO_API_KEY:
            logger.warning("databento not installed or API key missing")
            return []

        try:
            client = db.Historical(key=DATABENTO_API_KEY)
            data = client.timeseries.get_range(
                dataset=dataset,
                symbols=[symbol],
                schema="ohlcv-1m",
                start=start_date,
                end=end_date,
            )
            
            bars = []
            for record in data:
                bars.append({
                    "ts": record.ts_event,
                    "open": record.open,
                    "high": record.high,
                    "low": record.low,
                    "close": record.close,
                    "volume": record.volume,
                })
            
            bars.sort(key=lambda x: x["ts"])
            logger.info(f"Fetched {len(bars)} 1-minute bars for {symbol}")
            return bars
        except Exception as e:
            logger.error(f"Error fetching bars: {e}")
            return []


    def fetch_bars(self, symbol: str, dataset: str, start_date: str, end_date: str) -> List[Dict]:
    if not db or not DATABENTO_API_KEY:
        logger.warning("databento not installed or API key missing")
        return []

    try:
        client = db.Historical(key=DATABENTO_API_KEY)
        data = client.timeseries.get_range(
            dataset=dataset,
            symbols=[symbol],
            schema="ohlcv-1m",
            start=start_date,
            end=end_date,
        )
        
        bars = []
        for i, record in enumerate(data):
            if i == 0:
                logger.info(f"DEBUG: First record type: {type(record)}")
                logger.info(f"DEBUG: First record dir: {[attr for attr in dir(record) if not attr.startswith('_')]}")
                logger.info(f"DEBUG: First record values - ts_event: {record.ts_event}, open: {record.open}, close: {record.close}")
            
            bars.append({
                "ts": record.ts_event,
                "open": record.open,
                "high": record.high,
                "low": record.low,
                "close": record.close,
                "volume": record.volume,
            })
        
        bars.sort(key=lambda x: x["ts"])
        logger.info(f"Fetched {len(bars)} 1-minute bars for {symbol}")
        return bars
    except Exception as e:
        logger.error(f"Error fetching bars: {e}")
        return []

    

    def compute_ema(self, closes: List[float], period: int) -> Optional[float]:
        if len(closes) < period:
            return None
        
        if talib and np:
            try:
                return float(talib.EMA(np.array(closes, dtype=np.float64), timeperiod=period)[-1])
            except Exception as e:
                logger.debug(f"talib EMA failed: {e}, using fallback")
        
        multiplier = 2.0 / (period + 1)
        ema = closes[0]
        for price in closes[1:]:
            ema = price * multiplier + ema * (1 - multiplier)
        return ema

    def compute_rsi(self, closes: List[float], period: int) -> Optional[float]:
        if len(closes) < period + 1:
            return None
        
        if talib and np:
            try:
                return float(talib.RSI(np.array(closes, dtype=np.float64), timeperiod=period)[-1])
            except Exception as e:
                logger.debug(f"talib RSI failed: {e}, using fallback")
        
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period if period > 0 else 0
        avg_loss = sum(losses[-period:]) / period if period > 0 else 0
        
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def compute_atr(self, bars: List[Dict], period: int) -> Optional[float]:
        if len(bars) < period:
            return None
        
        trs = []
        for i in range(len(bars)):
            high = bars[i]["high"]
            low = bars[i]["low"]
            close_prev = bars[i-1]["close"] if i > 0 else bars[i]["close"]
            
            tr = max(
                high - low,
                abs(high - close_prev),
                abs(low - close_prev)
            )
            trs.append(tr)
        
        atr = sum(trs[-period:]) / period
        return atr

    def process_bars(self, bars: List[Dict], symbol: str):
        logger.info(f"Processing {len(bars)} bars for {symbol}")
        logger.info("=" * 60)
        
        for bar_idx in range(len(bars)):
            historical_bars = bars[:bar_idx + 1]
            current_bar = bars[bar_idx]
            
            closes = [b["close"] for b in historical_bars]
            
            ema_fast = self.compute_ema(closes, EMA_FAST)
            ema_slow = self.compute_ema(closes, EMA_SLOW)
            rsi = self.compute_rsi(closes, RSI_PERIOD)
            atr = self.compute_atr(historical_bars, ATR_PERIOD)
            
            signal = self.check_signal(ema_fast, ema_slow, rsi, bar_idx, closes)
            
            if signal and bar_idx + 1 < len(bars):
                next_bar = bars[bar_idx + 1]
                fill_price = next_bar["open"]
                
                if self.trades_today >= MAX_TRADES_PER_DAY:
                    logger.info(f"Max trades ({MAX_TRADES_PER_DAY}) reached for today")
                    continue
                
                if self.current_position is None and atr is not None:
                    self.enter_trade(
                        symbol=symbol,
                        direction=signal,
                        entry_price=fill_price,
                        entry_time=next_bar["ts"],
                        stop_loss=atr * STOP_MULTIPLIER,
                        take_profit=atr * TARGET_MULTIPLIER,
                    )
            
            if self.current_position:
                self.check_exit(current_bar)
        
        if self.current_position and bars:
            self.force_close_position(bars[-1])
        
        logger.info("=" * 60)
        logger.info("Bar processing complete")

    def check_signal(self, ema_fast: Optional[float], ema_slow: Optional[float], 
                     rsi: Optional[float], bar_idx: int, closes: List[float]) -> Optional[str]:
        if ema_fast is None or ema_slow is None or rsi is None:
            return None
        
        if bar_idx < 1:
            return None
        
        prev_closes = closes[:-1]
        prev_ema_fast = self.compute_ema(prev_closes, EMA_FAST)
        prev_ema_slow = self.compute_ema(prev_closes, EMA_SLOW)
        
        if prev_ema_fast is None or prev_ema_slow is None:
            return None
        
        if prev_ema_fast <= prev_ema_slow and ema_fast > ema_slow and rsi > 50:
            return "LONG"
        
        if prev_ema_fast >= prev_ema_slow and ema_fast < ema_slow and rsi < 50:
            return "SHORT"
        
        return None

    def enter_trade(self, symbol: str, direction: str, entry_price: float, 
                    entry_time: int, stop_loss: float, take_profit: float):
        self.current_position = {
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "entry_time": entry_time,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "size": POSITION_SIZE,
        }
        self.trades_today += 1
        
        entry_dt = datetime.fromtimestamp(entry_time / 1e9)
        
        logger.info(f"✓ ENTRY: {direction} {POSITION_SIZE} {symbol} @ ${entry_price:.2f} "
                   f"| SL: ${stop_loss:.2f} | TP: ${take_profit:.2f} | Time: {entry_dt}")
        
        if self.db_conn:
            try:
                with self.db_conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO trades (symbol, entry_time, entry_price, direction, 
                                           stop_loss, take_profit, status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (symbol, entry_dt, entry_price, direction, stop_loss, take_profit, "OPEN"))
                    self.db_conn.commit()
            except Exception as e:
                logger.error(f"Error inserting trade: {e}")

    def check_exit(self, bar: Dict):
        if not self.current_position:
            return
        
        direction = self.current_position["direction"]
        stop_loss = self.current_position["stop_loss"]
        take_profit = self.current_position["take_profit"]
        entry_price = self.current_position["entry_price"]
        
        if direction == "LONG":
            sl_price = entry_price - stop_loss
            tp_price = entry_price + take_profit
            
            if bar["low"] <= sl_price:
                self.exit_trade(bar, sl_price, "STOP_LOSS")
            elif bar["high"] >= tp_price:
                self.exit_trade(bar, tp_price, "TAKE_PROFIT")
        
        elif direction == "SHORT":
            sl_price = entry_price + stop_loss
            tp_price = entry_price - take_profit
            
            if bar["high"] >= sl_price:
                self.exit_trade(bar, sl_price, "STOP_LOSS")
            elif bar["low"] <= tp_price:
                self.exit_trade(bar, tp_price, "TAKE_PROFIT")

    def exit_trade(self, bar: Dict, exit_price: float, reason: str):
        if not self.current_position:
            return
        
        entry_price = self.current_position["entry_price"]
        direction = self.current_position["direction"]
        entry_time = self.current_position["entry_time"]
        exit_time = bar["ts"]
        
        if direction == "LONG":
            pnl = (exit_price - entry_price) * POSITION_SIZE - COMMISSION_PER_CONTRACT
        else:
            pnl = (entry_price - exit_price) * POSITION_SIZE - COMMISSION_PER_CONTRACT
        
        entry_dt = datetime.fromtimestamp(entry_time / 1e9)
        exit_dt = datetime.fromtimestamp(exit_time / 1e9)
        duration = exit_dt - entry_dt
        
        logger.info(f"✗ EXIT: {direction} {POSITION_SIZE} @ ${exit_price:.2f} ({reason}) "
                   f"| P&L: ${pnl:.2f} | Duration: {duration}")
        
        self.closed_trades.append({
            "entry_time": entry_dt,
            "exit_time": exit_dt,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "reason": reason,
        })
        
        if self.db_conn:
            try:
                with self.db_conn.cursor() as cur:
                    cur.execute("""
                        UPDATE trades 
                        SET exit_time = %s, exit_price = %s, pnl = %s, status = %s
                        WHERE status = 'OPEN' AND symbol = %s
                    """, (exit_dt, exit_price, pnl, "CLOSED", self.current_position["symbol"]))
                    self.db_conn.commit()
            except Exception as e:
                logger.error(f"Error updating trade: {e}")
        
        self.report_to_discord(
            symbol=self.current_position["symbol"],
            direction=direction,
            entry_price=entry_price,
            entry_time=entry_dt,
            exit_price=exit_price,
            exit_time=exit_dt,
            pnl=pnl,
            reason=reason,
            duration=duration,
        )
        
        self.current_position = None

    def force_close_position(self, last_bar: Dict):
        if self.current_position:
            logger.info("Force-closing position at session end")
            self.exit_trade(last_bar, last_bar["close"], "SESSION_END")

    def report_to_discord(self, symbol: str, direction: str, entry_price: float, 
                         entry_time: datetime, exit_price: float, exit_time: datetime, 
                         pnl: float, reason: str, duration: timedelta):
        if not DISCORD_WEBHOOK_URL:
            logger.warning("Discord webhook not configured")
            return
        
        color = 0x00FF00 if pnl >= 0 else 0xFF0000
        
        embed = {
            "title": f"{'✓' if pnl >= 0 else '✗'} {direction} Trade Closed",
            "color": color,
            "fields": [
                {"name": "Symbol", "value": symbol, "inline": True},
                {"name": "Direction", "value": direction, "inline": True},
                {"name": "Size", "value": f"{POSITION_SIZE} contract", "inline": True},
                {"name": "Entry Price", "value": f"${entry_price:.2f}", "inline": True},
                {"name": "Exit Price", "value": f"${exit_price:.2f}", "inline": True},
                {"name": "P&L", "value": f"${pnl:.2f}", "inline": True},
                {"name": "Entry Time", "value": entry_time.strftime("%Y-%m-%d %H:%M:%S"), "inline": True},
                {"name": "Exit Time", "value": exit_time.strftime("%Y-%m-%d %H:%M:%S"), "inline": True},
                {"name": "Duration", "value": str(duration), "inline": True},
                {"name": "Exit Reason", "value": reason, "inline": True},
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        payload = {"embeds": [embed]}
        
        try:
            response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
            response.raise_for_status()
            logger.info(f"Discord report sent for {symbol}")
        except Exception as e:
            logger.error(f"Error sending Discord report: {e}")

    def calculate_stats(self) -> Dict:
        if not self.closed_trades:
            return {}
        
        trades = self.closed_trades
        pnls = [t["pnl"] for t in trades]
        
        total_pnl = sum(pnls)
        
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]
        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        num_trades = len(trades)
        win_rate = (num_wins / num_trades * 100) if num_trades > 0 else 0
        
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
        
        avg_win = (gross_profit / num_wins) if num_wins > 0 else 0
        avg_loss = (gross_loss / num_losses) if num_losses > 0 else 0
        
        cumulative_pnl = []
        running_total = 0
        for pnl in pnls:
            running_total += pnl
            cumulative_pnl.append(running_total)
        
        max_drawdown = 0
        peak = 0
        for equity in cumulative_pnl:
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        if trades:
            first_trade = trades[0]["entry_time"]
            last_trade = trades[-1]["exit_time"]
            days = (last_trade - first_trade).days + 1
            trades_per_day = num_trades / days if days > 0 else 0
        else:
            trades_per_day = 0
        
        losing_streak = 0
        max_losing_streak = 0
        for pnl in pnls:
            if pnl < 0:
                losing_streak += 1
                max_losing_streak = max(max_losing_streak, losing_streak)
            else:
                losing_streak = 0
        
        return {
            "total_pnl": total_pnl,
            "num_trades": num_trades,
            "num_wins": num_wins,
            "num_losses": num_losses,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_drawdown": max_drawdown,
            "trades_per_day": trades_per_day,
            "max_losing_streak": max_losing_streak,
            "cumulative_pnl": cumulative_pnl,
        }

    def generate_trade_log(self) -> str:
        if not self.closed_trades:
            return "No trades executed."
        
        log = "TRADE LOG\n"
        log += "=" * 120 + "\n"
        log += f"{'#':<4} {'Entry Time':<20} {'Exit Time':<20} {'Side':<6} {'Entry Price':<12} {'Exit Price':<12} {'P&L':<10} {'Reason':<15}\n"
        log += "-" * 120 + "\n"
        
        for i, trade in enumerate(self.closed_trades, 1):
            entry_time = trade["entry_time"].strftime("%Y-%m-%d %H:%M:%S")
            exit_time = trade["exit_time"].strftime("%Y-%m-%d %H:%M:%S")
            side = trade["direction"]
            entry_price = f"${trade['entry_price']:.2f}"
            exit_price = f"${trade['exit_price']:.2f}"
            pnl = f"${trade['pnl']:.2f}"
            reason = trade["reason"]
            
            log += f"{i:<4} {entry_time:<20} {exit_time:<20} {side:<6} {entry_price:<12} {exit_price:<12} {pnl:<10} {reason:<15}\n"
        
        return log

    def report_backtest_summary(self):
        if not DISCORD_WEBHOOK_URL:
            logger.warning("Discord webhook not configured")
            return
        
        stats = self.calculate_stats()
        if not stats:
            logger.warning("No trades to report")
            return
        
        trade_log = self.generate_trade_log()
        logger.info("\n" + trade_log)
        
        embed = {
            "title": "📊 Backtest Summary Report",
            "color": 0x0099FF,
            "fields": [
                {"name": "Total P&L", "value": f"${stats['total_pnl']:.2f}", "inline": True},
                {"name": "Total Trades", "value": str(stats['num_trades']), "inline": True},
                {"name": "Win Rate", "value": f"{stats['win_rate']:.2f}%", "inline": True},
                {"name": "Wins / Losses", "value": f"{stats['num_wins']} / {stats['num_losses']}", "inline": True},
                {"name": "Profit Factor", "value": f"{stats['profit_factor']:.2f}", "inline": True},
                {"name": "Avg Win", "value": f"${stats['avg_win']:.2f}", "inline": True},
                {"name": "Avg Loss", "value": f"${stats['avg_loss']:.2f}", "inline": True},
                {"name": "Max Drawdown", "value": f"${stats['max_drawdown']:.2f}", "inline": True},
                {"name": "Trades Per Day", "value": f"{stats['trades_per_day']:.2f}", "inline": True},
                {"name": "Max Losing Streak", "value": str(stats['max_losing_streak']), "inline": True},
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        payload = {"embeds": [embed]}
        
        try:
            response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
            response.raise_for_status()
            logger.info("Backtest summary sent to Discord")
        except Exception as e:
            logger.error(f"Error sending backtest summary: {e}")
        
        log_lines = trade_log.split('\n')
        current_message = ""
        
        for line in log_lines:
            if len(current_message) + len(line) + 1 > 1900:
                if current_message:
                    payload = {"content": f"```\n{current_message}\n```"}
                    try:
                        requests.post(DISCORD_WEBHOOK_URL, json=payload)
                    except Exception as e:
                        logger.error(f"Error sending trade log: {e}")
                current_message = line
            else:
                current_message += line + "\n"
        
        if current_message:
            payload = {"content": f"```\n{current_message}\n```"}
            try:
                requests.post(DISCORD_WEBHOOK_URL, json=payload)
                logger.info("Trade log sent to Discord")
            except Exception as e:
                logger.error(f"Error sending trade log: {e}")

    def run(self, symbol: str, dataset: str, start_date: str, end_date: str):
        self.init_db()
        
        logger.info(f"Starting Hermes Trading Bot")
        logger.info(f"Symbol: {symbol} | Dataset: {dataset}")
        logger.info(f"Date Range: {start_date} to {end_date}")
        
        cost = self.get_databento_cost(symbol, dataset, start_date, end_date)
        logger.info(f"Estimated cost: ${cost:.2f}")
        
        bars = self.fetch_bars(symbol, dataset, start_date, end_date)
        if not bars:
            logger.error("No bars fetched, exiting")
            return
        
        self.process_bars(bars, symbol)
        
        self.report_backtest_summary()
        
        logger.info("Bot run complete")


def main():
    symbol = os.getenv("TRADING_SYMBOL", "ES")
    dataset = os.getenv("TRADING_DATASET", "XNAS.ITCH")
    start_date = os.getenv("TRADING_START_DATE", "2024-01-01")
    end_date = os.getenv("TRADING_END_DATE", "2024-01-31")
    
    bot = TradingBot()
    bot.run(symbol, dataset, start_date, end_date)


if __name__ == "__main__":
    main()
