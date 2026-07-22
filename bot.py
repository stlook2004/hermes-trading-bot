import anthropic
import databento as db
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import requests

# Initialize clients
databento_key = os.getenv("DATABENTO_API_KEY")
discord_webhook = os.getenv("DISCORD_WEBHOOK_URL")
client = anthropic.Anthropic()

def post_to_discord(embed_data):
    """Post results to Discord webhook"""
    if not discord_webhook:
        print("[Warning] DISCORD_WEBHOOK_URL not set, skipping Discord post")
        return
    
    try:
        payload = {"embeds": [embed_data]}
        response = requests.post(discord_webhook, json=payload)
        if response.status_code == 204:
            print("[Discord] Trade posted successfully")
        else:
            print(f"[Discord] Failed to post: {response.status_code}")
    except Exception as e:
        print(f"[Discord] Error posting: {e}")

def is_trading_day(date: datetime) -> bool:
    """Check if date is a weekday (Mon-Fri)"""
    return date.weekday() < 5

def fetch_and_aggregate_5min_data(symbol: str, trade_date: datetime):
    """Fetch 1-minute data and aggregate to 5-minute bars"""
    try:
        hist_client = db.Historical(databento_key)
        
        # Fetch 1-min bars for the trading day
        start = trade_date.replace(hour=0, minute=0, second=0)
        end = trade_date.replace(hour=23, minute=59, second=59)
        
        print(f"[Databento] Fetching {symbol} 1-min bars for {trade_date.date()}")
        
        data = hist_client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=f"{symbol}.FUT",
            stype_in="parent",
            start=start.isoformat(),
            end=end.isoformat(),
            schema="ohlcv-1m"
        )
        
        df = data.to_df()
        if len(df) == 0:
            print(f"[Databento] No data for {symbol} on {trade_date.date()}")
            return None
        
        print(f"[Databento] Retrieved {len(df)} 1-min bars for {symbol}")
        
        # The index is the timestamp, use it for aggregation
        df.index.name = 'time'
        df = df.reset_index()
        
        # Aggregate to 5-minute bars using the time column
        df['time_5m'] = df['time'].dt.floor('5min')
        bars_5m = df.groupby('time_5m').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()
        
        print(f"[Databento] Aggregated to {len(bars_5m)} 5-min bars")
        return bars_5m
    except Exception as e:
        print(f"[Error] Failed to fetch {symbol} data: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_ai_entry_decision(nq_bars: list, es_bars: list, nq_context: list, es_context: list, strategy: str) -> dict:
    """Use Claude to analyze early bars and decide on entry"""
    
    # Format early bars (first 2 hours = 24 bars of 5-min)
    nq_early = nq_bars[:24] if len(nq_bars) >= 24 else nq_bars
    es_early = es_bars[:24] if len(es_bars) >= 24 else es_bars
    
    nq_early_str = "\n".join([
        f"{i*5}min: O={b['open']:.2f} H={b['high']:.2f} L={b['low']:.2f} C={b['close']:.2f}"
        for i, b in enumerate(nq_early)
    ])
    
    es_early_str = "\n".join([
        f"{i*5}min: O={b['open']:.2f} H={b['high']:.2f} L={b['low']:.2f} C={b['close']:.2f}"
        for i, b in enumerate(es_early)
    ])
    
    nq_context_str = "\n".join([
        f"{d['date']}: O={d['open']:.2f} H={d['high']:.2f} L={d['low']:.2f} C={d['close']:.2f}"
        for d in nq_context[-5:]  # Last 5 days
    ]) if nq_context else "No data"
    
    es_context_str = "\n".join([
        f"{d['date']}: O={d['open']:.2f} H={d['high']:.2f} L={d['low']:.2f} C={d['close']:.2f}"
        for d in es_context[-5:]  # Last 5 days
    ]) if es_context else "No data"
    
    prompt = f"""You are a futures day trader. Your current strategy: {strategy}

Analyze early trading action (first 2 hours) and pick ONE intraday trade to ENTER now.

TODAY'S FIRST 2 HOURS (5-min bars):

NQ:
{nq_early_str}

ES:
{es_early_str}

PRIOR 5 DAYS CONTEXT:

NQ: {nq_context_str}
ES: {es_context_str}

Pick ONE market (NQ or ES) and BUY or SELL based on early momentum. You will close this trade later in the day.

Respond ONLY as JSON:
{{"market": "NQ"|"ES", "action": "BUY"|"SELL", "entry_reason": "one sentence", "confidence": 0.0-1.0}}"""

    try:
        print(f"[Claude] Sending prompt ({len(prompt)} chars)...")
        message = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )
        
        print(f"[Claude] Response received. Content length: {len(message.content)}")
        
        if not message.content or len(message.content) == 0:
            print("[Error] Empty content array from Claude")
            return None
        
        response_text = message.content[0].text
        
        print(f"[Claude] Response text: {response_text}")
        
        if not response_text:
            print("[Error] No text in Claude response")
            return None
        
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx == -1 or end_idx <= start_idx:
            print(f"[Error] No JSON found in response: {response_text}")
            return None
        
        json_str = response_text[start_idx:end_idx]
        result = json.loads(json_str)
        print(f"[Claude] Parsed decision: {result}")
        return result
        
    except Exception as e:
        print(f"[Error] Failed to get entry decision: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_ai_exit_decision(entry_price: float, market: str, bars_since_entry: list, current_bar: dict) -> dict:
    """Use Claude to decide when to exit the trade"""
    
    bars_str = "\n".join([
        f"{i*5}min: O={b['open']:.2f} H={b['high']:.2f} L={b['low']:.2f} C={b['close']:.2f}"
        for i, b in enumerate(bars_since_entry[-12:])  # Last hour
    ])
    
    prompt = f"""You entered a {market} trade at {entry_price:.2f}. Current price: {current_bar['close']:.2f}. P&L: {current_bar['close'] - entry_price:.2f}

Recent bars (last hour):
{bars_str}

Current bar: O={current_bar['open']:.2f} H={current_bar['high']:.2f} L={current_bar['low']:.2f} C={current_bar['close']:.2f}

Should you EXIT now? Consider: profit taking, stop loss, trend reversal.

Respond ONLY as JSON:
{{"should_exit": true|false, "reason": "one sentence"}}"""

    try:
        message = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text if message.content else ""
        
        if not response_text:
            return {"should_exit": False, "reason": "no response"}
        
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx == -1 or end_idx <= start_idx:
            return {"should_exit": False, "reason": "no json"}
        
        json_str = response_text[start_idx:end_idx]
        return json.loads(json_str)
        
    except Exception as e:
        print(f"[Error] Failed to get exit decision: {e}")
        return {"should_exit": False, "reason": "error"}

def get_ai_strategy_update(recent_trades: list, weekly_pnl: float, win_rate: float) -> dict:
    """Use Claude to evaluate strategy and suggest improvements"""
    
    trades_str = "\n".join([
        f"{t['date']}: {t['action']} {t['market']} @ {t['entry_price']:.2f} → {t['exit_price']:.2f} | P&L: ${t['pnl']:.2f}"
        for t in recent_trades[-7:]
    ])
    
    prompt = f"""You are a futures day trading strategy consultant. Review the last 7 trades and evaluate the strategy.

LAST 7 TRADES:
{trades_str}

WEEKLY STATS:
- Total P&L: ${weekly_pnl:.2f}
- Win Rate: {win_rate:.1f}%
- Trades: 7

Based on this performance, should you adjust your trading strategy? Consider:
1. Market selection (NQ vs ES preference)
2. Entry timing (early in the day vs later)
3. Exit strategy (profit targets, stop losses)
4. Risk management

Respond ONLY as JSON:
{{"should_adjust": true|false, "new_strategy": "concise description of updated strategy or null if no change", "rationale": "one sentence explanation"}}"""

    try:
        message = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text if message.content else ""
        
        if not response_text:
            return {"should_adjust": False, "new_strategy": None, "rationale": "no response"}
        
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx == -1 or end_idx <= start_idx:
            return {"should_adjust": False, "new_strategy": None, "rationale": "no json"}
        
        json_str = response_text[start_idx:end_idx]
        return json.loads(json_str)
        
    except Exception as e:
        print(f"[Error] Failed to get strategy update: {e}")
        return {"should_adjust": False, "new_strategy": None, "rationale": "error"}

def load_state():
    """Load current state from persistent volume"""
    state_file = "/data/hermes_state.json"
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
                print(f"[State] Loaded: current_date={state.get('current_date')}, trades={len(state.get('trades', []))}")
                return state
        except Exception as e:
            print(f"[Error] Failed to load state: {e}")
    
    print("[State] Starting fresh from 2020-07-06 (first trading day)")
    return {
        "current_date": "2020-07-06",
        "portfolio": {"position": None, "market": None, "entry_price": 0, "entry_bar_idx": 0},
        "daily_pnl": [],
        "trades": [],
        "strategy": "Early momentum trading: BUY/SELL during first 2 hours based on 5-min bar trends. Exit on reversal or profit target.",
        "strategy_history": []
    }

def save_state(state):
    """Save current state to persistent volume"""
    state_file = "/data/hermes_state.json"
    try:
        with open(state_file, 'w') as f:
            json.dump(state, f)
        print(f"[State] Saved: trades={len(state.get('trades', []))}")
    except Exception as e:
        print(f"[Error] Failed to save state: {e}")

def format_trade_embed(trade_date: str, entry_price: float, exit_price: float, market: str, action: str, pnl: float, state: dict):
    """Format single trade as Discord embed"""
    
    total_pnl = sum(state['daily_pnl'])
    num_trades = len([p for p in state['daily_pnl'] if p != 0])
    winning_trades = len([p for p in state['daily_pnl'] if p > 0])
    win_rate = (winning_trades / num_trades * 100) if num_trades > 0 else 0
    
    color = 0x00ff00 if pnl > 0 else 0xff0000 if pnl < 0 else 0x999999
    
    trade_desc = f"**{action} {market}** @ {entry_price:.2f} → {exit_price:.2f}\nP&L: ${pnl:.2f}"
    
    embed = {
        "title": f"📊 Trade #{len(state['trades'])} - {trade_date}",
        "color": color,
        "fields": [
            {"name": "Trade", "value": trade_desc, "inline": False},
            {"name": "Account Stats", "value": f"Total P&L: ${total_pnl:.2f}\nTrades: {num_trades}\nWin Rate: {win_rate:.1f}%", "inline": False}
        ],
        "footer": {"text": f"Cumulative P&L: ${total_pnl:.2f}"}
    }
    
    return embed

def format_strategy_overview_embed(state: dict, recent_trades: list, strategy_update: dict):
    """Format 7-trade strategy overview as Discord embed"""
    
    weekly_pnl = sum([t['pnl'] for t in recent_trades])
    winning = len([t for t in recent_trades if t['pnl'] > 0])
    win_rate = (winning / len(recent_trades) * 100) if recent_trades else 0
    
    color = 0x00ff00 if weekly_pnl > 0 else 0xff0000 if weekly_pnl < 0 else 0x999999
    
    trades_list = "\n".join([
        f"#{state['trades'].index(t) + 1}: {t['action']} {t['market']} | P&L: ${t['pnl']:.2f}"
        for t in recent_trades
    ])
    
    strategy_change = ""
    if strategy_update.get('should_adjust'):
        strategy_change = f"✅ **STRATEGY UPDATED**\n{strategy_update.get('new_strategy', 'N/A')}\n\n**Rationale**: {strategy_update.get('rationale', 'N/A')}"
    else:
        strategy_change = f"⚪ **NO CHANGE** - Strategy working well\n{strategy_update.get('rationale', 'N/A')}"
    
    embed = {
        "title": f"📈 Strategy Overview - Every 7 Trades",
        "color": color,
        "fields": [
            {"name": "Last 7 Trades", "value": trades_list, "inline": False},
            {"name": "Weekly Performance", "value": f"**Total P&L**: ${weekly_pnl:.2f}\n**Win Rate**: {win_rate:.1f}%\n**Wins**: {winning}/7", "inline": False},
            {"name": "Strategy Review", "value": strategy_change, "inline": False},
            {"name": "Current Strategy", "value": state.get('strategy', 'N/A'), "inline": False}
        ],
        "footer": {"text": f"Total Cumulative P&L: ${sum(state['daily_pnl']):.2f}"}
    }
    
    return embed

def main():
    print("[Hermes] Starting intraday trading test...")
    
    state = load_state()
    trade_date = datetime.strptime(state['current_date'], '%Y-%m-%d')
    
    # Skip weekends
    while not is_trading_day(trade_date):
        print(f"[Hermes] Skipping {trade_date.strftime('%Y-%m-%d')} ({['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][trade_date.weekday()]})")
        trade_date += timedelta(days=1)
    
    print(f"[Hermes] Trading {trade_date.strftime('%Y-%m-%d')}...")
    
    # Fetch and aggregate 5-min data for today
    nq_5min = fetch_and_aggregate_5min_data("NQ", trade_date)
    es_5min = fetch_and_aggregate_5min_data("ES", trade_date)
    
    if nq_5min is None or es_5min is None or len(nq_5min) == 0 or len(es_5min) == 0:
        print("[Error] Failed to fetch 5-min data, skipping day")
        # Move to next day
        next_date = trade_date + timedelta(days=1)
        state['current_date'] = next_date.strftime('%Y-%m-%d')
        save_state(state)
        return
    
    # Convert to bar dicts
    nq_bars = [
        {"open": float(row['open']), "high": float(row['high']), "low": float(row['low']), "close": float(row['close']), "volume": int(row['volume'])}
        for _, row in nq_5min.iterrows()
    ]
    
    es_bars = [
        {"open": float(row['open']), "high": float(row['high']), "low": float(row['low']), "close": float(row['close']), "volume": int(row['volume'])}
        for _, row in es_5min.iterrows()
    ]
    
    print(f"[Hermes] NQ: {len(nq_bars)} 5-min bars, ES: {len(es_bars)} 5-min bars")
    
    # Fetch prior 5 trading days for context
    nq_context = []
    es_context = []
    
    context_date = trade_date - timedelta(days=1)
    days_fetched = 0
    
    while days_fetched < 5 and context_date.year >= 2020:
        if is_trading_day(context_date):
            try:
                prior_nq = fetch_and_aggregate_5min_data("NQ", context_date)
                prior_es = fetch_and_aggregate_5min_data("ES", context_date)
                
                if prior_nq is not None and len(prior_nq) > 0:
                    nq_context.insert(0, {
                        "date": context_date.strftime('%Y-%m-%d'),
                        "open": float(prior_nq.iloc[0]['open']),
                        "high": float(prior_nq['high'].max()),
                        "low": float(prior_nq['low'].min()),
                        "close": float(prior_nq.iloc[-1]['close']),
                        "volume": int(prior_nq['volume'].sum())
                    })
                
                if prior_es is not None and len(prior_es) > 0:
                    es_context.insert(0, {
                        "date": context_date.strftime('%Y-%m-%d'),
                        "open": float(prior_es.iloc[0]['open']),
                        "high": float(prior_es['high'].max()),
                        "low": float(prior_es['low'].min()),
                        "close": float(prior_es.iloc[-1]['close']),
                        "volume": int(prior_es['volume'].sum())
                    })
                
                days_fetched += 1
            except:
                pass
        
        context_date -= timedelta(days=1)
    
    portfolio = state['portfolio']
    
    # ENTRY PHASE: First 2 hours
    if portfolio['position'] is None and len(nq_bars) >= 24:
        print("[Hermes] Calling AI for entry decision...")
        entry_decision = get_ai_entry_decision(nq_bars, es_bars, nq_context, es_context, state.get('strategy', ''))
        
        if entry_decision and entry_decision.get('action') in ['BUY', 'SELL']:
            market = entry_decision.get('market', 'NQ')
            action = entry_decision.get('action')
            bars = nq_bars if market == 'NQ' else es_bars
            entry_price = bars[23]['close']  # Entry at end of 2nd hour
            
            portfolio['position'] = 'LONG' if action == 'BUY' else 'SHORT'
            portfolio['market'] = market
            portfolio['entry_price'] = entry_price
            portfolio['entry_bar_idx'] = 24
            
            print(f"[Trade] ENTRY: {action} {market} @ {entry_price:.2f}")
        else:
            print(f"[Hermes] No entry decision made. Decision: {entry_decision}")
    
    # EXIT PHASE: After entry, look for exit signal
    if portfolio['position'] is not None:
        market = portfolio['market']
        bars = nq_bars if market == 'NQ' else es_bars
        entry_bar_idx = portfolio['entry_bar_idx']
        
        # Check bars from entry onwards
        for bar_idx in range(entry_bar_idx, len(bars)):
            current_bar = bars[bar_idx]
            bars_since_entry = bars[entry_bar_idx:bar_idx+1]
            
            exit_decision = get_ai_exit_decision(
                portfolio['entry_price'],
                market,
                bars_since_entry,
                current_bar
            )
            
            if exit_decision.get('should_exit'):
                exit_price = current_bar['close']
                pnl = (exit_price - portfolio['entry_price']) if portfolio['position'] == 'LONG' else (portfolio['entry_price'] - exit_price)
                
                state['daily_pnl'].append(pnl)
                
                print(f"[Trade] EXIT: {market} @ {exit_price:.2f} | P&L: ${pnl:.2f}")
                
                # Post individual trade to Discord
                embed = format_trade_embed(
                    trade_date.strftime('%Y-%m-%d'),
                    portfolio['entry_price'],
                    exit_price,
                    market,
                    'BUY' if portfolio['position'] == 'LONG' else 'SELL',
                    pnl,
                    state
                )
                post_to_discord(embed)
                
                state['trades'].append({
                    "date": trade_date.strftime('%Y-%m-%d'),
                    "market": market,
                    "action": 'BUY' if portfolio['position'] == 'LONG' else 'SELL',
                    "entry_price": portfolio['entry_price'],
                    "exit_price": exit_price,
                    "pnl": pnl
                })
                
                # Every 7 trades, post strategy overview
                if len(state['trades']) % 7 == 0:
                    print(f"[Hermes] 7 trades reached! Posting strategy overview...")
                    recent_trades = state['trades'][-7:]
                    weekly_pnl = sum([t['pnl'] for t in recent_trades])
                    winning = len([t for t in recent_trades if t['pnl'] > 0])
                    win_rate = (winning / 7 * 100) if recent_trades else 0
                    
                    # Get strategy evaluation from Claude
                    strategy_update = get_ai_strategy_update(state['trades'], weekly_pnl, win_rate)
                    
                    # Update strategy if needed
                    if strategy_update.get('should_adjust') and strategy_update.get('new_strategy'):
                        old_strategy = state.get('strategy', '')
                        state['strategy'] = strategy_update['new_strategy']
                        state['strategy_history'].append({
                            "timestamp": datetime.now().isoformat(),
                            "old_strategy": old_strategy,
                            "new_strategy": state['strategy'],
                            "reason": strategy_update.get('rationale', '')
                        })
                        print(f"[Strategy] Updated: {state['strategy']}")
                    
                    # Post overview to Discord
                    overview_embed = format_strategy_overview_embed(state, recent_trades, strategy_update)
                    post_to_discord(overview_embed)
                
                portfolio['position'] = None
                portfolio['market'] = None
                break
        
        # If no exit signal by end of day, force exit at close
        if portfolio['position'] is not None:
            exit_price = bars[-1]['close']
            pnl = (exit_price - portfolio['entry_price']) if portfolio['position'] == 'LONG' else (portfolio['entry_price'] - exit_price)
            
            state['daily_pnl'].append(pnl)
            
            print(f"[Trade] FORCE EXIT (EOD): {market} @ {exit_price:.2f} | P&L: ${pnl:.2f}")
            
            embed = format_trade_embed(
                trade_date.strftime('%Y-%m-%d'),
                portfolio['entry_price'],
                exit_price,
                market,
                'BUY' if portfolio['position'] == 'LONG' else 'SELL',
                pnl,
                state
            )
            post_to_discord(embed)
            
            state['trades'].append({
                "date": trade_date.strftime('%Y-%m-%d'),
                "market": market,
                "action": 'BUY' if portfolio['position'] == 'LONG' else 'SELL',
                "entry_price": portfolio['entry_price'],
                "exit_price": exit_price,
                "pnl": pnl
            })
            
            # Every 7 trades, post strategy overview
            if len(state['trades']) % 7 == 0:
                print(f"[Hermes] 7 trades reached! Posting strategy overview...")
                recent_trades = state['trades'][-7:]
                weekly_pnl = sum([t['pnl'] for t in recent_trades])
                winning = len([t for t in recent_trades if t['pnl'] > 0])
                win_rate = (winning / 7 * 100) if recent_trades else 0
                
                # Get strategy evaluation from Claude
                strategy_update = get_ai_strategy_update(state['trades'], weekly_pnl, win_rate)
                
                # Update strategy if needed
                if strategy_update.get('should_adjust') and strategy_update.get('new_strategy'):
                    old_strategy = state.get('strategy', '')
                    state['strategy'] = strategy_update['new_strategy']
                    state['strategy_history'].append({
                        "timestamp": datetime.now().isoformat(),
                        "old_strategy": old_strategy,
                        "new_strategy": state['strategy'],
                        "reason": strategy_update.get('rationale', '')
                    })
                    print(f"[Strategy] Updated: {state['strategy']}")
                
                # Post overview to Discord
                overview_embed = format_strategy_overview_embed(state, recent_trades, strategy_update)
                post_to_discord(overview_embed)
            
            portfolio['position'] = None
            portfolio['market'] = None
    
    # Move to next day
    next_date = trade_date + timedelta(days=1)
    state['current_date'] = next_date.strftime('%Y-%m-%d')
    save_state(state)
    
    print(f"[Hermes] {trade_date.strftime('%Y-%m-%d')} complete. Next: {next_date.strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    main()

