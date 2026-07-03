import anthropic
import databento as db
import json
import os
import sys
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
        payload = {
            "embeds": [embed_data]
        }
        response = requests.post(discord_webhook, json=payload)
        if response.status_code == 204:
            print("[Discord] Trade posted successfully")
        else:
            print(f"[Discord] Failed to post: {response.status_code}")
    except Exception as e:
        print(f"[Discord] Error posting: {e}")

def fetch_futures_data(symbol: str, days_back: int = 2190):
    """Fetch 6 years of historical futures data from Databento"""
    try:
        hist_client = db.Historical(databento_key)
        
        # Use yesterday's date as end (Databento only has data up to previous day close)
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=days_back)
        
        print(f"[Databento] Fetching {symbol} from {start_date.date()} to {end_date.date()}")
        
        # Use GLBX.MDP3 for CME futures (NQ and ES)
        # Symbol format: NQ.FUT or ES.FUT
        data = hist_client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=f"{symbol}.FUT",
            stype_in="parent",
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            schema="ohlcv-1d"
        )
        
        # Convert to DataFrame for easier processing
        df = data.to_df()
        df = df.sort_values('ts_event').reset_index(drop=True)
        print(f"[Databento] Retrieved {len(df)} bars for {symbol}")
        return df
    except Exception as e:
        print(f"[Error] Failed to fetch {symbol} data: {e}")
        return None

def get_ai_trade_decision(nq_data: dict, es_data: dict, nq_context: list, es_context: list) -> dict:
    """Use Claude to analyze both NQ and ES and pick the best trade"""
    
    nq_context_str = ""
    if len(nq_context) > 0:
        recent_days = nq_context[-10:]
        nq_context_str = "\n".join([
            f"{d['date']}: O={d['open']:.2f} H={d['high']:.2f} L={d['low']:.2f} C={d['close']:.2f}"
            for d in recent_days
        ])
    
    es_context_str = ""
    if len(es_context) > 0:
        recent_days = es_context[-10:]
        es_context_str = "\n".join([
            f"{d['date']}: O={d['open']:.2f} H={d['high']:.2f} L={d['low']:.2f} C={d['close']:.2f}"
            for d in recent_days
        ])
    
    prompt = f"""You are a futures trader managing a single account. Analyze both NQ (Nasdaq 100 Mini) and ES (S&P 500 Mini) for today and pick the BEST trade opportunity.

TODAY'S DATA:

NQ (Nasdaq 100 Mini):
Open: {nq_data['open']:.2f}
High: {nq_data['high']:.2f}
Low: {nq_data['low']:.2f}
Close: {nq_data['close']:.2f}
Volume: {nq_data['volume']}

ES (S&P 500 Mini):
Open: {es_data['open']:.2f}
High: {es_data['high']:.2f}
Low: {es_data['low']:.2f}
Close: {es_data['close']:.2f}
Volume: {es_data['volume']}

NQ LAST 10 DAYS:
{nq_context_str if nq_context_str else "No prior data"}

ES LAST 10 DAYS:
{es_context_str if es_context_str else "No prior data"}

Analyze both markets and pick the SINGLE BEST trade for today. Consider:
- Momentum and trend strength
- Volume patterns
- Relative strength between markets
- Risk/reward setup

Respond ONLY with JSON:
{{"market": "NQ"|"ES", "action": "BUY"|"SELL"|"HOLD", "confidence": 0.0-1.0, "reason": "brief explanation of why this market and action"}}"""

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    
    try:
        response_text = message.content[0].text
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)
    except Exception as e:
        print(f"[Error] Failed to parse AI response: {e}")
    
    return {"market": "NONE", "action": "HOLD", "confidence": 0.5, "reason": "parse error"}

def load_state():
    """Load current state from file"""
    state_file = "/tmp/hermes_state.json"
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                return json.load(f)
        except:
            pass
    
    return {
        "current_day_index": 0,
        "portfolio": {"position": None, "market": None, "entry_price": 0},
        "daily_pnl": [],
        "trades": []
    }

def save_state(state):
    """Save current state to file"""
    state_file = "/tmp/hermes_state.json"
    with open(state_file, 'w') as f:
        json.dump(state, f)

def format_discord_embed(nq_data, es_data, decision, pnl, state):
    """Format single trade as Discord embed"""
    
    # Calculate stats
    total_pnl = sum(state['daily_pnl'])
    num_trades = len([p for p in state['daily_pnl'] if p != 0])
    winning_trades = len([p for p in state['daily_pnl'] if p > 0])
    win_rate = (winning_trades / num_trades * 100) if num_trades > 0 else 0
    
    # Determine color based on action
    color_map = {
        "BUY": 0x0099ff,   # Blue
        "SELL": 0xff6600,  # Orange
        "HOLD": 0x999999   # Gray
    }
    
    action = decision.get('action', 'HOLD')
    market = decision.get('market', 'NONE')
    color = color_map.get(action, 0x999999)
    
    # Get the traded market data
    if market == "NQ":
        traded_data = nq_data
        market_name = "NQ (Nasdaq 100 Mini)"
    elif market == "ES":
        traded_data = es_data
        market_name = "ES (S&P 500 Mini)"
    else:
        traded_data = None
        market_name = "NONE"
    
    # Build trade description
    if action == "BUY" and market != "NONE":
        trade_desc = f"🟢 **BUY {market}** @ {traded_data['close']:.2f}"
    elif action == "SELL" and market != "NONE":
        trade_desc = f"🔴 **SELL {market}** @ {traded_data['close']:.2f}\nP&L: ${pnl:.2f}"
    else:
        trade_desc = f"⚪ **HOLD** - No trade"
    
    # Build market comparison
    market_comp = f"""**NQ**: O={nq_data['open']:.2f} H={nq_data['high']:.2f} L={nq_data['low']:.2f} C={nq_data['close']:.2f}
**ES**: O={es_data['open']:.2f} H={es_data['high']:.2f} L={es_data['low']:.2f} C={es_data['close']:.2f}"""
    
    embed = {
        "title": f"📊 Futures Trading - {nq_data['date']}",
        "color": color,
        "fields": [
            {
                "name": "Trade Decision",
                "value": trade_desc,
                "inline": False
            },
            {
                "name": "Market Data",
                "value": market_comp,
                "inline": False
            },
            {
                "name": "AI Reasoning",
                "value": decision.get('reason', 'N/A'),
                "inline": False
            },
            {
                "name": "Account Stats",
                "value": f"Total P&L: ${total_pnl:.2f}\nTrades: {num_trades}\nWin Rate: {win_rate:.1f}%",
                "inline": False
            }
        ],
        "footer": {
            "text": f"Confidence: {decision.get('confidence', 0):.0%} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        }
    }
    
    return embed

def main():
    print("[Hermes] Starting daily futures trade cycle...")
    
    # Load state
    state = load_state()
    current_day_index = state['current_day_index']
    
    print(f"\n[Hermes] Processing day {current_day_index + 1}...")
    
    # Fetch all historical data for both markets
    nq_df = fetch_futures_data("NQ", days_back=2190)
    es_df = fetch_futures_data("ES", days_back=2190)
    
    if nq_df is None or es_df is None or len(nq_df) == 0 or len(es_df) == 0:
        print("[Error] Failed to fetch futures data")
        return
    
    # Check if we've reached the end
    if current_day_index >= len(nq_df):
        print("[Hermes] Backtest complete! Resetting to day 1")
        state['current_day_index'] = 0
        current_day_index = 0
    
    # Get current day data for both markets
    nq_row = nq_df.iloc[current_day_index]
    es_row = es_df.iloc[current_day_index]
    
    # Use the index (which is the date) instead of ts_event
    nq_date = nq_df.index[current_day_index] if hasattr(nq_df.index, '__getitem__') else str(nq_row.name)
    es_date = es_df.index[current_day_index] if hasattr(es_df.index, '__getitem__') else str(es_row.name)
    
    nq_data = {
        "date": pd.Timestamp(nq_date).strftime('%Y-%m-%d') if nq_date else "unknown",
        "open": float(nq_row['open']),
        "high": float(nq_row['high']),
        "low": float(nq_row['low']),
        "close": float(nq_row['close']),
        "volume": int(nq_row['volume'])
    }
    
    es_data = {
        "date": pd.Timestamp(es_date).strftime('%Y-%m-%d') if es_date else "unknown",
        "open": float(es_row['open']),
        "high": float(es_row['high']),
        "low": float(es_row['low']),
        "close": float(es_row['close']),
        "volume": int(es_row['volume'])
    }
    
    # Get historical context (last 10 days)
    start_idx = max(0, current_day_index - 10)
    
    nq_context = []
    for i in range(start_idx, current_day_index):
        hist_row = nq_df.iloc[i]
        hist_date = nq_df.index[i] if hasattr(nq_df.index, '__getitem__') else str(hist_row.name)
        nq_context.append({
            "date": pd.Timestamp(hist_date).strftime('%Y-%m-%d') if hist_date else "unknown",
            "open": float(hist_row['open']),
            "high": float(hist_row['high']),
            "low": float(hist_row['low']),
            "close": float(hist_row['close']),
            "volume": int(hist_row['volume'])
        })
    
    es_context = []
    for i in range(start_idx, current_day_index):
        hist_row = es_df.iloc[i]
        hist_date = es_df.index[i] if hasattr(es_df.index, '__getitem__') else str(hist_row.name)
        es_context.append({
            "date": pd.Timestamp(hist_date).strftime('%Y-%m-%d') if hist_date else "unknown",
            "open": float(hist_row['open']),
            "high": float(hist_row['high']),
            "low": float(hist_row['low']),
            "close": float(hist_row['close']),
            "volume": int(hist_row['volume'])
        })
    
    # Get AI decision (picks best market and action)
    decision = get_ai_trade_decision(nq_data, es_data, nq_context, es_context)
    
    # Execute trade on the selected market
    pnl = 0
    portfolio = state['portfolio']
    market = decision.get('market', 'NONE')
    action = decision.get('action', 'HOLD')
    
    # Get the price of the selected market
    selected_price = nq_data['close'] if market == "NQ" else es_data['close'] if market == "ES" else 0
    
    if action == 'BUY' and portfolio['position'] is None and market != 'NONE':
        portfolio['position'] = 'LONG'
        portfolio['market'] = market
        portfolio['entry_price'] = selected_price
        print(f"[Trade] BUY {market} @ {selected_price:.2f}")
    
    elif action == 'SELL' and portfolio['position'] == 'LONG' and portfolio['market'] == market:
        pnl = selected_price - portfolio['entry_price']
        portfolio['position'] = None
        portfolio['market'] = None
        state['daily_pnl'].append(pnl)
        print(f"[Trade] SELL {market} @ {selected_price:.2f} | P&L: ${pnl:.2f}")
    
    else:
        print(f"[Trade] HOLD")
    
    # Post to Discord
    print(f"[Discord] Posting trade...")
    embed = format_discord_embed(nq_data, es_data, decision, pnl, state)
    post_to_discord(embed)
    
    # Store trade
    state['trades'].append({
        "date": nq_data['date'],
        "market": market,
        "action": action,
        "price": selected_price,
        "pnl": pnl
    })
    
    # Increment day counter
    state['current_day_index'] += 1
    save_state(state)
    
    print(f"\n[Hermes] Day {current_day_index + 1} complete. Next run in 20 minutes.")

if __name__ == "__main__":
    main()

