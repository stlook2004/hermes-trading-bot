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

def fetch_futures_data(symbol: str, start_date: datetime, end_date: datetime):
    """Fetch futures data for a specific date range"""
    try:
        hist_client = db.Historical(databento_key)
        
        print(f"[Databento] Fetching {symbol} from {start_date.date()} to {end_date.date()}")
        
        data = hist_client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=f"{symbol}.FUT",
            stype_in="parent",
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            schema="ohlcv-1d"
        )
        
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
        nq_context_str = "\n".join([
            f"{d['date']}: O={d['open']:.2f} H={d['high']:.2f} L={d['low']:.2f} C={d['close']:.2f}"
            for d in nq_context
        ])
    
    es_context_str = ""
    if len(es_context) > 0:
        es_context_str = "\n".join([
            f"{d['date']}: O={d['open']:.2f} H={d['high']:.2f} L={d['low']:.2f} C={d['close']:.2f}"
            for d in es_context
        ])
    
    prompt = f"""Analyze both NQ and ES futures for {nq_data['date']} and pick the BEST single trade.

TODAY - NQ: O={nq_data['open']:.2f} H={nq_data['high']:.2f} L={nq_data['low']:.2f} C={nq_data['close']:.2f} V={nq_data['volume']}
TODAY - ES: O={es_data['open']:.2f} H={es_data['high']:.2f} L={es_data['low']:.2f} C={es_data['close']:.2f} V={es_data['volume']}

NQ PRIOR 10 DAYS:
{nq_context_str if nq_context_str else "No prior data"}

ES PRIOR 10 DAYS:
{es_context_str if es_context_str else "No prior data"}

Pick ONE trade: NQ or ES, BUY/SELL/HOLD. Respond ONLY as JSON:
{{"market": "NQ"|"ES", "action": "BUY"|"SELL"|"HOLD", "confidence": 0.0-1.0, "reason": "one sentence"}}"""

    try:
        message = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )
        
        if not message.content or len(message.content) == 0:
            print("[Error] Empty response from Claude")
            return {"market": "NONE", "action": "HOLD", "confidence": 0.5, "reason": "empty response"}
        
        response_text = message.content[0].text
        
        if not response_text:
            print("[Error] No text in Claude response")
            return {"market": "NONE", "action": "HOLD", "confidence": 0.5, "reason": "no text"}
        
        print(f"[Claude] Raw response: {response_text}")
        
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx == -1 or end_idx <= start_idx:
            print(f"[Error] No JSON found in response: {response_text}")
            return {"market": "NONE", "action": "HOLD", "confidence": 0.5, "reason": "no json"}
        
        json_str = response_text[start_idx:end_idx]
        result = json.loads(json_str)
        print(f"[Claude] Parsed decision: {result}")
        return result
        
    except json.JSONDecodeError as e:
        print(f"[Error] JSON parse error: {e}")
        return {"market": "NONE", "action": "HOLD", "confidence": 0.5, "reason": "json error"}
    except Exception as e:
        print(f"[Error] Failed to get AI decision: {e}")
        return {"market": "NONE", "action": "HOLD", "confidence": 0.5, "reason": "api error"}

def load_state():
    """Load current state from persistent volume"""
    state_file = "/data/hermes_state.json"
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
                print(f"[State] Loaded state from {state_file}: current_date={state.get('current_date')}")
                return state
        except Exception as e:
            print(f"[Error] Failed to load state: {e}")
    
    print("[State] No state file found, starting fresh from 2020-07-04")
    return {
        "current_date": "2020-07-04",
        "portfolio": {"position": None, "market": None, "entry_price": 0},
        "daily_pnl": [],
        "trades": []
    }

def save_state(state):
    """Save current state to persistent volume"""
    state_file = "/data/hermes_state.json"
    try:
        with open(state_file, 'w') as f:
            json.dump(state, f)
        print(f"[State] Saved state to {state_file}: current_date={state.get('current_date')}")
    except Exception as e:
        print(f"[Error] Failed to save state: {e}")

def format_discord_embed(nq_data, es_data, decision, pnl, state):
    """Format single trade as Discord embed"""
    
    total_pnl = sum(state['daily_pnl'])
    num_trades = len([p for p in state['daily_pnl'] if p != 0])
    winning_trades = len([p for p in state['daily_pnl'] if p > 0])
    win_rate = (winning_trades / num_trades * 100) if num_trades > 0 else 0
    
    color_map = {"BUY": 0x0099ff, "SELL": 0xff6600, "HOLD": 0x999999}
    action = decision.get('action', 'HOLD')
    market = decision.get('market', 'NONE')
    color = color_map.get(action, 0x999999)
    
    if market == "NQ":
        traded_data = nq_data
    elif market == "ES":
        traded_data = es_data
    else:
        traded_data = None
    
    if action == "BUY" and market != "NONE":
        trade_desc = f"🟢 **BUY {market}** @ {traded_data['close']:.2f}"
    elif action == "SELL" and market != "NONE":
        trade_desc = f"🔴 **SELL {market}** @ {traded_data['close']:.2f}\nP&L: ${pnl:.2f}"
    else:
        trade_desc = f"⚪ **HOLD** - No trade"
    
    market_comp = f"""**NQ**: O={nq_data['open']:.2f} H={nq_data['high']:.2f} L={nq_data['low']:.2f} C={nq_data['close']:.2f}
**ES**: O={es_data['open']:.2f} H={es_data['high']:.2f} L={es_data['low']:.2f} C={es_data['close']:.2f}"""
    
    embed = {
        "title": f"📊 Futures Trading - {nq_data['date']}",
        "color": color,
        "fields": [
            {"name": "Trade Decision", "value": trade_desc, "inline": False},
            {"name": "Market Data", "value": market_comp, "inline": False},
            {"name": "AI Reasoning", "value": decision.get('reason', 'N/A'), "inline": False},
            {"name": "Account Stats", "value": f"Total P&L: ${total_pnl:.2f}\nTrades: {num_trades}\nWin Rate: {win_rate:.1f}%", "inline": False}
        ],
        "footer": {"text": f"Confidence: {decision.get('confidence', 0):.0%}"}
    }
    
    return embed

def main():
    print("[Hermes] Starting daily futures trade cycle...")
    
    state = load_state()
    current_date = datetime.strptime(state['current_date'], '%Y-%m-%d')
    
    print(f"[Hermes] Processing {current_date.strftime('%Y-%m-%d')}...")
    
    # Fetch only the data we need: current day + prior 10 days
    fetch_start = current_date - timedelta(days=10)
    fetch_end = current_date
    
    nq_df = fetch_futures_data("NQ", fetch_start, fetch_end)
    es_df = fetch_futures_data("ES", fetch_start, fetch_end)
    
    if nq_df is None or es_df is None or len(nq_df) == 0 or len(es_df) == 0:
        print("[Error] Failed to fetch futures data")
        return
    
    # Get today's data (last row)
    nq_row = nq_df.iloc[-1]
    es_row = es_df.iloc[-1]
    
    nq_data = {
        "date": current_date.strftime('%Y-%m-%d'),
        "open": float(nq_row['open']),
        "high": float(nq_row['high']),
        "low": float(nq_row['low']),
        "close": float(nq_row['close']),
        "volume": int(nq_row['volume'])
    }
    
    es_data = {
        "date": current_date.strftime('%Y-%m-%d'),
        "open": float(es_row['open']),
        "high": float(es_row['high']),
        "low": float(es_row['low']),
        "close": float(es_row['close']),
        "volume": int(es_row['volume'])
    }
    
    # Get prior 10 days context (all rows except the last one)
    nq_context = []
    for i in range(len(nq_df) - 1):
        row = nq_df.iloc[i]
        nq_context.append({
            "date": (fetch_start + timedelta(days=i)).strftime('%Y-%m-%d'),
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "close": float(row['close']),
            "volume": int(row['volume'])
        })
    
    es_context = []
    for i in range(len(es_df) - 1):
        row = es_df.iloc[i]
        es_context.append({
            "date": (fetch_start + timedelta(days=i)).strftime('%Y-%m-%d'),
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "close": float(row['close']),
            "volume": int(row['volume'])
        })
    
    # Get AI decision
    decision = get_ai_trade_decision(nq_data, es_data, nq_context, es_context)
    
    # Execute trade
    pnl = 0
    portfolio = state['portfolio']
    market = decision.get('market', 'NONE')
    action = decision.get('action', 'HOLD')
    
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
    
    # Move to next day
    next_date = current_date + timedelta(days=1)
    state['current_date'] = next_date.strftime('%Y-%m-%d')
    save_state(state)
    
    print(f"[Hermes] {current_date.strftime('%Y-%m-%d')} complete. Next: {next_date.strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    main()

