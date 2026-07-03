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

def fetch_historical_data(symbol: str, days_back: int = 2190):
    """Fetch 6 years of historical data from Databento"""
    try:
        hist_client = db.Historical(databento_key)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        print(f"[Databento] Fetching {symbol} from {start_date.date()} to {end_date.date()}")
        
        # Get OHLCV data (daily bars)
        data = hist_client.timeseries.get_range(
            dataset="XNAS.ITCH",
            symbols=symbol,
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
        print(f"[Error] Failed to fetch data: {e}")
        return None

def get_ai_trade_decision(symbol: str, current_day_data: dict, historical_context: list) -> dict:
    """Use Claude to analyze current day and make a trade decision"""
    
    context_str = ""
    if len(historical_context) > 0:
        # Show last 10 days of context
        recent_days = historical_context[-10:]
        context_str = "\n".join([
            f"{d['date']}: O=${d['open']:.2f} H=${d['high']:.2f} L=${d['low']:.2f} C=${d['close']:.2f}"
            for d in recent_days
        ])
    
    prompt = f"""You are a day trader analyzing {symbol}. 

TODAY ({current_day_data['date']}):
Open: ${current_day_data['open']:.2f}
High: ${current_day_data['high']:.2f}
Low: ${current_day_data['low']:.2f}
Close: ${current_day_data['close']:.2f}
Volume: {current_day_data['volume']}

LAST 10 DAYS:
{context_str if context_str else "No prior data"}

Make a trading decision. Respond ONLY with JSON:
{{"action": "BUY"|"SELL"|"HOLD", "confidence": 0.0-1.0, "reason": "brief explanation"}}"""

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=150,
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
        pass
    
    return {"action": "HOLD", "confidence": 0.5, "reason": "parse error"}

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
        "portfolio": {"position": None, "entry_price": 0},
        "daily_pnl": [],
        "trades": []
    }

def save_state(state):
    """Save current state to file"""
    state_file = "/tmp/hermes_state.json"
    with open(state_file, 'w') as f:
        json.dump(state, f)

def format_discord_embed(symbol, current_day, decision, pnl, state):
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
    color = color_map.get(action, 0x999999)
    
    # Build trade description
    if action == "BUY":
        trade_desc = f"🟢 **BUY** @ ${current_day['close']:.2f}"
    elif action == "SELL":
        trade_desc = f"🔴 **SELL** @ ${current_day['close']:.2f}\nP&L: ${pnl:.2f}"
    else:
        trade_desc = f"⚪ **HOLD** @ ${current_day['close']:.2f}"
    
    embed = {
        "title": f"📈 {symbol} - {current_day['date']}",
        "color": color,
        "fields": [
            {
                "name": "Trade Decision",
                "value": trade_desc,
                "inline": False
            },
            {
                "name": "Daily Data",
                "value": f"O: ${current_day['open']:.2f} | H: ${current_day['high']:.2f} | L: ${current_day['low']:.2f} | C: ${current_day['close']:.2f}\nVolume: {current_day['volume']:,}",
                "inline": False
            },
            {
                "name": "AI Reasoning",
                "value": decision.get('reason', 'N/A'),
                "inline": False
            },
            {
                "name": "Portfolio Stats",
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
    print("[Hermes] Starting daily trade cycle...")
    
    # Load state
    state = load_state()
    current_day_index = state['current_day_index']
    
    symbols = ["AAPL", "MSFT", "NVDA"]
    
    for symbol in symbols:
        print(f"\n[{symbol}] Processing day {current_day_index + 1}...")
        
        # Fetch all historical data
        df = fetch_historical_data(symbol, days_back=2190)
        
        if df is None or len(df) == 0:
            print(f"[{symbol}] No data available")
            continue
        
        # Check if we've reached the end
        if current_day_index >= len(df):
            print(f"[{symbol}] Backtest complete! Resetting to day 1")
            state['current_day_index'] = 0
            current_day_index = 0
        
        # Get current day data
        row = df.iloc[current_day_index]
        current_day = {
            "date": pd.Timestamp(row['ts_event']).strftime('%Y-%m-%d'),
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "close": float(row['close']),
            "volume": int(row['volume'])
        }
        
        # Get historical context (last 10 days)
        start_idx = max(0, current_day_index - 10)
        historical_context = []
        for i in range(start_idx, current_day_index):
            hist_row = df.iloc[i]
            historical_context.append({
                "date": pd.Timestamp(hist_row['ts_event']).strftime('%Y-%m-%d'),
                "open": float(hist_row['open']),
                "high": float(hist_row['high']),
                "low": float(hist_row['low']),
                "close": float(hist_row['close']),
                "volume": int(hist_row['volume'])
            })
        
        # Get AI decision
        decision = get_ai_trade_decision(symbol, current_day, historical_context)
        
        # Execute trade
        pnl = 0
        portfolio = state['portfolio']
        
        if decision.get('action') == 'BUY' and portfolio['position'] is None:
            portfolio['position'] = 'LONG'
            portfolio['entry_price'] = current_day['close']
            print(f"[{symbol}] BUY @ ${current_day['close']:.2f}")
        
        elif decision.get('action') == 'SELL' and portfolio['position'] == 'LONG':
            pnl = current_day['close'] - portfolio['entry_price']
            portfolio['position'] = None
            state['daily_pnl'].append(pnl)
            print(f"[{symbol}] SELL @ ${current_day['close']:.2f} | P&L: ${pnl:.2f}")
        
        else:
            print(f"[{symbol}] HOLD @ ${current_day['close']:.2f}")
        
        # Post to Discord
        print(f"[Discord] Posting {symbol} trade...")
        embed = format_discord_embed(symbol, current_day, decision, pnl, state)
        post_to_discord(embed)
        
        # Store trade
        state['trades'].append({
            "date": current_day['date'],
            "symbol": symbol,
            "action": decision.get('action', 'HOLD'),
            "price": current_day['close'],
            "pnl": pnl
        })
    
    # Increment day counter
    state['current_day_index'] += 1
    save_state(state)
    
    print(f"\n[Hermes] Day {current_day_index + 1} complete. Next run in 20 minutes.")

if __name__ == "__main__":
    main()

