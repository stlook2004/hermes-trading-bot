import anthropic
import databento as db
import json
import os
import sys
from datetime import datetime, timedelta
import pandas as pd

# Initialize clients
databento_key = os.getenv("DATABENTO_API_KEY")
client = anthropic.Anthropic()

def fetch_historical_data(symbol: str, days_back: int = 2190):
    """Fetch 6 years of historical data from Databento"""
    try:
        hist_client = db.Historical(databento_key)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        print(f"[Databento] Fetching {symbol} from {start_date.date()} to {end_date.date()}")
        
        # Get OHLCV data (daily bars)
        data = hist_client.timeseries.get_range(
            dataset="XNAS.ITCH",  # NASDAQ data
            symbols=symbol,
            stype_in="parent",
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            schema="ohlcv-1d"  # Daily OHLCV
        )
        
        # Convert to DataFrame for easier processing
        df = data.to_df()
        df = df.sort_values('ts_event').reset_index(drop=True)
        print(f"[Databento] Retrieved {len(df)} bars for {symbol}")
        return df
    except Exception as e:
        print(f"[Error] Failed to fetch data: {e}")
        return None

def get_ai_trade_decision(symbol: str, current_day_data: dict, historical_context: list, day_num: int, total_days: int) -> dict:
    """Use Claude to analyze current day and make a trade decision"""
    
    context_str = ""
    if len(historical_context) > 0:
        # Show last 10 days of context
        recent_days = historical_context[-10:]
        context_str = "\n".join([
            f"Day {d['day_num']}: O=${d['open']:.2f} H=${d['high']:.2f} L=${d['low']:.2f} C=${d['close']:.2f}"
            for d in recent_days
        ])
    
    prompt = f"""You are a day trader analyzing {symbol}. Make a quick decision.

Day {day_num}/{total_days} ({current_day_data['date']})
Today: O=${current_day_data['open']:.2f} H=${current_day_data['high']:.2f} L=${current_day_data['low']:.2f} C=${current_day_data['close']:.2f} V={current_day_data['volume']}

Recent: {context_str if context_str else "No prior data"}

Respond ONLY with JSON:
{{"action": "BUY"|"SELL"|"HOLD", "confidence": 0.0-1.0, "reason": "brief"}}"""

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

def run_daily_backtest(symbol: str, df):
    """Run day-by-day backtest with AI making decisions"""
    if df is None or len(df) < 2:
        return {"error": f"Insufficient data for {symbol}"}
    
    results = []
    historical_context = []
    portfolio = {"position": None, "entry_price": 0}
    daily_pnl = []
    
    print(f"\n[{symbol}] Starting day-by-day backtest ({len(df)} days)...")
    
    for idx, row in df.iterrows():
        day_num = idx + 1
        
        # Prepare current day data
        current_day = {
            "day_num": day_num,
            "date": pd.Timestamp(row['ts_event']).strftime('%Y-%m-%d'),
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "close": float(row['close']),
            "volume": int(row['volume'])
        }
        
        # Get AI decision
        decision = get_ai_trade_decision(symbol, current_day, historical_context, day_num, len(df))
        
        # Execute trade
        pnl = 0
        action_result = "HOLD"
        
        if decision.get('action') == 'BUY' and portfolio['position'] is None:
            portfolio['position'] = 'LONG'
            portfolio['entry_price'] = current_day['close']
            action_result = f"BUY @ ${current_day['close']:.2f}"
        
        elif decision.get('action') == 'SELL' and portfolio['position'] == 'LONG':
            pnl = current_day['close'] - portfolio['entry_price']
            portfolio['position'] = None
            action_result = f"SELL @ ${current_day['close']:.2f} | P&L: ${pnl:.2f}"
            daily_pnl.append(pnl)
        
        # Store result
        result = {
            "day": day_num,
            "date": current_day['date'],
            "action": decision.get('action', 'HOLD'),
            "price": current_day['close'],
            "pnl": pnl,
            "result": action_result
        }
        results.append(result)
        
        # Add to historical context
        historical_context.append(current_day)
        
        # Print progress every 100 days
        if day_num % 100 == 0 or day_num == len(df):
            total_pnl = sum(daily_pnl)
            print(f"[{symbol}] Day {day_num}/{len(df)} | P&L: ${total_pnl:.2f} | Trades: {len(daily_pnl)}")
    
    # Calculate final metrics
    total_pnl = sum(daily_pnl)
    num_trades = len(daily_pnl)
    winning_trades = len([p for p in daily_pnl if p > 0])
    win_rate = (winning_trades / num_trades * 100) if num_trades > 0 else 0
    
    return {
        "symbol": symbol,
        "total_days": len(df),
        "total_pnl": round(total_pnl, 2),
        "num_trades": num_trades,
        "winning_trades": winning_trades,
        "win_rate": round(win_rate, 2),
        "avg_trade": round(total_pnl / num_trades, 2) if num_trades > 0 else 0,
        "daily_results": results[-20:]  # Last 20 days
    }

def main():
    print("[Hermes] Starting day-by-day backtest cycle...")
    
    symbols = ["AAPL", "MSFT", "NVDA"]
    all_results = []
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"[{symbol}] Fetching 6 years of data...")
        df = fetch_historical_data(symbol, days_back=2190)
        
        if df is not None:
            result = run_daily_backtest(symbol, df)
            all_results.append(result)
            
            print(f"\n[{symbol}] BACKTEST COMPLETE")
            print(f"  Total P&L: ${result['total_pnl']}")
            print(f"  Trades: {result['num_trades']}")
            print(f"  Win Rate: {result['win_rate']}%")
            print(f"  Avg Trade: ${result['avg_trade']}")
            
            # Print last 20 days
            print(f"\n  Last 20 Days:")
            for daily in result['daily_results']:
                print(f"    {daily['date']} | {daily['action']:4s} @ ${daily['price']:7.2f} | {daily['result']}")
    
    # Summary
    if all_results:
        print(f"\n{'='*60}")
        print("[Hermes] SUMMARY")
        total_pnl_all = sum(r['total_pnl'] for r in all_results)
        total_trades_all = sum(r['num_trades'] for r in all_results)
        print(f"  Combined P&L: ${total_pnl_all:.2f}")
        print(f"  Total Trades: {total_trades_all}")
        print(f"  Symbols Analyzed: {len(all_results)}")
    
    print(f"\n[Hermes] Backtest cycle complete.")

if __name__ == "__main__":
    main()

