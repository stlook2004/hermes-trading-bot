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
        print(f"[Databento] Retrieved {len(df)} bars for {symbol}")
        return df
    except Exception as e:
        print(f"[Error] Failed to fetch data: {e}")
        return None

def run_backtest(symbol: str, df) -> dict:
    """Run backtest on historical data"""
    if df is None or len(df) == 0:
        return {"error": f"No data for {symbol}"}
    
    # Simple momentum strategy
    df['returns'] = df['close'].pct_change()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    
    # Generate signals
    df['signal'] = 0
    df.loc[df['sma_20'] > df['sma_50'], 'signal'] = 1  # Buy
    df.loc[df['sma_20'] < df['sma_50'], 'signal'] = -1  # Sell
    
    # Calculate strategy returns
    df['strategy_returns'] = df['signal'].shift(1) * df['returns']
    
    # Metrics
    total_return = (1 + df['strategy_returns']).prod() - 1
    sharpe_ratio = df['strategy_returns'].mean() / df['strategy_returns'].std() * (252 ** 0.5)
    max_drawdown = (df['strategy_returns'].cumsum().expanding().max() - df['strategy_returns'].cumsum()).max()
    win_rate = (df['strategy_returns'] > 0).sum() / len(df[df['strategy_returns'] != 0]) * 100 if (df['strategy_returns'] != 0).any() else 0
    
    return {
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        "total_return": round(total_return * 100, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "max_drawdown": round(max_drawdown * 100, 2),
        "win_rate": round(win_rate, 2),
        "trades": int((df['signal'].diff() != 0).sum()),
        "bars_analyzed": len(df)
    }

def improve_strategy(results: list, metrics: dict) -> dict:
    """Use Claude to suggest strategy improvements"""
    prompt = f"""You are a quantitative trading strategist. Analyze these 6-year backtest results and suggest improvements:

Results (last 3 backtests):
{json.dumps(results[-3:], indent=2)}

Current Metrics:
{json.dumps(metrics, indent=2)}

Provide 2-3 specific, actionable improvements to the trading strategy. Focus on:
1. Entry/exit signal adjustments
2. Risk management tweaks
3. Market regime detection

Format as JSON: {{"improvements": ["improvement1", "improvement2", ...], "reasoning": "..."}}"""

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    try:
        return json.loads(message.content[0].text)
    except:
        return {
            "improvements": [
                "Increase SMA periods for longer-term trends",
                "Add volatility filter to reduce whipsaws"
            ],
            "reasoning": "Conservative adjustments based on 6-year data"
        }

def main():
    print("[Hermes] Starting 6-year backtest cycle...")
    
    symbols = ["AAPL", "MSFT", "NVDA"]
    results = []
    
    for symbol in symbols:
        print(f"\n[{symbol}] Fetching 6 years of data...")
        df = fetch_historical_data(symbol, days_back=2190)
        
        if df is not None:
            result = run_backtest(symbol, df)
            results.append(result)
            print(f"[{symbol}] Return: {result['total_return']}% | Sharpe: {result['sharpe_ratio']} | Win Rate: {result['win_rate']}%")
    
    # Calculate aggregate metrics
    if results:
        avg_metrics = {
            "avg_return": round(sum(r['total_return'] for r in results) / len(results), 2),
            "avg_sharpe": round(sum(r['sharpe_ratio'] for r in results) / len(results), 2),
            "avg_win_rate": round(sum(r['win_rate'] for r in results) / len(results), 2)
        }
        
        print("\n[Claude] Analyzing strategy improvements...")
        improvements = improve_strategy(results, avg_metrics)
        
        print("Suggested Improvements:")
        for imp in improvements.get("improvements", []):
            print(f"  • {imp}")
        print(f"Reasoning: {improvements.get('reasoning', '')}")
    
    print("\n[Hermes] Backtest cycle complete.")

if __name__ == "__main__":
    main()

