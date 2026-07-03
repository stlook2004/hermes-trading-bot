# Hermes Trading Bot

A self-improving day trading bot that uses Databento for historical backtesting and Claude AI for strategy optimization.

## Features

- **6-year historical backtesting** via Databento API
- **SMA crossover strategy** (20/50 day moving averages)
- **Claude-powered strategy improvement** - AI analyzes results and suggests tweaks
- **Automated cron execution** - runs every 20 minutes
- **Postgres + Redis integration** - stores results and caches data

## Setup

### Prerequisites

- Python 3.11+
- Databento API key
- Anthropic API key (for Claude)
- Railway account

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABENTO_API_KEY="your_key_here"
export ANTHROPIC_API_KEY="your_key_here"

# Run backtest
python bot.py
```

### Deploy to Railway

1. Push this repo to GitHub
2. Connect to Railway
3. Set environment variables:
   - `DATABENTO_API_KEY`
   - `ANTHROPIC_API_KEY`
   - `DATABASE_URL` (auto-set by Postgres service)
   - `REDIS_URL` (auto-set by Redis service)
4. Deploy with cron schedule: `*/20 * * * *`

## Strategy

Current strategy: **SMA Crossover**
- Buy when 20-day SMA > 50-day SMA
- Sell when 20-day SMA < 50-day SMA
- Backtests on AAPL, MSFT, NVDA

Claude analyzes results and suggests improvements every cycle.

## Metrics

- **Total Return**: Cumulative strategy return over 6 years
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: % of profitable trades

## Next Steps

- Add more symbols
- Implement risk management (stop-loss, position sizing)
- Add market regime detection
- Store results in Postgres for historical tracking
- Build dashboard to visualize backtest results

