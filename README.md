# Trading Strategies Backtesting Engine

A Python-based quantitative research framework for backtesting, evaluating, and optimizing systematic trading strategies across single-asset and multi-asset portfolios.
https://trading-strategies-backtesting-engine.streamlit.app/


The project combines **Backtrader**, **Optuna**, and **Streamlit** to provide an interactive environment for strategy development, risk analysis, hyperparameter optimization, and out-of-sample evaluation.

## Features

### Trading Strategies

The framework currently supports:

- Single-Asset Trend Following
- Multi-Asset Trend Following
- Single-Asset Mean Reversion
- Multi-Asset Mean Reversion
- Pairs Trading

The strategies include risk-management mechanisms such as:

- ATR-based position sizing
- ATR stop-losses and trailing stops
- Maximum position exposure
- Maximum portfolio exposure
- Maximum number of simultaneous positions
- Portfolio drawdown limits
- Volatility filters
- Time-based exits
- Long and short positions

### Backtesting Engine

The Backtrader-based engine supports:

- Historical OHLCV data
- Multiple assets
- Configurable initial capital
- Transaction commissions
- Slippage assumptions
- Portfolio-level performance tracking
- Trade-level statistics

### Performance Analytics

The application calculates and visualizes metrics including:

- Total Return
- CAGR
- Sharpe Ratio
- Sortino Ratio
- Annualized Volatility
- Maximum Drawdown
- Calmar Ratio
- Profit Factor
- Win Rate
- Net P&L
- Average Trade
- Average Win / Loss
- Trade Expectancy
- Number of Winning and Losing Trades

It also provides:

- Equity Curve
- Drawdown Curve
- Annual Returns
- Trade Statistics

## Hyperparameter Optimization with Optuna

The framework includes an Optuna-based optimization engine for all implemented strategies.

Instead of selecting parameters solely on the basis of a single in-sample backtest, the optimization pipeline is designed to reduce overfitting through:

- Chronological walk-forward validation
- Multiple validation folds
- Final out-of-sample holdout period
- Indicator warm-up periods
- Embargo periods
- Minimum trade-count constraints
- Drawdown constraints
- Stability penalties across validation periods
- Worst-fold performance penalties
- Optuna TPE sampling
- Trial pruning

The optimization objective combines several dimensions of strategy performance rather than maximizing raw return alone.

Hyperparameter importance and individual Optuna trial results can also be analyzed directly from the application.

## Pairs Trading

The pairs trading module uses a dynamic hedge ratio and statistical spread modeling.

The strategy includes:

- Log-price transformation
- Dynamic hedge-ratio estimation
- Spread construction
- Rolling Z-score
- Mean-reversion entry and exit rules
- Hedge-ratio constraints
- Cointegration filtering
- Position and portfolio risk controls

## Technology Stack

- Python
- Backtrader
- Optuna
- Streamlit
- Pandas
- NumPy
- Statsmodels
- yfinance

## Project Architecture

The application separates:

- Trading strategy logic
- Market data handling
- Backtesting
- Risk management
- Performance analytics
- Hyperparameter optimization
- Streamlit user interface

This makes it possible to extend the framework with additional strategies, optimization objectives, risk models, and data sources.

## Purpose

The project was developed as a quantitative finance and software engineering portfolio project demonstrating practical implementation of:

- Systematic trading strategies
- Portfolio risk management
- Statistical arbitrage
- Backtesting methodology
- Hyperparameter optimization
- Out-of-sample validation
- Financial performance analysis
- Interactive financial applications

## Disclaimer

This project is intended for educational, research, and demonstration purposes only. Historical backtest results do not guarantee future performance and should not be interpreted as investment advice.
