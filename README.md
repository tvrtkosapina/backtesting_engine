Backtesting engine

This is an event-driven backtesting engine that turns OHLCV data into portfolios, aligns trading calendars across
exchanges, executes at the next day’s open to avoid look-ahead bias, and models transaction costs and short
selling.

Implemented CAGR, Sharpe, Sortino, Calmar and max drawdown, historical, parametric and Monte Carlo (normal /
Student-t) VaR and Expected Shortfall; risk attribution and hypothetical/historical stress testing are part od the risk.py.

StrategyAnalysis.ipyb containes tested mean-reversion and cointegration-based pairs trading on eight US and European large caps, using in-sample
parameter selection and out-of-sample evaluation against equal-weight and minimum-variance benchmarks.

This repository is still being developed, I hope you enyoj it!
