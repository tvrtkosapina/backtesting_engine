import numpy as np
import pandas as pd
from portfolio import Stock, Portfolio, CASH

# BACKTESTING ENGINE
# raw data (Date, Open, High, Low, Close, Volume for each security) -> aligned data -> strategy -> Stock -> Portfolio
#
# A strategy is any function with this signature:
#
#     def my_strategy(date, data, positions, cash, **params) -> dict[str, float] | None
#
#     date      : current date (pd.Timestamp)
#     data      : dictionary name -> DataFrame (Open, High, Low, Close, Volume) with rows ONLY up to and including date
#     positions : dictionary name -> number of shares currently held (+ long, - short)
#     cash      : current cash
#     returns   : dictionary name -> number of shares we WANT to hold. Names that are left out stay unchanged, None means no change.
#
# Orders are executed on the NEXT trading day (at Open by default), so the strategy can never use future data.
# Negative cash means borrowing (margin); short selling adds money to cash. Margin interest is not modelled.

OHLCV = ["Open", "High", "Low", "Close", "Volume"]
PRICES = ["Open", "High", "Low", "Close"]

# LOADING DATA

def standardise_ohlcv(df: pd.DataFrame, rename: dict | None = None) -> pd.DataFrame:
    # Returns DataFrame with columns Date, Open, High, Low, Close, Volume sorted by date
    # rename: dictionary for sources with other column names, eg. Zagrebačka burza {"Datum": "Date", "Prva": "Open", ...}
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex): # eg. Yahoo Finance (Price, Ticker)
        df.columns = df.columns.get_level_values(0)
    if rename is not None:
        df = df.rename(columns=rename)
    if "Date" not in df.columns and "date" not in df.columns: # date may be in the index
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "Date"})
    df.columns = [str(column).strip().title() for column in df.columns] # "close", "CLOSE" -> "Close"

    if "Volume" not in df.columns: # some sources (eg. indices) have no volume
        df["Volume"] = 0
    missing = [column for column in ["Date"] + OHLCV if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Use rename to map your column names.")

    df["Date"] = pd.to_datetime(df["Date"])
    if df["Date"].dt.tz is not None:
        df["Date"] = df["Date"].dt.tz_localize(None)
    df["Date"] = df["Date"].dt.normalize() # only the day matters
    for column in OHLCV:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df[["Date"] + OHLCV].sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)

def load_csv(path, rename: dict | None = None, **read_csv_kwargs) -> pd.DataFrame:
    # eg. load_csv("HT.csv", rename={"Datum": "Date"}, sep=";", decimal=",") for Zagrebačka burza
    return standardise_ohlcv(pd.read_csv(path, **read_csv_kwargs), rename)

def load_yahoo(tickers, start = None, end = None) -> dict[str, pd.DataFrame]:
    # Downloads daily data from Yahoo Finance. Returns dictionary ticker -> DataFrame (Date, Open, High, Low, Close, Volume)
    import yfinance as yf # imported here so the rest of the engine works without yfinance
    if isinstance(tickers, str):
        tickers = [tickers]
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False) # auto_adjust: prices adjusted for splits and dividends
        if df is None or df.empty:
            raise ValueError(f"No data downloaded for {ticker}.")
        data[ticker] = standardise_ohlcv(df)
    return data

def align_data(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    # All securities get the same dates (union of all trading days), indexed by Date.
    # On days a security didn't trade, the last Close is carried forward (Open = High = Low = Close, Volume = 0).
    # Dates before the LAST security started trading are dropped, so every security has a price on every date.
    if not data:
        raise ValueError("No data provided.")
    if CASH in data:
        raise ValueError(f"'{CASH}' is reserved for the cash position.")

    frames = {}
    for name, df in data.items():
        df = standardise_ohlcv(df).set_index("Date")
        df[PRICES] = df[PRICES].where(df[PRICES] > 0) # zero or negative prices are data errors
        frames[name] = df

    dates = pd.DatetimeIndex(sorted(set().union(*[df.index for df in frames.values()])), name="Date")

    aligned = {}
    for name, df in frames.items():
        df = df.reindex(dates)
        df["Close"] = df["Close"].ffill()
        for column in ["Open", "High", "Low"]:
            df[column] = df[column].fillna(df["Close"])
        df["Volume"] = df["Volume"].fillna(0)
        if df["Close"].first_valid_index() is None:
            raise ValueError(f"{name} has no valid prices.")
        aligned[name] = df

    first_date = max(df["Close"].first_valid_index() for df in aligned.values())
    return {name: df.loc[first_date:] for name, df in aligned.items()}

# ENGINE

class Backtest():
    """
    Event-driven backtesting engine.
    Parameters:
    data : dict[str, pd.DataFrame]
        name -> DataFrame with Date, Open, High, Low, Close, Volume (raw, they get aligned here)
    strategy : function (see the signature at the top of the file)
    initial_capital : float
    transaction_cost : float
        proportional cost per trade (0.001 = 0.1% of traded value)
    execution_price : "Open" or "Close"
        price of the NEXT day at which orders are executed
    """

    def __init__(self, data, strategy, initial_capital = 10_000, transaction_cost = 0.001, main_currency = "USD", execution_price = "Open"):
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive.")
        if transaction_cost < 0:
            raise ValueError("Transaction cost cannot be negative.")
        if execution_price not in ("Open", "Close"):
            raise ValueError("execution_price must be 'Open' or 'Close'.")

        self.data = align_data(data)
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.main_currency = main_currency
        self.execution_price = execution_price

        self.portfolio = None
        self.trades = None

    def run(self, start_date = None, end_date = None, **strategy_params) -> Portfolio:
        # start_date: first day of trading. Data BEFORE start_date is still visible to the strategy (warm-up for indicators)
        # end_date: last day of the backtest
        # strategy_params: passed to the strategy function (eg. lookback=20)
        data = self.data if end_date is None else {name: df.loc[:pd.Timestamp(end_date)] for name, df in self.data.items()}
        names = list(data.keys())
        dates = data[names[0]].index
        start_index = 0 if start_date is None else dates.searchsorted(pd.Timestamp(start_date))
        if start_index >= len(dates) - 1:
            raise ValueError("Not enough data between start_date and end_date.")

        closes = pd.DataFrame({name: df["Close"] for name, df in data.items()})
        execution_prices = pd.DataFrame({name: df[self.execution_price] for name, df in data.items()})

        positions = {name: 0.0 for name in names}
        cash = float(self.initial_capital)
        orders = None
        shares_history, cash_history, trades = [], [], []

        for i in range(start_index, len(dates)):
            date = dates[i]

            # 1) execute yesterday's orders at today's price
            if orders is not None:
                for name, target in orders.items():
                    if name not in positions:
                        raise ValueError(f"Strategy returned unknown security '{name}'.")
                    if pd.isna(target):
                        raise ValueError(f"Strategy returned NaN for '{name}' on {date.date()}.")
                    quantity = target - positions[name]
                    if quantity != 0:
                        price = execution_prices.at[date, name]
                        trade_value = quantity * price
                        cost = abs(trade_value) * self.transaction_cost
                        cash -= trade_value + cost
                        positions[name] = float(target)
                        trades.append({"date": date, "name": name, "quantity": quantity, "price": price, "value": trade_value, "cost": cost})

            # 2) end of day: record positions (valued at Close)
            shares_history.append(dict(positions))
            cash_history.append(cash)

            # 3) strategy sees data up to today's close and decides what to hold tomorrow
            window = {name: df.iloc[:i + 1] for name, df in data.items()}
            orders = self.strategy(date, window, dict(positions), cash, **strategy_params)
            if orders is not None:
                orders = dict(orders) # works for pd.Series too

        # orders from the last day are never executed (there is no next day)
        index = dates[start_index:]
        shares = pd.DataFrame(shares_history, index=index)
        instruments = []
        for name in names:
            history = pd.DataFrame({"date": index, "price": closes.loc[index, name].values, "num_of_shares": shares[name].values})
            instruments.append(Stock(name, history))
        instruments.append(Stock(CASH, pd.DataFrame({"date": index, "price": 1.0, "num_of_shares": cash_history})))

        self.portfolio = Portfolio(self.main_currency, instruments, index[-1])
        self.trades = pd.DataFrame(trades, columns=["date", "name", "quantity", "price", "value", "cost"])
        return self.portfolio

# HELPERS

def buy_and_hold(date, data, positions, cash, weights = None, cash_buffer = 0.01):
    # Buys once (equal weights if weights is None) and holds until the end. Useful as a benchmark.
    # cash_buffer: part of cash left aside for transaction costs and overnight gaps
    if any(quantity != 0 for quantity in positions.values()):
        return None
    if weights is None:
        weights = {name: 1 / len(data) for name in data}
    investable = cash * (1 - cash_buffer)
    return {name: np.floor(weight * investable / data[name]["Close"].iloc[-1]) for name, weight in weights.items()}

def summary(portfolio: Portfolio, MAR = 0.0) -> pd.Series:
    # Main performance metrics of a portfolio (MAR is daily)
    return pd.Series({
        "Total return": portfolio.total_returns(),
        "Annualised return": portfolio.annualized_returns(),
        "CAGR": portfolio.compound_annual_growth_rate(),
        "Annualised volatility": portfolio.volatility() * np.sqrt(252),
        "Sharpe ratio": portfolio.sharpe_ratio(),
        "Sortino ratio": portfolio.sortino_ratio(MAR),
        "Maximum drawdown": portfolio.maximum_drawdown(),
        "Calmar ratio": portfolio.calmar_ratio(),
    })
