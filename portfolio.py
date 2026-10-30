import numpy as np 
import pandas as pd
import scipy.optimize
from dataclasses import dataclass
from datetime import datetime

CASH = "CASH" # name of the cash instrument (price is always 1, num_of_shares = amount of cash). Without it, money from a sale would disappear from the portfolio and look like a loss

@dataclass
class Stock:
    # Represents a single financial instrument held in a portfolio.
    # Parameters: name (str): Instrument name/ticker
    # history (pd.DataFrame): price and position history

    name: str
    # currency: str = "EUR" In future implementations other currencys would be added
    history: pd.DataFrame | None = None # columns are: date, price(always >= 0), num_of_shares (+ for long buy and minus for shorting/ selling)

    def __post_init__(self):
        if not self.name:
            raise ValueError("Stock name cannot be empty.")

        if self.history is None or self.history.empty: 
            raise ValueError("Stock history cannot be empty.")

        if (self.history["price"] < 0).any(): 
            raise ValueError("Price cannot be negative.") # this makes sure that only one of num_of_shares, price notes the longed/shorted position

class Portfolio:
    """
    Represents a portfolio of financial instruments and their allocations. There are no money coming in or coming out of the portfolio.
    Parameters:
    instruments : list[Stock]
        List of financial instruments held in the portfolio.
    """

    def __init__(
        self,
        main_currency: str,
        instruments: list[Stock], # ALL STOCKS MUST BE OF THE SAME LENGTH!!!
        timestamp: datetime,
    ):
        if not instruments:
            raise ValueError("Portfolio must contain at least one instrument.")

        dates = instruments[0].history["date"].reset_index(drop=True)
        for stock in instruments: 
            if not stock.history["date"].reset_index(drop=True).equals(dates):
                raise ValueError(f"All stocks must have the same dates in history ({stock.name} differs).")

        self.instruments = instruments
        self.main_currency = main_currency
        self.timestamp = timestamp
        
        self.portfolio_value = sum(stock.history.set_index("date")["price"] * stock.history.set_index("date")["num_of_shares"] for stock in instruments) 
        self.returns = self.portfolio_value.pct_change().dropna() 
            
    def _timestamp(self, timestamp = None): 
        return pd.Timestamp(self.timestamp if timestamp is None else timestamp)
        
    def number_of_assets(self, timestamp = None) -> int: # Returns the number of assets in the portfolio 
        timestamp = self._timestamp(timestamp)
        num = 0
        for stock in self.instruments:
            if stock.name == CASH: 
                continue
            rows = stock.history.loc[stock.history["date"] == timestamp, "num_of_shares"] 
            if not rows.empty and rows.iloc[0] != 0:
                num+=1
        return num ### NAPOMENA: DATUM date je prva kolona dataframea; AKO JE DATAFRAME DRUKCIJI NECE RADITI!!!
    
    def value(self, timestamp = None) -> float:
        return self.portfolio_value.loc[self._timestamp(timestamp)] 

    def position_weights(self, timestamp = None) -> dict[str, float]: # Return portfolio weights by instrument name at timestamp.
        timestamp = self._timestamp(timestamp)
        portfolio_value = self.value(timestamp) 
        dictionary = {}
        for stock in self.instruments:
            rows = stock.history[stock.history["date"] == timestamp]
            if not rows.empty:
                dictionary[stock.name] = float(rows["price"].iloc[0] * rows["num_of_shares"].iloc[0] / portfolio_value) # FIX: dict has no append, Series -> float
            else:
                dictionary[stock.name] = 0.0

        return dictionary

    def position_values(self, timestamp = None) -> dict[str, float]: # Returns market value of each position as a dictionary. 
        timestamp = self._timestamp(timestamp)
        dictionary = {}
        for stock in self.instruments:
            rows = stock.history[stock.history["date"] == timestamp]
            if not rows.empty:
                dictionary[stock.name] = float(rows["price"].iloc[0] * rows["num_of_shares"].iloc[0])
            else:
                dictionary[stock.name] = 0.0

        return dictionary

    @property
    def current_value(self) -> tuple[float, str, datetime | None]: # Returns current portfolio value, currency and timestamp
        return (self.value(), self.main_currency, self.timestamp) 

    def total_returns(self) -> float:
        if self.returns is None:
            raise ValueError("Portfolio returns not assigned.")
        return (self.portfolio_value.iloc[-1]-self.portfolio_value.iloc[0])/self.portfolio_value.iloc[0]

    def annualized_returns(self) -> float:
        if self.returns is None:
            raise ValueError("Portfolio returns not assigned.")
        return ((self.returns + 1).prod())**(252/len(self.returns)) - 1
    
    def volatility(self) -> float: # daily volatility
        if self.returns is None:
            raise ValueError("Portfolio returns not assigned.")
        if len(self.returns) < 2:
            raise ValueError("Not enough portfolio returns data (currently < 2).")
        return self.returns.std()
    
    def downside_volatility(self, MAR) -> float: # MAR- minimal accepted return (daily)
        if self.returns is None:
            raise ValueError("Portfolio returns not assigned.")
        if len(self.returns) < 2:
            raise ValueError("Not enough portfolio returns data (currently < 2).")
        return np.sqrt((np.minimum(self.returns - MAR, 0)**2).mean()) 

    def maximum_drawdown(self) -> float:
        if self.returns is None:
            raise ValueError("Portfolio returns not assigned.")
        S = self.portfolio_value 
        return ((S.cummax() - S)/S.cummax()).max() 
    
    def compound_annual_growth_rate(self) -> float: 
        years = (self.portfolio_value.index.max() - self.portfolio_value.index.min()).days / 365.25
        return (1 + self.total_returns())**(1 / years) - 1 
        
    def sharpe_ratio(self) -> float: # annualised, risk free rate = 0
        if self.returns is None:
            raise ValueError("Portfolio returns not assigned.")
        return self.returns.mean()/self.volatility() * np.sqrt(252) 

    def sortino_ratio(self, MAR) -> float: # annualised
        return (self.returns.mean() - MAR) / self.downside_volatility(MAR) * np.sqrt(252) 
        
    def calmar_ratio(self) -> float:
        if self.returns is None:
            raise ValueError("Portfolio returns not assigned.")
        return self.compound_annual_growth_rate()/self.maximum_drawdown() # if max drawdown is 0, numpy returns inf

    def diversification_ratio(self, timestamp = None) -> float: # weighted average of asset volatilities / portfolio volatility
        weights = pd.Series(self.position_weights(timestamp))
        returns = pd.DataFrame({stock.name: stock.history.set_index("date")["price"].pct_change() for stock in self.instruments}).dropna() 
        weights = weights[returns.columns]
        return (returns.std() * weights.abs()).sum() / np.sqrt(weights @ returns.cov() @ weights) 
    
    def herfindahl_index(self, date = None)->float: 
        weights = self.position_weights(date) 
        return sum(weights[stock.name]**2 for stock in self.instruments) 

    # OPTIMISATION
    
    def MeanVarianceOptimisation(self, bounds = None): # minimum variance weights (long only by default)
        # covariance matrix and expected returns
        prices = {}
        for instrument in self.instruments: 
            if instrument.name == CASH: 
                continue
            prices[instrument.name] = instrument.history.set_index('date')['price']
        df = pd.DataFrame(prices)
        returns = df.pct_change().dropna()
        covariance_matrix = returns.cov() * 252 
        expected_returns = returns.mean()

        N = len(df.columns) # number of instruments in the covariance matrix (number_of_assets counts only currently held ones)
        weights = pd.Series(np.ones(N) / N)
        if bounds is None:
            bounds = [(0, 1)] * N 

        # Defining a function we will be minimising 
        def portfolio_variance(weights):
            return weights @ covariance_matrix.values @ weights

        constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}] # Constraints: weights sum to 1

        result = scipy.optimize.minimize(portfolio_variance, weights.values, method='SLSQP', bounds=bounds, constraints=constraints)

        if not result.success:
            raise ValueError(result.message)

        optimal_weights = pd.Series(result.x, index=df.columns)

        return optimal_weights
