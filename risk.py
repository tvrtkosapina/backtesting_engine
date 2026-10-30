import numpy as np 
import pandas as pd
from datetime import datetime
from scipy.optimize import curve_fit
from scipy import stats
from portfolio import Stock, Portfolio

# IMPORTANT: alpha is significance level; 5% or 1% NOT 99% or 95%
# All VaR / ES values are 1-day, in money, for the CURRENT (last) portfolio value and returned as POSITIVE numbers (losses)

class RiskMetrics(): 
    def __init__(self, p: Portfolio):
        self.p = p

    # RISK MEASURES
    
    def historicVaR(self, alpha):
        return -self.p.returns.quantile(alpha) * self.p.portfolio_value.iloc[-1] 

    def parametricVaR(self, alpha):
        return self.p.volatility() * self.p.portfolio_value.iloc[-1] * stats.norm.isf(alpha) 

    def MonteCarloVaR(self, alpha, num_of_simulations, distribution="normal", degrees_of_freedom=None):
        returns = self.p.returns.dropna()
        portfolio_value = self.p.portfolio_value.iloc[-1]
        mu = returns.mean()
        sigma = returns.std()

        distribution = distribution.lower()
        if distribution == "normal":
            model = lambda: np.random.normal(0, 1)
        elif distribution == "student" and degrees_of_freedom is not None and degrees_of_freedom > 2: 
            model = lambda: np.random.standard_t(degrees_of_freedom) * np.sqrt((degrees_of_freedom - 2) / degrees_of_freedom)
        else:
            raise ValueError("distribution must be 'normal' or 'student' with > 2 degrees_of_freedom.")

        scenario_returns = [
            mu + sigma * model()
            for _ in range(num_of_simulations)
        ]
        scenario_pnl = portfolio_value * np.array(scenario_returns)

        return -np.percentile(scenario_pnl, 100 * alpha)

    # https://riskhub.org/blogs/expected-shortfall-calculation-methods

    def historicES(self, alpha):
        returns = self.p.returns
        return -returns[returns <= returns.quantile(alpha)].mean() * self.p.portfolio_value.iloc[-1] 

    def parametricES(self, alpha):
        return self.p.portfolio_value.iloc[-1] * self.p.volatility() * stats.norm.pdf(stats.norm.isf(alpha)) / alpha # normal ES = V * sigma * pdf(z_alpha) / alpha (same zero-mean assumption as parametricVaR)

    def MonteCarloES(self, alpha, num_of_simulations, distribution="normal", degrees_of_freedom=None):
        returns = self.p.returns.dropna()
        portfolio_value = self.p.portfolio_value.iloc[-1]
        mu = returns.mean()
        sigma = returns.std()

        distribution = distribution.lower()
        if distribution == "normal":
            model = lambda: np.random.normal(0, 1)
        elif distribution == "student" and degrees_of_freedom is not None and degrees_of_freedom > 2: # FIX: condition was inverted
            model = lambda: np.random.standard_t(degrees_of_freedom) * np.sqrt((degrees_of_freedom - 2) / degrees_of_freedom)
        else:
            raise ValueError("distribution must be 'normal' or 'student' with > 2 degrees_of_freedom.")

        scenario_returns = [
            mu + sigma * model()
            for _ in range(num_of_simulations)
        ]
        scenario_pnl = portfolio_value * np.array(scenario_returns)

        return -scenario_pnl[scenario_pnl <= np.percentile(scenario_pnl, 100 * alpha)].mean() # FIX: mean of the scenarios below the VaR percentile

    # RISK ATTRIBUTION

    def risk_contribution(self, timestamp = None): # timestamp added so any point in time can be evaluated
        # covariance matrix
        prices = {}
        for instrument in self.p.instruments: 
            prices[instrument.name] = instrument.history.set_index('date')['price']
        df = pd.DataFrame(prices)
        covariance_matrix = df.pct_change().dropna().cov() # important: covariance of returns, not of prices

        # weights
        weights = self.p.position_weights(timestamp) # dictionary
        weights_series = pd.Series(weights).reindex(covariance_matrix.columns).fillna(0)

        # risk contribution of each instrument
        marginal_risk = covariance_matrix @ weights_series
        portfolio_volatility = np.sqrt(weights_series @ marginal_risk) # volatility implied by these weights, so contributions sum up to it
        risk_contributions = (weights_series * marginal_risk / portfolio_volatility)

        return risk_contributions.sort_values(ascending=False).to_dict() # dictionary sorted from highest to lowest contribution

    def pct_risk_contribution(self, timestamp = None):
        # covariance matrix
        prices = {}
        for instrument in self.p.instruments: 
            prices[instrument.name] = instrument.history.set_index('date')['price']
        df = pd.DataFrame(prices)
        covariance_matrix = df.pct_change().dropna().cov() 

        # weights
        weights = self.p.position_weights(timestamp) # dictionary
        weights_series = pd.Series(weights).reindex(covariance_matrix.columns).fillna(0)

        # risk contribution of each instrument
        marginal_risk = covariance_matrix @ weights_series
        risk_contributions = (weights_series * marginal_risk / (weights_series @ marginal_risk)) # divided by portfolio variance w'Cw, so contributions sum up to 1

        return risk_contributions.sort_values(ascending=False).to_dict() # dictionary sorted from highest to lowest contribution

    def marginal_risk(self, timestamp = None):
        # covariance matrix
        prices = {}
        for instrument in self.p.instruments: 
            prices[instrument.name] = instrument.history.set_index('date')['price']
        df = pd.DataFrame(prices)
        covariance_matrix = df.pct_change().dropna().cov()

        # weights
        weights = self.p.position_weights(timestamp) # dictionary
        weights_series = pd.Series(weights).reindex(covariance_matrix.columns).fillna(0)

        # marginal risk of each instrument (derivative of portfolio volatility with respect to the weight)
        marginal_risk = covariance_matrix @ weights_series / np.sqrt(weights_series @ covariance_matrix @ weights_series)
        return marginal_risk.sort_values(ascending=False).to_dict() # dictionary sorted from highest to lowest contribution

    # STRESS TESTING

    def hypothetical_stress_test(self, D):  # dictionary: instrument name -> percentage change (eg. decrease of 10% is -0.1)
        # Returns a NEW portfolio with one extra day in which the shocks from D happen; positions stay the same
        new_instruments = []
        for instrument in self.p.instruments: 
            history = instrument.history.copy()
            last = history.iloc[-1]
            new_row = {
                "date": last["date"] + pd.offsets.BDay(1), # date + 1 does not work for dates
                "price": last["price"] * (1 + D.get(instrument.name, 0)), # instruments not in D get a row too, so all histories keep the same length
                "num_of_shares": last["num_of_shares"],
            }
            history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
            new_instruments.append(Stock(instrument.name, history))

        return Portfolio(self.p.main_currency, new_instruments, new_instruments[0].history["date"].iloc[-1])

    def historical_stress_test(self, start_date, end_date):
        # Replays the returns from [start_date, end_date] on the CURRENT positions and prices
        start_date, end_date = pd.Timestamp(start_date), pd.Timestamp(end_date)
        new_instruments = []
        for instrument in self.p.instruments: 
            history = instrument.history
            window = history[(history["date"] >= start_date)&(history["date"] <= end_date)].copy()
            if window.empty:
                raise ValueError(f"No data for {instrument.name} between {start_date.date()} and {end_date.date()}.") # skipping would make histories of different lengths
            window["returns"] = window["price"].pct_change().fillna(0)
                
            new_price = history["price"].iloc[-1]
            new_quantity = history["num_of_shares"].iloc[-1]

            window["price"] = new_price * (1 + window["returns"]).cumprod()
            window["num_of_shares"] = new_quantity

            new_instruments.append(Stock(instrument.name, window.drop(columns="returns").reset_index(drop=True))) # original portfolio remains unchanged
        return Portfolio(self.p.main_currency, new_instruments, new_instruments[0].history["date"].iloc[-1])
