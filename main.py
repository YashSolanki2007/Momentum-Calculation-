'''
Imports i.e the external libraries to include in the project sklearn is used to do the 
linear regression. 
Pandas is used to process the data
Numpy is used to do matrix manipulation 
Yahoo finance is used to load the stock information 
'''
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yahoo_fin.stock_info as si


# The tickers are the list of stocks that are a part of the momentum calculation
tickers = ["SPY", "AAPL"]

# Number of days to perform the calculation over
DAYS = 120

# Getting the close prices of the portfolio stocks
portfolio_close_prices = []
for i in range(len(tickers)):
    df = si.get_data(tickers[i])
    df = df.dropna()
    close_prices = df['close'].to_list()
    close_prices = close_prices[len(close_prices) - DAYS: ]
    portfolio_close_prices.append(close_prices)
portfolio_close_prices = np.array(portfolio_close_prices)

# Converting the x-values into a list of numbers from one to the number of days
days = np.array([i+1 for i in range(DAYS)])

portfolio_close_prices_df = pd.DataFrame(portfolio_close_prices.T)
portfolio_pct_change = portfolio_close_prices_df.pct_change()
portfolio_pct_change.iloc[0] = 0
portfolio_pct_change = np.cumsum(portfolio_pct_change) * 100

# Calculating the linear regression line (slopes and the R-square)
slopes = []
r_squares = []
for i in range(len(tickers)):
    reg = LinearRegression().fit(days.reshape(DAYS, 1), np.array(
        portfolio_pct_change[i]).reshape(DAYS, 1))
    y_pred = reg.predict(days.reshape(-1, 1))
    r_squares.append(r2_score(portfolio_pct_change[i], y_pred))
    slopes.append(reg.coef_[0][0])



momentums = [slopes[i] * r_squares[i] for i in range(len(slopes))]
absolute_momentums = [abs(i) for i in momentums]

print(slopes)
print(r_squares)
