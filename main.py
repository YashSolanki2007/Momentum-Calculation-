'''
Imports i.e the external libraries to include in the project sklearn is used to do the 
linear regression. 
Pandas is used to process the data
Numpy is used to do matrix manipulation 
Yahoo finance is used to load the stock information 
'''
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import yahoo_fin.stock_info as si


# The tickers are the list of stocks that are a part of the momentum calculation
tickers = ["GOOG", "AAPL"]

# Number of days to perform the calculation over
DAYS = 60

# Getting the close prices of the portfolio stocks
portfolio_close_prices = []
for i in range(len(tickers)):
    df = si.get_data(tickers[i])
    close_prices = df['close'].to_list()
    close_prices = close_prices[len(close_prices) - DAYS: ]
    portfolio_close_prices.append(close_prices)
portfolio_close_prices = np.array(portfolio_close_prices)

# Converting the x-values into a list of numbers from one to the number of days
days = np.array([i+1 for i in range(DAYS)])


# Calculating the linear regression line (slopes and the R-square)
slopes = []
r_squares = []
for i in range(len(tickers)):
    reg = LinearRegression().fit(days.reshape(
        DAYS, 1), portfolio_close_prices[i].reshape(DAYS, 1))
    y_pred = reg.predict(days.reshape(-1, 1))
    r_squares.append(r2_score(portfolio_close_prices[i], y_pred))
    slopes.append(reg.coef_[0])

momentums = [slopes[i] * r_squares[i] for i in range(len(slopes))]
absolute_momentums = [abs(i) for i in momentums]

print(momentums)
print(absolute_momentums)
