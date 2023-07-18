import streamlit as st
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yahoo_fin.stock_info as si
import plotly.express as px
from plotly.subplots import make_subplots


st.title("Testing Environment for Custom Momentum Function")

ticker = st.text_input("Enter ticker", value='GOOG')
STR_FORMATER = "%d/%m/%Y"
start_date = st.text_input("Enter start date in dd/mm/yyyy format", value="01/01/2020")
end_date = st.text_input("Enter end date in dd/mm/yyyy format", value="01/01/2023")


portfolio_close_prices = [] # Currently just one stock but for future
df = si.get_data(ticker, start_date=start_date, end_date=end_date)
df = df.dropna()
close_prices = df['close'].to_list()
portfolio_close_prices.append(close_prices)
portfolio_close_prices = np.array(portfolio_close_prices)

DAYS = len(df)
days = np.array([i+1 for i in range(DAYS)])

portfolio_close_prices_df = pd.DataFrame(portfolio_close_prices.T)
portfolio_pct_change = portfolio_close_prices_df.pct_change()
portfolio_pct_change.iloc[0] = 0
portfolio_pct_change = np.cumsum(portfolio_pct_change) * 100

# Calculating the linear regression line (slopes and the R-square)
slopes = []
r_squares = []
reg = LinearRegression().fit(days.reshape(DAYS, 1), np.array(
    portfolio_pct_change).reshape(DAYS, 1))
y_pred = reg.predict(days.reshape(-1, 1))
r_squares.append(r2_score(portfolio_pct_change, y_pred))
slopes.append(reg.coef_[0][0])


momentums = [slopes[i] * r_squares[i] for i in range(len(slopes))]
absolute_momentums = [abs(i) for i in momentums]

fig = px.line(portfolio_pct_change, title=f'Cumulative Percent change of {ticker}', labels={
    "index": "Number of Days", 
    "value": "Cumulative Percent Change", 
    "variable": "Legend"
})

st.plotly_chart(fig, use_container_width=True)

st.text(f"Momentum: {momentums[0]}")
st.text(f"Slope: {slopes[0]}")
st.text(f"R square: {r_squares[0]}")
