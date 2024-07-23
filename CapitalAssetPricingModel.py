#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:29:45 2024

@author: jamesturner
"""

#Build a capital asset pricing model (CAPM) using the S&P 500 and a stock of your choice.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
from scipy import stats

# Define the time period
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2020, 1, 1)

# Fetch data for S&P 500 and Apple stock
try:
    snp = yf.download('^GSPC', start=start, end=end)  # S&P 500
    stock = yf.download('AAPL', start=start, end=end)  # Apple stock
except Exception as e:
    print(f"Error fetching data: {e}")
    snp, stock = None, None

if snp is not None and stock is not None:
    # Calculate the daily returns
    snp['daily_return'] = snp['Adj Close'].pct_change()
    stock['daily_return'] = stock['Adj Close'].pct_change()

    # Remove the first row of NaN
    snp = snp.dropna()
    stock = stock.dropna()

    # Calculate the beta
    beta, alpha, r_value, p_value, std_err = stats.linregress(snp['daily_return'], stock['daily_return'])
    print('Beta:', beta)

    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.scatter(snp['daily_return'], stock['daily_return'])
    plt.xlabel('S&P 500 Daily Return')
    plt.ylabel('Stock Daily Return')
    plt.title('CAPM')
    plt.grid()
    plt.show()

    # Calculate the expected return
    rf = 0.02  # risk-free rate
    expected_return = rf + beta * (snp['daily_return'].mean() - rf)
    print('Expected Return:', expected_return)

    # Calculate the Sharpe ratio
    sharpe_ratio = (expected_return - rf) / stock['daily_return'].std()
    print('Sharpe Ratio:', sharpe_ratio)

    # Calculate the Treynor ratio
    treynor_ratio = (expected_return - rf) / beta
    print('Treynor Ratio:', treynor_ratio)

    # Calculate the Jensen's alpha
    jensens_alpha = stock['daily_return'].mean() - (rf + beta * (snp['daily_return'].mean() - rf))
    print("Jensen's Alpha:", jensens_alpha)

    # Calculate the R-squared
    print('R-squared:', r_value ** 2)

    # Calculate the p-value
    print('P-value:', p_value)

    # Calculate the standard error
    print('Standard Error:', std_err)

    # Calculate the 95% confidence interval
    confidence_interval = stats.t.interval(0.95, len(snp['daily_return']) - 2, beta, std_err)
    print('95% Confidence Interval:', confidence_interval)

    # Calculate the 99% confidence interval
    confidence_interval = stats.t.interval(0.99, len(snp['daily_return']) - 2, beta, std_err)
    print('99% Confidence Interval:', confidence_interval)
else:
    print("Data could not be fetched. Please check your internet connection or the data source.")