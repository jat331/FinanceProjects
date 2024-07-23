#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:12:34 2024

@author: jamesturner
"""

import numpy as np
import pandas as pd

import statsmodels.api as sm
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate hypothetical data for factors
dates = pd.date_range(start='2020-01-01', periods=100, freq='M')
n_assets = 5

market_index = np.random.normal(0.01, 0.02, len(dates))
interest_rate = np.random.normal(0.005, 0.01, len(dates))
inflation_rate = np.random.normal(0.002, 0.005, len(dates))

# Combine factors into a dataframe
factor_data = pd.DataFrame({
    'market_index': market_index,
    'interest_rate': interest_rate,
    'inflation_rate': inflation_rate
}, index=dates)

# Generate hypothetical asset returns with some noise
asset_returns = np.random.normal(0.01, 0.02, (len(dates), n_assets))
asset_returns_df = pd.DataFrame(asset_returns, index=dates, columns=[f'asset_{i}' for i in range(n_assets)])

# Factor loading: use linear regression to estimate the sensitivity of asset returns to factors
factor_loadings = {}
for asset in asset_returns_df.columns:
    model = sm.OLS(asset_returns_df[asset], sm.add_constant(factor_data)).fit()
    factor_loadings[asset] = model.params[1:]

factor_loadings_df = pd.DataFrame(factor_loadings).T
print(factor_loadings_df)

# Estimate Returns

# Assume expected values of the factors
expected_factors = pd.Series({
    'market_index': 0.01,
    'interest_rate': 0.005,
    'inflation_rate': 0.002
})

# Calculate expected returns
expected_returns = factor_loadings_df.dot(expected_factors)
print(expected_returns)

# Calculate Covariance Matrix
covariance_matrix = asset_returns_df.cov()
print(covariance_matrix)

# Calculate Portfolio Risk
weights = np.random.random(n_assets)

portfolio_risk = np.sqrt(weights.T @ covariance_matrix @ weights)
print(portfolio_risk)

# Calculate Portfolio Return
portfolio_return = weights @ expected_returns
print(portfolio_return)

# Calculate Sharpe Ratio
risk_free_rate = 0.002
sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
print(sharpe_ratio)

# Simulate Portfolio Returns
n_portfolios = 1000
portfolio_returns = []
portfolio_risks = []
portfolio_sharpe_ratios = []

# Plot
plt.figure(figsize=(12, 8))
plt.plot(expected_returns, label='Expected Returns', marker='o')
plt.plot(asset_returns_df.mean(), label='Actual Average Returns', marker='x')
plt.legend()
plt.title('Expected vs Actual Returns')
plt.xlabel('Assets')
plt.ylabel('Returns')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(dates, market_index, label='Market Index')
plt.plot(dates, interest_rate, label='Interest Rate')
plt.plot(dates, inflation_rate, label='Inflation Rate')
plt.legend()
plt.title('Factors Over Time')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))
plt.scatter(expected_returns, asset_returns_df.mean(), marker='o')
plt.plot(expected_returns, expected_returns, color='red', linestyle='--')  # Line of perfect agreement
plt.title('Expected Returns vs Actual Average Returns')
plt.xlabel('Expected Returns')
plt.ylabel('Actual Average Returns')
plt.grid(True)
plt.show()