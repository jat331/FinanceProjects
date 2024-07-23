#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 09:34:45 2024

@author: jamesturner
"""

import numpy as np
import scipy.stats as si

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Black-Scholes option price.
    
    Parameters:
    S: float, Current stock price
    K: float, Strike price
    T: float, Time to expiration in years
    r: float, Risk-free interest rate
    sigma: float, Volatility of the stock
    option_type: str, 'call' or 'put'
    
    Returns:
    float, Option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        option_price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
    
    return option_price

def delta(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return si.norm.cdf(d1)
    elif option_type == 'put':
        return si.norm.cdf(d1) - 1

def gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return si.norm.pdf(d1) / (S * sigma * np.sqrt(T))

def theta(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        theta = (-S * si.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2))
    elif option_type == 'put':
        theta = (-S * si.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * si.norm.cdf(-d2))
    return theta / 365

def vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * si.norm.pdf(d1) * np.sqrt(T) / 100

def rho(S, K, T, r, sigma, option_type='call'):
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return K * T * np.exp(-r * T) * si.norm.cdf(d2) / 100
    elif option_type == 'put':
        return -K * T * np.exp(-r * T) * si.norm.cdf(-d2) / 100

# Example usage
S = 100  # Current stock price
K = 100  # Strike price
T = 1    # Time to expiration in years
r = 0.05 # Risk-free interest rate
sigma = 0.2 # Volatility

call_price = black_scholes(S, K, T, r, sigma, option_type='call')
put_price = black_scholes(S, K, T, r, sigma, option_type='put')

call_delta = delta(S, K, T, r, sigma, option_type='call')
put_delta = delta(S, K, T, r, sigma, option_type='put')

gamma_value = gamma(S, K, T, r, sigma)

call_theta = theta(S, K, T, r, sigma, option_type='call')
put_theta = theta(S, K, T, r, sigma, option_type='put')

vega_value = vega(S, K, T, r, sigma)

call_rho = rho(S, K, T, r, sigma, option_type='call')
put_rho = rho(S, K, T, r, sigma, option_type='put')

print("Call Price: ", call_price)
print("Put Price: ", put_price)
print("Call Delta: ", call_delta)
print("Put Delta: ", put_delta)
print("Gamma: ", gamma_value)
print("Call Theta: ", call_theta)
print("Put Theta: ", put_theta)
print("Vega: ", vega_value)
print("Call Rho: ", call_rho)
print("Put Rho: ", put_rho)