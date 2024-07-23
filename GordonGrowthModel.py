#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:19:16 2024

@author: jamesturner
"""

def gordon_growth_model(D0, g, r):
    """
    Calculate the intrinsic value of a stock using the Gordon Growth Model.
    
    Parameters:
    D0 (float): Most recent dividend paid
    g (float): Growth rate of dividends
    r (float): Required rate of return
    
    Returns:
    float: Intrinsic value of the stock
    """
    if r <= g:
        raise ValueError("The required rate of return must be greater than the growth rate.")
    
    P0 = D0 * (1 + g) / (r - g)
    return P0

# Example usage
D0 = 2.00  # Most recent dividend
g = 0.05   # 5% growth rate
r = 0.10   # 10% required rate of return

intrinsic_value = gordon_growth_model(D0, g, r)
print(f"The intrinsic value of the stock is: ${intrinsic_value:.2f}")