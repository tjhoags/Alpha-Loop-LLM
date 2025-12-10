"""
ALC Options Logic
================
Advanced institutional-grade options analytics.
Focus: Volatility Surface, Greeks, Put-Call Parity, Arbitrage
"""

import math
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

class OptionsAnalytics:
    """
    Institutional Options Analytics.
    "Better than the best."
    """
    
    def __init__(self):
        self.risk_free_rate = 0.045  # Approx 4.5% - should be dynamic from FRED
        
    def calculate_greeks(
        self, 
        S: float, # Spot price
        K: float, # Strike price
        T: float, # Time to maturity (years)
        r: float, # Risk-free rate
        sigma: float, # Volatility
        option_type: str = 'call'
    ) -> Dict[str, float]:
        """
        Calculate Black-Scholes Greeks with high precision.
        """
        if T <= 0 or sigma <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
            
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        n_d1 = norm.pdf(d1)
        
        greeks = {}
        
        if option_type.lower() == 'call':
            greeks['price'] = S * N_d1 - K * math.exp(-r * T) * N_d2
            greeks['delta'] = N_d1
            greeks['rho'] = K * T * math.exp(-r * T) * N_d2
            greeks['theta'] = (-S * n_d1 * sigma / (2 * math.sqrt(T)) 
                             - r * K * math.exp(-r * T) * N_d2) / 365.0
        else: # put
            N_minus_d1 = norm.cdf(-d1)
            N_minus_d2 = norm.cdf(-d2)
            greeks['price'] = K * math.exp(-r * T) * N_minus_d2 - S * N_minus_d1
            greeks['delta'] = N_d1 - 1
            greeks['rho'] = -K * T * math.exp(-r * T) * N_minus_d2
            greeks['theta'] = (-S * n_d1 * sigma / (2 * math.sqrt(T)) 
                             + r * K * math.exp(-r * T) * N_minus_d2) / 365.0
                             
        greeks['gamma'] = n_d1 / (S * sigma * math.sqrt(T))
        greeks['vega'] = S * math.sqrt(T) * n_d1 / 100.0 # Per 1% vol change
        
        return greeks

    def check_put_call_parity(
        self,
        call_price: float,
        put_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        dividends: float = 0.0
    ) -> Dict[str, Any]:
        """
        Check for Put-Call Parity arbitrage opportunities.
        C + PV(K) = P + S
        """
        lhs = call_price + K * math.exp(-r * T)
        rhs = put_price + S - dividends
        
        diff = lhs - rhs
        
        # Arbitrage threshold (transaction costs, bid-ask spread)
        threshold = 0.05 
        
        is_arb = abs(diff) > threshold
        
        return {
            'is_arbitrage': is_arb,
            'deviation': diff,
            'strategy': 'Long Call + Short Put + Short Stock' if diff < 0 else 'Short Call + Long Put + Long Stock',
            'profit_potential': abs(diff)
        }

    def implied_volatility(
        self,
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = 'call'
    ) -> float:
        """
        Calculate Implied Volatility using Newton-Raphson.
        """
        MAX_ITER = 100
        PRECISION = 1.0e-5
        sigma = 0.5 # Initial guess
        
        for i in range(0, MAX_ITER):
            greeks = self.calculate_greeks(S, K, T, r, sigma, option_type)
            price_est = greeks['price']
            vega = greeks['vega'] * 100 # Adjust back
            
            diff = price - price_est
            
            if abs(diff) < PRECISION:
                return sigma
                
            if abs(vega) < 1.0e-5: # Avoid division by zero
                break
                
            sigma = sigma + diff / vega
            
        return sigma # Return best guess if not converged

    def kelly_criterion_options(
        self,
        prob_win: float,
        win_loss_ratio: float
    ) -> float:
        """
        Kelly Criterion for optimal position sizing.
        f* = (bp - q) / b
        """
        if win_loss_ratio <= 0:
            return 0.0
        return max(0.0, (win_loss_ratio * prob_win - (1 - prob_win)) / win_loss_ratio)

