"""
Options Arbitrage Agent
Identifies and exploits options pricing inefficiencies

Strategies:
1. Put-Call Parity Violations
2. Vertical Spread Mispricing
3. Box Spread Arbitrage
4. Conversion/Reversal Arbitrage
5. Volatility Arbitrage (IV vs RV)

Author: Tom Hogan | Alpha Loop Capital, LLC
Date: 2025-12-09
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class OptionsArbitrageOpportunity:
    """Options arbitrage opportunity"""
    strategy_type: str
    symbol: str
    legs: List[Dict]
    expected_profit: float
    risk: float
    probability_of_profit: float
    capital_required: float
    annual_return: float


class OptionsArbitrageAgent:
    """
    Options Arbitrage Agent

    Identifies risk-free or low-risk options arbitrage opportunities
    Focus on liquid mid-cap stocks with tight bid-ask spreads
    """

    def __init__(
        self,
        min_profit_threshold: float = 0.02,  # 2% min profit
        max_capital_per_trade: float = 50000,
        min_open_interest: int = 100,
        max_bid_ask_spread_pct: float = 0.05  # 5% max spread
    ):
        self.min_profit_threshold = min_profit_threshold
        self.max_capital_per_trade = max_capital_per_trade
        self.min_open_interest = min_open_interest
        self.max_bid_ask_spread_pct = max_bid_ask_spread_pct

        logger.info(f"Options Arbitrage Agent initialized")
        logger.info(f"  Min profit: {min_profit_threshold:.1%}")
        logger.info(f"  Max capital per trade: ${max_capital_per_trade:,.0f}")

    def analyze_put_call_parity(
        self,
        symbol: str,
        stock_price: float,
        strike: float,
        expiration_days: int,
        call_bid: float,
        call_ask: float,
        put_bid: float,
        put_ask: float,
        risk_free_rate: float = 0.05
    ) -> Optional[OptionsArbitrageOpportunity]:
        """
        Check for put-call parity violations

        Put-Call Parity: C - P = S - K*e^(-r*T)
        Where: C=call, P=put, S=stock, K=strike, r=rate, T=time
        """

        T = expiration_days / 365.0
        pv_strike = strike * np.exp(-risk_free_rate * T)

        # Theoretical relationship
        theoretical_diff = stock_price - pv_strike

        # Actual bid-ask spread on options
        # Buy call (ask), sell put (bid)
        actual_diff_buy_call = call_ask - put_bid

        # Sell call (bid), buy put (ask)
        actual_diff_sell_call = call_bid - put_ask

        # Check for arbitrage
        # If we can buy call + sell put for less than stock - PV(strike)
        buy_call_arb = theoretical_diff - actual_diff_buy_call

        # If we can sell call + buy put for more than stock - PV(strike)
        sell_call_arb = actual_diff_sell_call - theoretical_diff

        profit = 0
        legs = []
        strategy = None

        if buy_call_arb > self.min_profit_threshold * strike:
            # Arbitrage: Buy call, sell put, sell stock
            profit = buy_call_arb
            strategy = "put_call_parity_long_synthetic"
            legs = [
                {'action': 'buy', 'type': 'call', 'strike': strike, 'price': call_ask},
                {'action': 'sell', 'type': 'put', 'strike': strike, 'price': put_bid},
                {'action': 'short', 'type': 'stock', 'price': stock_price}
            ]

        elif sell_call_arb > self.min_profit_threshold * strike:
            # Arbitrage: Sell call, buy put, buy stock
            profit = sell_call_arb
            strategy = "put_call_parity_short_synthetic"
            legs = [
                {'action': 'sell', 'type': 'call', 'strike': strike, 'price': call_bid},
                {'action': 'buy', 'type': 'put', 'strike': strike, 'price': put_ask},
                {'action': 'long', 'type': 'stock', 'price': stock_price}
            ]

        if profit > 0:
            capital_required = strike  # Approximate
            annual_return = (profit / capital_required) * (365 / expiration_days)

            return OptionsArbitrageOpportunity(
                strategy_type=strategy,
                symbol=symbol,
                legs=legs,
                expected_profit=profit,
                risk=0.01 * profit,  # Very low risk if true arbitrage
                probability_of_profit=0.95,
                capital_required=capital_required,
                annual_return=annual_return
            )

        return None

    def analyze_box_spread(
        self,
        symbol: str,
        stock_price: float,
        strike_low: float,
        strike_high: float,
        expiration_days: int,
        call_low_bid: float,
        call_low_ask: float,
        call_high_bid: float,
        call_high_ask: float,
        put_low_bid: float,
        put_low_ask: float,
        put_high_bid: float,
        put_high_ask: float,
        risk_free_rate: float = 0.05
    ) -> Optional[OptionsArbitrageOpportunity]:
        """
        Box spread arbitrage

        Box = Long call spread + Short put spread (same strikes)
        Theoretical value = PV(strike_high - strike_low)
        """

        T = expiration_days / 365.0
        theoretical_value = (strike_high - strike_low) * np.exp(-risk_free_rate * T)

        # Cost to enter box spread
        # Buy low call, sell high call, sell low put, buy high put
        cost = call_low_ask - call_high_bid - put_low_bid + put_high_ask

        # Profit = theoretical value - cost
        profit = theoretical_value - cost

        if profit > self.min_profit_threshold * (strike_high - strike_low):
            capital_required = cost
            annual_return = (profit / capital_required) * (365 / expiration_days) if capital_required > 0 else 0

            legs = [
                {'action': 'buy', 'type': 'call', 'strike': strike_low, 'price': call_low_ask},
                {'action': 'sell', 'type': 'call', 'strike': strike_high, 'price': call_high_bid},
                {'action': 'sell', 'type': 'put', 'strike': strike_low, 'price': put_low_bid},
                {'action': 'buy', 'type': 'put', 'strike': strike_high, 'price': put_high_ask}
            ]

            return OptionsArbitrageOpportunity(
                strategy_type="box_spread",
                symbol=symbol,
                legs=legs,
                expected_profit=profit,
                risk=0.01 * profit,
                probability_of_profit=0.99,  # Very high for true box arbitrage
                capital_required=capital_required,
                annual_return=annual_return
            )

        return None

    def analyze_volatility_arbitrage(
        self,
        symbol: str,
        stock_price: float,
        strike: float,
        expiration_days: int,
        option_price: float,
        option_type: str,
        historical_volatility: float,
        implied_volatility: float,
        risk_free_rate: float = 0.05
    ) -> Optional[OptionsArbitrageOpportunity]:
        """
        Volatility arbitrage: Trade when IV significantly diverges from HV
        """

        # IV vs HV ratio
        iv_hv_ratio = implied_volatility / historical_volatility if historical_volatility > 0 else 1.0

        # Thresholds
        IV_OVERPRICED_THRESHOLD = 1.3  # IV > 1.3x HV
        IV_UNDERPRICED_THRESHOLD = 0.7  # IV < 0.7x HV

        opportunity = None

        if iv_hv_ratio > IV_OVERPRICED_THRESHOLD:
            # IV too high -> Sell options (expect IV to decrease)
            strategy = f"sell_{option_type}_high_iv"

            # Estimate profit from IV crush
            iv_expected = (implied_volatility + historical_volatility) / 2
            expected_price_after_crush = self._black_scholes_price(
                stock_price, strike, expiration_days, risk_free_rate,
                iv_expected, option_type
            )

            profit = option_price - expected_price_after_crush

            if profit > 0:
                legs = [
                    {'action': 'sell', 'type': option_type, 'strike': strike, 'price': option_price}
                ]

                opportunity = OptionsArbitrageOpportunity(
                    strategy_type=strategy,
                    symbol=symbol,
                    legs=legs,
                    expected_profit=profit,
                    risk=option_price * 0.5,  # Max loss if stock moves against us
                    probability_of_profit=0.65,
                    capital_required=option_price * 100,  # Margin requirement
                    annual_return=(profit / (option_price * 100)) * (365 / expiration_days)
                )

        elif iv_hv_ratio < IV_UNDERPRICED_THRESHOLD:
            # IV too low -> Buy options (expect IV to increase)
            strategy = f"buy_{option_type}_low_iv"

            iv_expected = (implied_volatility + historical_volatility) / 2
            expected_price_after_expansion = self._black_scholes_price(
                stock_price, strike, expiration_days, risk_free_rate,
                iv_expected, option_type
            )

            profit = expected_price_after_expansion - option_price

            if profit > 0:
                legs = [
                    {'action': 'buy', 'type': option_type, 'strike': strike, 'price': option_price}
                ]

                opportunity = OptionsArbitrageOpportunity(
                    strategy_type=strategy,
                    symbol=symbol,
                    legs=legs,
                    expected_profit=profit,
                    risk=option_price,  # Max loss = premium paid
                    probability_of_profit=0.60,
                    capital_required=option_price * 100,
                    annual_return=(profit / (option_price * 100)) * (365 / expiration_days)
                )

        return opportunity

    def _black_scholes_price(
        self,
        S: float,  # Stock price
        K: float,  # Strike
        T_days: int,  # Days to expiration
        r: float,  # Risk-free rate
        sigma: float,  # Volatility
        option_type: str  # 'call' or 'put'
    ) -> float:
        """Calculate Black-Scholes option price"""

        T = T_days / 365.0

        if T <= 0:
            # At expiration
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return price

    def scan_for_opportunities(
        self,
        options_data: pd.DataFrame,
        stock_prices: Dict[str, float],
        historical_volatilities: Dict[str, float]
    ) -> List[OptionsArbitrageOpportunity]:
        """
        Scan options universe for arbitrage opportunities
        """

        opportunities = []

        logger.info(f"Scanning {len(options_data)} options for arbitrage...")

        # Filter for liquid options
        liquid_options = options_data[
            (options_data['openInterest'] >= self.min_open_interest) &
            (options_data['volume'] > 0)
        ].copy()

        # Calculate bid-ask spread
        liquid_options['spread_pct'] = (
            (liquid_options['ask'] - liquid_options['bid']) /
            ((liquid_options['ask'] + liquid_options['bid']) / 2)
        )

        liquid_options = liquid_options[
            liquid_options['spread_pct'] <= self.max_bid_ask_spread_pct
        ]

        logger.info(f"  {len(liquid_options)} liquid options after filtering")

        # Group by symbol and expiration
        for (symbol, expiration), group in liquid_options.groupby(['symbol', 'expiration']):
            if symbol not in stock_prices:
                continue

            stock_price = stock_prices[symbol]
            expiration_date = pd.to_datetime(expiration)
            days_to_expiration = (expiration_date - datetime.now()).days

            if days_to_expiration < 1:
                continue

            # Get calls and puts
            calls = group[group['type'] == 'call'].sort_values('strike')
            puts = group[group['type'] == 'put'].sort_values('strike')

            # Check put-call parity for each strike
            for strike in calls['strike'].unique():
                if strike not in puts['strike'].values:
                    continue

                call_row = calls[calls['strike'] == strike].iloc[0]
                put_row = puts[puts['strike'] == strike].iloc[0]

                # Put-call parity check
                opp = self.analyze_put_call_parity(
                    symbol=symbol,
                    stock_price=stock_price,
                    strike=strike,
                    expiration_days=days_to_expiration,
                    call_bid=call_row['bid'],
                    call_ask=call_row['ask'],
                    put_bid=put_row['bid'],
                    put_ask=put_row['ask']
                )

                if opp:
                    opportunities.append(opp)

            # Box spread check (need at least 2 strikes)
            if len(calls) >= 2:
                for i in range(len(calls) - 1):
                    strike_low = calls.iloc[i]['strike']
                    strike_high = calls.iloc[i+1]['strike']

                    if strike_low not in puts['strike'].values or strike_high not in puts['strike'].values:
                        continue

                    call_low = calls[calls['strike'] == strike_low].iloc[0]
                    call_high = calls[calls['strike'] == strike_high].iloc[0]
                    put_low = puts[puts['strike'] == strike_low].iloc[0]
                    put_high = puts[puts['strike'] == strike_high].iloc[0]

                    opp = self.analyze_box_spread(
                        symbol=symbol,
                        stock_price=stock_price,
                        strike_low=strike_low,
                        strike_high=strike_high,
                        expiration_days=days_to_expiration,
                        call_low_bid=call_low['bid'],
                        call_low_ask=call_low['ask'],
                        call_high_bid=call_high['bid'],
                        call_high_ask=call_high['ask'],
                        put_low_bid=put_low['bid'],
                        put_low_ask=put_low['ask'],
                        put_high_bid=put_high['bid'],
                        put_high_ask=put_high['ask']
                    )

                    if opp:
                        opportunities.append(opp)

            # Volatility arbitrage
            if symbol in historical_volatilities:
                for _, option_row in group.iterrows():
                    if 'impliedVolatility' in option_row and option_row['impliedVolatility'] > 0:
                        opp = self.analyze_volatility_arbitrage(
                            symbol=symbol,
                            stock_price=stock_price,
                            strike=option_row['strike'],
                            expiration_days=days_to_expiration,
                            option_price=(option_row['bid'] + option_row['ask']) / 2,
                            option_type=option_row['type'],
                            historical_volatility=historical_volatilities[symbol],
                            implied_volatility=option_row['impliedVolatility']
                        )

                        if opp:
                            opportunities.append(opp)

        # Sort by expected profit
        opportunities.sort(key=lambda x: x.expected_profit, reverse=True)

        logger.info(f"Found {len(opportunities)} arbitrage opportunities")

        return opportunities

    def generate_signals(
        self,
        options_data: pd.DataFrame,
        stock_prices: Dict[str, float],
        historical_volatilities: Dict[str, float],
        max_positions: int = 10
    ) -> List[OptionsArbitrageOpportunity]:
        """Generate top arbitrage signals"""

        opportunities = self.scan_for_opportunities(
            options_data, stock_prices, historical_volatilities
        )

        # Filter by capital requirements
        filtered = [
            opp for opp in opportunities
            if opp.capital_required <= self.max_capital_per_trade
        ]

        # Return top N
        return filtered[:max_positions]
