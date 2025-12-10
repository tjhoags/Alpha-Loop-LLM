"""
================================================================================
INSTITUTIONAL VALUATION SUITE
================================================================================
Complete valuation toolkit for a long-short quant hedge fund:

INTRINSIC VALUATION:
- DCF (Discounted Cash Flow) - FCFF and FCFE approaches
- SBC-Adjusted DCF (Hogan Model) - Stock-based comp adjusted
- Dividend Discount Model (DDM)
- Residual Income Model
- APV (Adjusted Present Value)

RELATIVE VALUATION:
- Trading Comparables (Peer Analysis)
- Precedent Transactions Analysis
- Sum-of-the-Parts (SOTP)
- Football Field Analysis

TRANSACTION MODELS:
- LBO (Leveraged Buyout) Model
- M&A Accretion/Dilution Analysis
- Merger Model

FACTOR STYLES:
- Value Factors (P/E, P/B, EV/EBITDA, FCF Yield)
- Growth Factors (Revenue Growth, EPS Growth, Guidance)
- Quality Factors (ROE, ROIC, Margins, Stability)
- Momentum Factors (Price momentum, Earnings momentum)
- Size Factors (Market cap, Enterprise value)
- Volatility Factors (Beta, Idiosyncratic vol)
- Dividend Factors (Yield, Payout, Growth)
- Sentiment Factors (Analyst revisions, Short interest)

================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FinancialProjections:
    """Multi-year financial projections for DCF."""
    years: int = 5
    revenue: List[float] = field(default_factory=list)
    revenue_growth: List[float] = field(default_factory=list)
    ebitda: List[float] = field(default_factory=list)
    ebitda_margin: List[float] = field(default_factory=list)
    ebit: List[float] = field(default_factory=list)
    depreciation: List[float] = field(default_factory=list)
    capex: List[float] = field(default_factory=list)
    nwc_change: List[float] = field(default_factory=list)
    tax_rate: float = 0.25
    
    # Stock-Based Compensation (for Hogan Model)
    sbc: List[float] = field(default_factory=list)
    sbc_as_pct_revenue: List[float] = field(default_factory=list)


@dataclass
class LBOAssumptions:
    """LBO model assumptions."""
    purchase_price: float
    equity_contribution_pct: float = 0.30
    senior_debt_multiple: float = 4.0
    sub_debt_multiple: float = 1.5
    senior_rate: float = 0.06
    sub_rate: float = 0.10
    exit_multiple: float = 8.0
    holding_period: int = 5
    management_rollover_pct: float = 0.05
    transaction_fees_pct: float = 0.02


@dataclass
class MergerAssumptions:
    """M&A model assumptions."""
    acquirer_shares: float
    acquirer_price: float
    target_shares: float
    target_price: float
    premium_pct: float = 0.30
    cash_pct: float = 0.50  # Cash vs stock consideration
    synergies: float = 0.0
    synergy_phase_in_years: int = 3
    integration_costs: float = 0.0
    financing_rate: float = 0.05


class FactorStyle(Enum):
    """Factor investment styles."""
    VALUE = "value"
    GROWTH = "growth"
    QUALITY = "quality"
    MOMENTUM = "momentum"
    SIZE = "size"
    LOW_VOLATILITY = "low_volatility"
    DIVIDEND = "dividend"
    SENTIMENT = "sentiment"
    PROFITABILITY = "profitability"
    INVESTMENT = "investment"


# =============================================================================
# DCF MODELS
# =============================================================================

class DCFModel:
    """
    Discounted Cash Flow Model - Gold standard of intrinsic valuation.
    """
    
    @staticmethod
    def calculate_wacc(
        equity_value: float,
        debt_value: float,
        cost_of_equity: float,
        cost_of_debt: float,
        tax_rate: float
    ) -> float:
        """
        Weighted Average Cost of Capital.
        
        WACC = (E/V) * Re + (D/V) * Rd * (1 - T)
        """
        total_value = equity_value + debt_value
        if total_value == 0:
            return 0.10  # Default
        
        weight_equity = equity_value / total_value
        weight_debt = debt_value / total_value
        
        wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1 - tax_rate))
        return wacc
    
    @staticmethod
    def cost_of_equity_capm(
        risk_free_rate: float,
        beta: float,
        equity_risk_premium: float = 0.055
    ) -> float:
        """
        CAPM: Re = Rf + β * (Rm - Rf)
        """
        return risk_free_rate + beta * equity_risk_premium
    
    @staticmethod
    def calculate_fcff(
        ebit: float,
        tax_rate: float,
        depreciation: float,
        capex: float,
        nwc_change: float
    ) -> float:
        """
        Free Cash Flow to Firm.
        
        FCFF = EBIT * (1 - T) + D&A - CapEx - ΔNWC
        """
        nopat = ebit * (1 - tax_rate)
        fcff = nopat + depreciation - capex - nwc_change
        return fcff
    
    @staticmethod
    def calculate_fcfe(
        fcff: float,
        interest_expense: float,
        tax_rate: float,
        net_borrowing: float
    ) -> float:
        """
        Free Cash Flow to Equity.
        
        FCFE = FCFF - Interest * (1 - T) + Net Borrowing
        """
        after_tax_interest = interest_expense * (1 - tax_rate)
        fcfe = fcff - after_tax_interest + net_borrowing
        return fcfe
    
    @staticmethod
    def terminal_value_gordon(
        final_fcf: float,
        discount_rate: float,
        perpetual_growth: float = 0.025
    ) -> float:
        """
        Gordon Growth Model for Terminal Value.
        
        TV = FCF * (1 + g) / (r - g)
        """
        if discount_rate <= perpetual_growth:
            raise ValueError("Discount rate must exceed perpetual growth rate")
        
        return final_fcf * (1 + perpetual_growth) / (discount_rate - perpetual_growth)
    
    @staticmethod
    def terminal_value_exit_multiple(
        final_ebitda: float,
        exit_multiple: float
    ) -> float:
        """
        Exit Multiple approach for Terminal Value.
        """
        return final_ebitda * exit_multiple
    
    @staticmethod
    def dcf_valuation(
        projections: FinancialProjections,
        wacc: float,
        perpetual_growth: float = 0.025,
        use_exit_multiple: bool = False,
        exit_multiple: float = 10.0
    ) -> Dict[str, Any]:
        """
        Full DCF Valuation.
        
        Returns enterprise value, equity value, and implied share price.
        """
        # Calculate FCF for each projected year
        fcfs = []
        for i in range(projections.years):
            fcf = DCFModel.calculate_fcff(
                ebit=projections.ebit[i] if i < len(projections.ebit) else projections.ebit[-1],
                tax_rate=projections.tax_rate,
                depreciation=projections.depreciation[i] if i < len(projections.depreciation) else projections.depreciation[-1],
                capex=projections.capex[i] if i < len(projections.capex) else projections.capex[-1],
                nwc_change=projections.nwc_change[i] if i < len(projections.nwc_change) else 0
            )
            fcfs.append(fcf)
        
        # Discount FCFs
        pv_fcfs = []
        for i, fcf in enumerate(fcfs):
            discount_factor = (1 + wacc) ** (i + 1)
            pv_fcfs.append(fcf / discount_factor)
        
        # Terminal Value
        if use_exit_multiple:
            final_ebitda = projections.ebitda[-1] if projections.ebitda else fcfs[-1] * 1.5
            tv = DCFModel.terminal_value_exit_multiple(final_ebitda, exit_multiple)
        else:
            tv = DCFModel.terminal_value_gordon(fcfs[-1], wacc, perpetual_growth)
        
        # Discount TV to present
        pv_tv = tv / ((1 + wacc) ** projections.years)
        
        # Enterprise Value
        enterprise_value = sum(pv_fcfs) + pv_tv
        
        return {
            "enterprise_value": enterprise_value,
            "pv_fcfs": sum(pv_fcfs),
            "pv_terminal_value": pv_tv,
            "terminal_value": tv,
            "fcf_projections": fcfs,
            "wacc": wacc,
            "implied_ev_ebitda": enterprise_value / projections.ebitda[-1] if projections.ebitda and projections.ebitda[-1] > 0 else None
        }


class HoganModel(DCFModel):
    """
    SBC-Adjusted DCF Model (Hogan Model)
    
    Treats Stock-Based Compensation as a TRUE economic cost, not just
    an accounting adjustment. This is critical for accurate valuation
    of tech companies with heavy SBC.
    
    Key Insight: SBC dilutes existing shareholders and is a real cost
    that should be deducted from cash flows, not added back.
    """
    
    @staticmethod
    def calculate_sbc_adjusted_fcf(
        ebit: float,
        tax_rate: float,
        depreciation: float,
        capex: float,
        nwc_change: float,
        sbc: float,
        sbc_tax_benefit: bool = True
    ) -> float:
        """
        SBC-Adjusted Free Cash Flow.
        
        Unlike traditional DCF which adds back SBC, the Hogan Model
        treats SBC as a real cash expense (economic cost to shareholders).
        
        FCFF_adj = EBIT * (1-T) + D&A - CapEx - ΔNWC - SBC_adj
        
        Where SBC_adj = SBC * (1 - T) if tax benefit, else SBC
        """
        nopat = ebit * (1 - tax_rate)
        
        # SBC is a real cost - do NOT add back
        # But account for tax benefit if applicable
        if sbc_tax_benefit:
            sbc_cost = sbc * (1 - tax_rate)
        else:
            sbc_cost = sbc
        
        fcff_adjusted = nopat + depreciation - capex - nwc_change - sbc_cost
        return fcff_adjusted
    
    @staticmethod
    def calculate_dilution_impact(
        current_shares: float,
        options_outstanding: float,
        rsus_outstanding: float,
        average_strike_price: float,
        current_price: float
    ) -> Dict[str, float]:
        """
        Calculate dilution from outstanding equity awards.
        Uses Treasury Stock Method.
        """
        # Options dilution (Treasury Stock Method)
        if current_price > average_strike_price:
            options_proceeds = options_outstanding * average_strike_price
            shares_repurchased = options_proceeds / current_price
            net_options_dilution = options_outstanding - shares_repurchased
        else:
            net_options_dilution = 0
        
        # RSUs always dilute
        rsu_dilution = rsus_outstanding
        
        total_dilution = net_options_dilution + rsu_dilution
        diluted_shares = current_shares + total_dilution
        dilution_pct = total_dilution / current_shares
        
        return {
            "basic_shares": current_shares,
            "diluted_shares": diluted_shares,
            "total_dilution": total_dilution,
            "dilution_pct": dilution_pct,
            "options_dilution": net_options_dilution,
            "rsu_dilution": rsu_dilution
        }
    
    @staticmethod
    def sbc_adjusted_dcf(
        projections: FinancialProjections,
        wacc: float,
        current_shares: float,
        annual_sbc_dilution_pct: float = 0.02,
        perpetual_growth: float = 0.025
    ) -> Dict[str, Any]:
        """
        Full SBC-Adjusted DCF (Hogan Model).
        
        Key differences from traditional DCF:
        1. SBC treated as real cost, not added back
        2. Future dilution explicitly modeled
        3. Per-share value accounts for growing share count
        """
        # Calculate SBC-adjusted FCFs
        fcfs_adjusted = []
        for i in range(projections.years):
            sbc = projections.sbc[i] if i < len(projections.sbc) else projections.sbc[-1] if projections.sbc else 0
            
            fcf = HoganModel.calculate_sbc_adjusted_fcf(
                ebit=projections.ebit[i] if i < len(projections.ebit) else projections.ebit[-1],
                tax_rate=projections.tax_rate,
                depreciation=projections.depreciation[i] if i < len(projections.depreciation) else projections.depreciation[-1],
                capex=projections.capex[i] if i < len(projections.capex) else projections.capex[-1],
                nwc_change=projections.nwc_change[i] if i < len(projections.nwc_change) else 0,
                sbc=sbc
            )
            fcfs_adjusted.append(fcf)
        
        # Discount FCFs
        pv_fcfs = []
        for i, fcf in enumerate(fcfs_adjusted):
            discount_factor = (1 + wacc) ** (i + 1)
            pv_fcfs.append(fcf / discount_factor)
        
        # Terminal value (using adjusted final FCF)
        tv = DCFModel.terminal_value_gordon(fcfs_adjusted[-1], wacc, perpetual_growth)
        pv_tv = tv / ((1 + wacc) ** projections.years)
        
        # Enterprise Value
        enterprise_value = sum(pv_fcfs) + pv_tv
        
        # Project future diluted share count
        shares_at_exit = current_shares * ((1 + annual_sbc_dilution_pct) ** projections.years)
        
        return {
            "enterprise_value": enterprise_value,
            "pv_fcfs": sum(pv_fcfs),
            "pv_terminal_value": pv_tv,
            "terminal_value": tv,
            "fcf_projections_adjusted": fcfs_adjusted,
            "current_shares": current_shares,
            "projected_diluted_shares": shares_at_exit,
            "cumulative_dilution_pct": (shares_at_exit / current_shares) - 1,
            "model_type": "Hogan SBC-Adjusted DCF"
        }


# =============================================================================
# LBO MODEL
# =============================================================================

class LBOModel:
    """
    Leveraged Buyout Model - Private Equity valuation approach.
    """
    
    @staticmethod
    def calculate_sources_uses(
        assumptions: LBOAssumptions,
        ltm_ebitda: float
    ) -> Dict[str, Any]:
        """
        Sources and Uses of Funds for LBO.
        """
        purchase_price = assumptions.purchase_price
        
        # Uses
        transaction_fees = purchase_price * assumptions.transaction_fees_pct
        total_uses = purchase_price + transaction_fees
        
        # Sources
        senior_debt = ltm_ebitda * assumptions.senior_debt_multiple
        sub_debt = ltm_ebitda * assumptions.sub_debt_multiple
        total_debt = senior_debt + sub_debt
        
        management_rollover = purchase_price * assumptions.management_rollover_pct
        sponsor_equity = total_uses - total_debt - management_rollover
        
        return {
            "uses": {
                "purchase_price": purchase_price,
                "transaction_fees": transaction_fees,
                "total_uses": total_uses
            },
            "sources": {
                "senior_debt": senior_debt,
                "subordinated_debt": sub_debt,
                "total_debt": total_debt,
                "management_rollover": management_rollover,
                "sponsor_equity": sponsor_equity,
                "total_sources": total_debt + management_rollover + sponsor_equity
            },
            "credit_stats": {
                "total_leverage": total_debt / ltm_ebitda,
                "senior_leverage": senior_debt / ltm_ebitda,
                "equity_contribution_pct": sponsor_equity / total_uses
            }
        }
    
    @staticmethod
    def project_debt_schedule(
        initial_debt: float,
        interest_rate: float,
        years: int,
        fcf_for_paydown: List[float],
        mandatory_amort_pct: float = 0.05
    ) -> Dict[str, List[float]]:
        """
        Project debt paydown over holding period.
        """
        beginning_balance = [initial_debt]
        interest_expense = []
        mandatory_amort = []
        optional_paydown = []
        ending_balance = []
        
        for i in range(years):
            # Interest
            interest = beginning_balance[i] * interest_rate
            interest_expense.append(interest)
            
            # Mandatory amortization
            mandatory = initial_debt * mandatory_amort_pct
            mandatory_amort.append(min(mandatory, beginning_balance[i]))
            
            # Optional paydown from excess FCF
            fcf = fcf_for_paydown[i] if i < len(fcf_for_paydown) else 0
            available_for_paydown = max(0, fcf - interest - mandatory)
            remaining_debt = beginning_balance[i] - mandatory_amort[i]
            optional = min(available_for_paydown, remaining_debt)
            optional_paydown.append(optional)
            
            # Ending balance
            end_bal = beginning_balance[i] - mandatory_amort[i] - optional_paydown[i]
            ending_balance.append(max(0, end_bal))
            
            if i < years - 1:
                beginning_balance.append(ending_balance[i])
        
        return {
            "beginning_balance": beginning_balance,
            "interest_expense": interest_expense,
            "mandatory_amortization": mandatory_amort,
            "optional_paydown": optional_paydown,
            "ending_balance": ending_balance,
            "total_debt_paydown": initial_debt - ending_balance[-1]
        }
    
    @staticmethod
    def calculate_returns(
        assumptions: LBOAssumptions,
        entry_ebitda: float,
        exit_ebitda: float,
        sponsor_equity: float,
        exit_debt: float
    ) -> Dict[str, float]:
        """
        Calculate LBO returns (IRR and MOIC).
        """
        # Exit Enterprise Value
        exit_ev = exit_ebitda * assumptions.exit_multiple
        
        # Exit Equity Value
        exit_equity = exit_ev - exit_debt
        
        # Sponsor proceeds (assuming management gets 10% of upside)
        management_promote = max(0, exit_equity - sponsor_equity) * 0.10 * (assumptions.management_rollover_pct / 0.05)
        sponsor_proceeds = exit_equity - management_promote
        
        # MOIC (Multiple of Invested Capital)
        moic = sponsor_proceeds / sponsor_equity if sponsor_equity > 0 else 0
        
        # IRR
        cash_flows = [-sponsor_equity] + [0] * (assumptions.holding_period - 1) + [sponsor_proceeds]
        irr = np.irr(cash_flows) if sponsor_equity > 0 else 0
        
        return {
            "exit_enterprise_value": exit_ev,
            "exit_equity_value": exit_equity,
            "sponsor_proceeds": sponsor_proceeds,
            "moic": moic,
            "irr": irr,
            "holding_period": assumptions.holding_period,
            "entry_multiple": assumptions.purchase_price / entry_ebitda,
            "exit_multiple": assumptions.exit_multiple
        }


# =============================================================================
# M&A MODEL
# =============================================================================

class MergerModel:
    """
    M&A Accretion/Dilution Analysis.
    """
    
    @staticmethod
    def calculate_deal_value(assumptions: MergerAssumptions) -> Dict[str, float]:
        """
        Calculate total deal value and consideration mix.
        """
        target_equity_value = assumptions.target_shares * assumptions.target_price
        premium_value = target_equity_value * assumptions.premium_pct
        offer_value = target_equity_value + premium_value
        
        cash_consideration = offer_value * assumptions.cash_pct
        stock_consideration = offer_value * (1 - assumptions.cash_pct)
        
        # Exchange ratio
        offer_price_per_share = offer_value / assumptions.target_shares
        exchange_ratio = offer_price_per_share / assumptions.acquirer_price
        new_shares_issued = assumptions.target_shares * exchange_ratio * (1 - assumptions.cash_pct)
        
        return {
            "target_equity_value": target_equity_value,
            "premium_pct": assumptions.premium_pct,
            "premium_value": premium_value,
            "offer_value": offer_value,
            "cash_consideration": cash_consideration,
            "stock_consideration": stock_consideration,
            "exchange_ratio": exchange_ratio,
            "new_shares_issued": new_shares_issued
        }
    
    @staticmethod
    def accretion_dilution(
        assumptions: MergerAssumptions,
        acquirer_net_income: float,
        target_net_income: float,
        synergies_after_tax: float = 0
    ) -> Dict[str, Any]:
        """
        Accretion/Dilution Analysis.
        """
        deal = MergerModel.calculate_deal_value(assumptions)
        
        # Pro Forma Shares
        pf_shares = assumptions.acquirer_shares + deal["new_shares_issued"]
        
        # Interest expense on debt for cash portion
        interest_expense = deal["cash_consideration"] * assumptions.financing_rate * (1 - 0.25)  # After-tax
        
        # Foregone interest on cash used
        foregone_interest = deal["cash_consideration"] * 0.02 * (1 - 0.25)  # Assume 2% yield on cash
        
        # Pro Forma Net Income
        pf_net_income = (
            acquirer_net_income +
            target_net_income +
            synergies_after_tax -
            interest_expense -
            foregone_interest -
            assumptions.integration_costs * (1 - 0.25)  # After-tax integration costs
        )
        
        # EPS Analysis
        acquirer_eps = acquirer_net_income / assumptions.acquirer_shares
        pf_eps = pf_net_income / pf_shares
        
        accretion_dilution = (pf_eps / acquirer_eps) - 1
        
        return {
            "acquirer_standalone_eps": acquirer_eps,
            "pro_forma_eps": pf_eps,
            "accretion_dilution_pct": accretion_dilution,
            "is_accretive": accretion_dilution > 0,
            "pro_forma_shares": pf_shares,
            "pro_forma_net_income": pf_net_income,
            "synergies_needed_for_breakeven": -accretion_dilution * acquirer_eps * pf_shares if accretion_dilution < 0 else 0,
            "deal_value": deal
        }


# =============================================================================
# COMPARABLE ANALYSIS
# =============================================================================

class ComparableAnalysis:
    """
    Trading Comparables and Precedent Transactions.
    """
    
    @staticmethod
    def trading_comps(
        peer_data: pd.DataFrame,
        target_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Trading Comparables Analysis.
        
        peer_data should have columns:
        - ticker
        - market_cap
        - ev
        - revenue
        - ebitda
        - net_income
        - growth_rate
        """
        # Calculate multiples for peers
        peer_data = peer_data.copy()
        peer_data["ev_revenue"] = peer_data["ev"] / peer_data["revenue"]
        peer_data["ev_ebitda"] = peer_data["ev"] / peer_data["ebitda"]
        peer_data["pe"] = peer_data["market_cap"] / peer_data["net_income"]
        peer_data["peg"] = peer_data["pe"] / (peer_data["growth_rate"] * 100)
        
        # Statistics
        stats = {}
        for multiple in ["ev_revenue", "ev_ebitda", "pe", "peg"]:
            stats[multiple] = {
                "mean": peer_data[multiple].mean(),
                "median": peer_data[multiple].median(),
                "25th": peer_data[multiple].quantile(0.25),
                "75th": peer_data[multiple].quantile(0.75),
                "min": peer_data[multiple].min(),
                "max": peer_data[multiple].max()
            }
        
        # Implied valuations for target
        implied = {}
        if "revenue" in target_metrics:
            implied["ev_from_revenue"] = {
                "mean": target_metrics["revenue"] * stats["ev_revenue"]["mean"],
                "median": target_metrics["revenue"] * stats["ev_revenue"]["median"]
            }
        if "ebitda" in target_metrics:
            implied["ev_from_ebitda"] = {
                "mean": target_metrics["ebitda"] * stats["ev_ebitda"]["mean"],
                "median": target_metrics["ebitda"] * stats["ev_ebitda"]["median"]
            }
        if "net_income" in target_metrics:
            implied["equity_from_pe"] = {
                "mean": target_metrics["net_income"] * stats["pe"]["mean"],
                "median": target_metrics["net_income"] * stats["pe"]["median"]
            }
        
        return {
            "peer_multiples": peer_data[["ticker", "ev_revenue", "ev_ebitda", "pe", "peg"]].to_dict("records"),
            "statistics": stats,
            "implied_valuations": implied
        }
    
    @staticmethod
    def precedent_transactions(
        transaction_data: pd.DataFrame,
        target_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Precedent Transactions Analysis.
        
        transaction_data should have:
        - date
        - target
        - acquirer
        - deal_value
        - target_revenue
        - target_ebitda
        - premium_pct
        """
        txn = transaction_data.copy()
        txn["ev_revenue"] = txn["deal_value"] / txn["target_revenue"]
        txn["ev_ebitda"] = txn["deal_value"] / txn["target_ebitda"]
        
        stats = {}
        for multiple in ["ev_revenue", "ev_ebitda", "premium_pct"]:
            stats[multiple] = {
                "mean": txn[multiple].mean(),
                "median": txn[multiple].median(),
                "25th": txn[multiple].quantile(0.25),
                "75th": txn[multiple].quantile(0.75)
            }
        
        # Implied takeover value
        implied = {}
        if "revenue" in target_metrics:
            implied["ev_from_revenue"] = target_metrics["revenue"] * stats["ev_revenue"]["median"]
        if "ebitda" in target_metrics:
            implied["ev_from_ebitda"] = target_metrics["ebitda"] * stats["ev_ebitda"]["median"]
        
        return {
            "transactions": txn.to_dict("records"),
            "statistics": stats,
            "implied_takeover_value": implied,
            "median_control_premium": stats["premium_pct"]["median"]
        }


# =============================================================================
# FACTOR MODELS - COMPREHENSIVE
# =============================================================================

class FactorModels:
    """
    Complete factor model suite for quantitative investing.
    """
    
    # -------------------------------------------------------------------------
    # VALUE FACTORS
    # -------------------------------------------------------------------------
    
    @staticmethod
    def value_factors(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate value factor scores.
        
        Required columns: market_cap, book_value, earnings, ebitda, ev, fcf, revenue
        """
        df = data.copy()
        
        # Price to Book
        df["ptb"] = df["market_cap"] / df["book_value"]
        df["ptb_rank"] = df["ptb"].rank(ascending=True, pct=True)
        
        # Price to Earnings
        df["pe"] = df["market_cap"] / df["earnings"]
        df["pe_rank"] = df["pe"].rank(ascending=True, pct=True)
        
        # EV/EBITDA
        df["ev_ebitda"] = df["ev"] / df["ebitda"]
        df["ev_ebitda_rank"] = df["ev_ebitda"].rank(ascending=True, pct=True)
        
        # FCF Yield
        df["fcf_yield"] = df["fcf"] / df["market_cap"]
        df["fcf_yield_rank"] = df["fcf_yield"].rank(ascending=False, pct=True)
        
        # Earnings Yield
        df["earnings_yield"] = df["earnings"] / df["market_cap"]
        df["earnings_yield_rank"] = df["earnings_yield"].rank(ascending=False, pct=True)
        
        # EV/Sales
        df["ev_sales"] = df["ev"] / df["revenue"]
        df["ev_sales_rank"] = df["ev_sales"].rank(ascending=True, pct=True)
        
        # Composite Value Score
        df["value_score"] = (
            df["ptb_rank"] * 0.15 +
            df["pe_rank"] * 0.20 +
            df["ev_ebitda_rank"] * 0.25 +
            df["fcf_yield_rank"] * 0.20 +
            df["earnings_yield_rank"] * 0.10 +
            df["ev_sales_rank"] * 0.10
        )
        
        return df
    
    # -------------------------------------------------------------------------
    # GROWTH FACTORS
    # -------------------------------------------------------------------------
    
    @staticmethod
    def growth_factors(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate growth factor scores.
        
        Required: revenue_growth, earnings_growth, fcf_growth, guidance_revision
        """
        df = data.copy()
        
        # Revenue Growth
        df["rev_growth_rank"] = df["revenue_growth"].rank(ascending=False, pct=True)
        
        # Earnings Growth
        df["earnings_growth_rank"] = df["earnings_growth"].rank(ascending=False, pct=True)
        
        # FCF Growth
        df["fcf_growth_rank"] = df["fcf_growth"].rank(ascending=False, pct=True)
        
        # Guidance Revision (positive = raised guidance)
        if "guidance_revision" in df.columns:
            df["guidance_rank"] = df["guidance_revision"].rank(ascending=False, pct=True)
        else:
            df["guidance_rank"] = 0.5
        
        # Sales Acceleration
        if "revenue_growth_prior" in df.columns:
            df["sales_acceleration"] = df["revenue_growth"] - df["revenue_growth_prior"]
            df["acceleration_rank"] = df["sales_acceleration"].rank(ascending=False, pct=True)
        else:
            df["acceleration_rank"] = 0.5
        
        # Composite Growth Score
        df["growth_score"] = (
            df["rev_growth_rank"] * 0.30 +
            df["earnings_growth_rank"] * 0.30 +
            df["fcf_growth_rank"] * 0.20 +
            df["guidance_rank"] * 0.10 +
            df["acceleration_rank"] * 0.10
        )
        
        return df
    
    # -------------------------------------------------------------------------
    # QUALITY FACTORS
    # -------------------------------------------------------------------------
    
    @staticmethod
    def quality_factors(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate quality factor scores.
        
        Required: roe, roic, gross_margin, operating_margin, debt_equity, 
                  earnings_variability, accruals
        """
        df = data.copy()
        
        # Profitability
        df["roe_rank"] = df["roe"].rank(ascending=False, pct=True)
        df["roic_rank"] = df["roic"].rank(ascending=False, pct=True)
        
        # Margins
        df["gross_margin_rank"] = df["gross_margin"].rank(ascending=False, pct=True)
        df["op_margin_rank"] = df["operating_margin"].rank(ascending=False, pct=True)
        
        # Leverage (lower is better for quality)
        df["leverage_rank"] = df["debt_equity"].rank(ascending=True, pct=True)
        
        # Stability (lower variability is better)
        if "earnings_variability" in df.columns:
            df["stability_rank"] = df["earnings_variability"].rank(ascending=True, pct=True)
        else:
            df["stability_rank"] = 0.5
        
        # Accruals (lower is better - more cash-based earnings)
        if "accruals" in df.columns:
            df["accruals_rank"] = df["accruals"].rank(ascending=True, pct=True)
        else:
            df["accruals_rank"] = 0.5
        
        # Composite Quality Score
        df["quality_score"] = (
            df["roe_rank"] * 0.15 +
            df["roic_rank"] * 0.20 +
            df["gross_margin_rank"] * 0.15 +
            df["op_margin_rank"] * 0.15 +
            df["leverage_rank"] * 0.15 +
            df["stability_rank"] * 0.10 +
            df["accruals_rank"] * 0.10
        )
        
        return df
    
    # -------------------------------------------------------------------------
    # MOMENTUM FACTORS
    # -------------------------------------------------------------------------
    
    @staticmethod
    def momentum_factors(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum factor scores.
        
        Required: return_1m, return_3m, return_6m, return_12m, 
                  earnings_surprise, revision_1m
        """
        df = data.copy()
        
        # Price Momentum (excluding most recent month - momentum crash protection)
        df["mom_12m_1m"] = df["return_12m"] - df["return_1m"]  # 12-1 month momentum
        df["price_mom_rank"] = df["mom_12m_1m"].rank(ascending=False, pct=True)
        
        # 6-month momentum
        df["mom_6m_rank"] = df["return_6m"].rank(ascending=False, pct=True)
        
        # 3-month momentum
        df["mom_3m_rank"] = df["return_3m"].rank(ascending=False, pct=True)
        
        # Earnings Momentum (surprise)
        if "earnings_surprise" in df.columns:
            df["surprise_rank"] = df["earnings_surprise"].rank(ascending=False, pct=True)
        else:
            df["surprise_rank"] = 0.5
        
        # Analyst Revisions
        if "revision_1m" in df.columns:
            df["revision_rank"] = df["revision_1m"].rank(ascending=False, pct=True)
        else:
            df["revision_rank"] = 0.5
        
        # Composite Momentum Score
        df["momentum_score"] = (
            df["price_mom_rank"] * 0.35 +
            df["mom_6m_rank"] * 0.20 +
            df["mom_3m_rank"] * 0.15 +
            df["surprise_rank"] * 0.15 +
            df["revision_rank"] * 0.15
        )
        
        return df
    
    # -------------------------------------------------------------------------
    # SIZE FACTORS
    # -------------------------------------------------------------------------
    
    @staticmethod
    def size_factors(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate size factor scores (small = higher score).
        
        Required: market_cap
        """
        df = data.copy()
        
        # Market Cap (smaller = higher rank for SMB factor)
        df["size_rank"] = df["market_cap"].rank(ascending=True, pct=True)
        
        # Log market cap for smoother distribution
        df["log_mcap"] = np.log(df["market_cap"])
        df["log_size_rank"] = df["log_mcap"].rank(ascending=True, pct=True)
        
        df["size_score"] = (df["size_rank"] + df["log_size_rank"]) / 2
        
        return df
    
    # -------------------------------------------------------------------------
    # LOW VOLATILITY FACTORS
    # -------------------------------------------------------------------------
    
    @staticmethod
    def low_volatility_factors(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate low volatility factor scores.
        
        Required: volatility_252d, beta, idio_vol, max_drawdown
        """
        df = data.copy()
        
        # Total Volatility (lower = better)
        df["vol_rank"] = df["volatility_252d"].rank(ascending=True, pct=True)
        
        # Beta (lower = better)
        df["beta_rank"] = df["beta"].rank(ascending=True, pct=True)
        
        # Idiosyncratic Volatility
        if "idio_vol" in df.columns:
            df["idio_vol_rank"] = df["idio_vol"].rank(ascending=True, pct=True)
        else:
            df["idio_vol_rank"] = df["vol_rank"]
        
        # Max Drawdown (less negative = better)
        if "max_drawdown" in df.columns:
            df["drawdown_rank"] = df["max_drawdown"].rank(ascending=False, pct=True)
        else:
            df["drawdown_rank"] = 0.5
        
        df["low_vol_score"] = (
            df["vol_rank"] * 0.35 +
            df["beta_rank"] * 0.30 +
            df["idio_vol_rank"] * 0.20 +
            df["drawdown_rank"] * 0.15
        )
        
        return df
    
    # -------------------------------------------------------------------------
    # DIVIDEND/YIELD FACTORS
    # -------------------------------------------------------------------------
    
    @staticmethod
    def dividend_factors(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate dividend factor scores.
        
        Required: dividend_yield, payout_ratio, dividend_growth, years_of_growth
        """
        df = data.copy()
        
        # Dividend Yield
        df["yield_rank"] = df["dividend_yield"].rank(ascending=False, pct=True)
        
        # Payout Ratio (moderate is best - not too high, not zero)
        df["payout_score"] = 1 - abs(df["payout_ratio"] - 0.40) / 0.60
        df["payout_score"] = df["payout_score"].clip(0, 1)
        df["payout_rank"] = df["payout_score"].rank(ascending=False, pct=True)
        
        # Dividend Growth
        if "dividend_growth" in df.columns:
            df["div_growth_rank"] = df["dividend_growth"].rank(ascending=False, pct=True)
        else:
            df["div_growth_rank"] = 0.5
        
        # Consecutive Years of Growth
        if "years_of_growth" in df.columns:
            df["consistency_rank"] = df["years_of_growth"].rank(ascending=False, pct=True)
        else:
            df["consistency_rank"] = 0.5
        
        df["dividend_score"] = (
            df["yield_rank"] * 0.35 +
            df["payout_rank"] * 0.20 +
            df["div_growth_rank"] * 0.25 +
            df["consistency_rank"] * 0.20
        )
        
        return df
    
    # -------------------------------------------------------------------------
    # SENTIMENT FACTORS
    # -------------------------------------------------------------------------
    
    @staticmethod
    def sentiment_factors(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sentiment factor scores.
        
        Required: short_interest, days_to_cover, analyst_rating, 
                  institutional_ownership, insider_buying
        """
        df = data.copy()
        
        # Short Interest (lower = better sentiment)
        if "short_interest" in df.columns:
            df["short_rank"] = df["short_interest"].rank(ascending=True, pct=True)
        else:
            df["short_rank"] = 0.5
        
        # Days to Cover
        if "days_to_cover" in df.columns:
            df["dtc_rank"] = df["days_to_cover"].rank(ascending=True, pct=True)
        else:
            df["dtc_rank"] = 0.5
        
        # Analyst Rating (higher = more bullish)
        if "analyst_rating" in df.columns:
            df["analyst_rank"] = df["analyst_rating"].rank(ascending=False, pct=True)
        else:
            df["analyst_rank"] = 0.5
        
        # Institutional Ownership
        if "institutional_ownership" in df.columns:
            df["inst_rank"] = df["institutional_ownership"].rank(ascending=False, pct=True)
        else:
            df["inst_rank"] = 0.5
        
        # Insider Buying
        if "insider_buying" in df.columns:
            df["insider_rank"] = df["insider_buying"].rank(ascending=False, pct=True)
        else:
            df["insider_rank"] = 0.5
        
        df["sentiment_score"] = (
            df["short_rank"] * 0.25 +
            df["dtc_rank"] * 0.15 +
            df["analyst_rank"] * 0.20 +
            df["inst_rank"] * 0.20 +
            df["insider_rank"] * 0.20
        )
        
        return df
    
    # -------------------------------------------------------------------------
    # COMPOSITE MULTI-FACTOR SCORE
    # -------------------------------------------------------------------------
    
    @staticmethod
    def composite_factor_score(
        data: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Calculate composite multi-factor score.
        """
        df = data.copy()
        
        # Default weights
        if weights is None:
            weights = {
                "value_score": 0.20,
                "growth_score": 0.15,
                "quality_score": 0.20,
                "momentum_score": 0.20,
                "size_score": 0.05,
                "low_vol_score": 0.10,
                "dividend_score": 0.05,
                "sentiment_score": 0.05
            }
        
        # Calculate composite
        composite = 0
        for factor, weight in weights.items():
            if factor in df.columns:
                composite += df[factor] * weight
        
        df["composite_factor_score"] = composite
        df["composite_rank"] = df["composite_factor_score"].rank(ascending=False, pct=True)
        
        return df


# =============================================================================
# FOOTBALL FIELD VISUALIZATION DATA
# =============================================================================

def football_field_data(
    dcf_range: Tuple[float, float],
    trading_comps_range: Tuple[float, float],
    precedent_txns_range: Tuple[float, float],
    lbo_range: Tuple[float, float],
    current_price: float,
    week_52_range: Tuple[float, float]
) -> Dict[str, Any]:
    """
    Prepare data for football field valuation chart.
    """
    return {
        "methods": [
            {"name": "DCF", "low": dcf_range[0], "high": dcf_range[1]},
            {"name": "Trading Comps", "low": trading_comps_range[0], "high": trading_comps_range[1]},
            {"name": "Precedent Transactions", "low": precedent_txns_range[0], "high": precedent_txns_range[1]},
            {"name": "LBO Analysis", "low": lbo_range[0], "high": lbo_range[1]},
            {"name": "52-Week Range", "low": week_52_range[0], "high": week_52_range[1]},
        ],
        "current_price": current_price,
        "implied_range": {
            "low": min(dcf_range[0], trading_comps_range[0], precedent_txns_range[0]),
            "high": max(dcf_range[1], trading_comps_range[1], precedent_txns_range[1]),
            "midpoint": np.mean([
                np.mean(dcf_range),
                np.mean(trading_comps_range),
                np.mean(precedent_txns_range)
            ])
        },
        "upside_downside": {
            "upside_pct": (np.mean([dcf_range[1], trading_comps_range[1]]) / current_price) - 1,
            "downside_pct": (np.mean([dcf_range[0], trading_comps_range[0]]) / current_price) - 1
        }
    }

