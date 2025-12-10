"""================================================================================
CROSS-ASSET SIGNALS - Multi-Market Relationships
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

Markets are interconnected. The best signals come from divergences.
================================================================================
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CrossAssetSignal:
    """Signal from cross-asset analysis."""

    signal_id: str
    signal_type: str
    primary_asset: str
    related_assets: List[str]
    direction: str
    confidence: float
    description: str
    divergence_data: Dict[str, Any]
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {k: v.isoformat() if isinstance(v, datetime) else v for k, v in self.__dict__.items()}


class CrossAssetSignals:
    """CROSS-ASSET SIGNALS

    1. Credit-Equity Divergence - CDS vs stock price
    2. Currency-Equity Correlation Break - FX exposure unpriced
    3. Commodity Input Cost Lag - Input prices vs margins
    4. Treasury-Equity Correlation Regime - Rate sensitivity shift
    5. VIX Term Structure vs Equity - Fear gauge mismatch
    6. Sector Rotation Leading Indicator - Early sector shifts
    7. Geographic Revenue Exposure - Country risk unpriced
    8. Intermarket Divergence Scanner - Cross-market gaps
    9. Factor Crowding Detector - Momentum/value crowding
    10. Liquidity Regime Shift - Cross-asset liquidity signals
    """

    def __init__(self):
        self.signals_detected: List[CrossAssetSignal] = []

    def credit_equity_divergence(
        self,
        ticker: str,
        stock_return_30d: float,
        cds_spread_change_30d: float,  # bps
        bond_spread_change_30d: float,  # bps
        correlation_historical: float,
    ) -> Optional[CrossAssetSignal]:
        """Credit and equity usually move together.
        When they diverge, one is wrong.

        Credit often leads (smarter money).
        """
        # Normalize CDS change (100bps = significant)
        cds_signal = -cds_spread_change_30d / 100  # Wider spread = bearish

        divergence = abs(stock_return_30d - cds_signal)

        if divergence < 0.05:
            return None

        # Credit leads equity most of the time
        if cds_spread_change_30d > 50 and stock_return_30d > 0:
            direction = "bearish"
            confidence = 0.70
            desc = f"CREDIT WARNING: {ticker} CDS widening +{cds_spread_change_30d:.0f}bps but stock up {stock_return_30d:.0%}"
        elif cds_spread_change_30d < -50 and stock_return_30d < 0:
            direction = "bullish"
            confidence = 0.68
            desc = f"CREDIT LEADING: {ticker} CDS tightening {cds_spread_change_30d:.0f}bps but stock down {stock_return_30d:.0%}"
        else:
            return None

        return CrossAssetSignal(
            signal_id=f"cred_{hashlib.sha256(f'{ticker}credit'.encode()).hexdigest()[:8]}",
            signal_type="credit_equity_divergence",
            primary_asset=ticker,
            related_assets=[f"{ticker}_CDS", f"{ticker}_BONDS"],
            direction=direction,
            confidence=confidence,
            description=desc,
            divergence_data={"stock_return": stock_return_30d, "cds_change": cds_spread_change_30d},
        )

    def currency_equity_correlation_break(
        self,
        ticker: str,
        foreign_revenue_pct: float,
        primary_currency_exposure: str,  # e.g., "EUR", "JPY"
        currency_move_30d: float,  # vs USD
        stock_move_30d: float,
        expected_correlation: float,
    ) -> Optional[CrossAssetSignal]:
        """Currency exposure often unpriced in equities.

        Strong foreign revenue + FX move should impact stock.
        """
        if foreign_revenue_pct < 0.30:
            return None  # Not enough exposure

        # Expected stock move based on FX
        expected_fx_impact = currency_move_30d * foreign_revenue_pct * 0.5
        actual_vs_expected = stock_move_30d - expected_fx_impact

        if abs(actual_vs_expected) < 0.05:
            return None

        if currency_move_30d > 0.03 and stock_move_30d < expected_fx_impact - 0.03:
            direction = "bullish"
            desc = f"FX UNPRICED TAILWIND: {ticker} ({foreign_revenue_pct:.0%} foreign) not pricing {primary_currency_exposure} strength"
        elif currency_move_30d < -0.03 and stock_move_30d > expected_fx_impact + 0.03:
            direction = "bearish"
            desc = f"FX UNPRICED HEADWIND: {ticker} ({foreign_revenue_pct:.0%} foreign) not pricing {primary_currency_exposure} weakness"
        else:
            return None

        return CrossAssetSignal(
            signal_id=f"fx_{hashlib.sha256(f'{ticker}{primary_currency_exposure}'.encode()).hexdigest()[:8]}",
            signal_type="currency_equity_break",
            primary_asset=ticker,
            related_assets=[f"{primary_currency_exposure}/USD"],
            direction=direction,
            confidence=0.62,
            description=desc,
            divergence_data={"fx_move": currency_move_30d, "exposure": foreign_revenue_pct, "expected": expected_fx_impact},
        )

    def commodity_input_cost_lag(
        self,
        ticker: str,
        primary_commodity: str,  # e.g., "OIL", "COPPER", "WHEAT"
        commodity_change_90d: float,
        company_margin_change: float,
        cost_as_pct_revenue: float,
        pricing_power: str,  # "high", "medium", "low"
    ) -> Optional[CrossAssetSignal]:
        """Input cost changes take time to flow through.

        High commodity move + stable margins = surprise coming.
        """
        if cost_as_pct_revenue < 0.15:
            return None  # Not material

        expected_margin_impact = -commodity_change_90d * cost_as_pct_revenue
        pricing_adjustment = {"high": 0.7, "medium": 0.5, "low": 0.2}.get(pricing_power, 0.5)
        expected_margin_impact *= (1 - pricing_adjustment)

        surprise = company_margin_change - expected_margin_impact

        if abs(surprise) < 0.02:
            return None

        if surprise > 0.03:  # Better than expected
            direction = "bullish"
            confidence = 0.65
            desc = f"COST BEAT: {ticker} margins holding despite {primary_commodity} +{commodity_change_90d:.0%}"
        elif surprise < -0.03:  # Worse than expected
            direction = "bearish"
            confidence = 0.65
            desc = f"COST CATCH-UP: {ticker} hasn't reflected {primary_commodity} {commodity_change_90d:+.0%} in margins"
        else:
            return None

        return CrossAssetSignal(
            signal_id=f"comm_{hashlib.sha256(f'{ticker}{primary_commodity}'.encode()).hexdigest()[:8]}",
            signal_type="commodity_cost_lag",
            primary_asset=ticker,
            related_assets=[primary_commodity],
            direction=direction,
            confidence=confidence,
            description=desc,
            divergence_data={"commodity": commodity_change_90d, "margin": company_margin_change, "expected": expected_margin_impact},
        )

    def treasury_equity_regime_shift(
        self,
        ticker: str,
        stock_treasury_correlation_60d: float,
        stock_treasury_correlation_historical: float,
        treasury_10y_change: float,
        stock_duration: float,  # Equity duration estimate
    ) -> Optional[CrossAssetSignal]:
        """Rate sensitivity can shift suddenly.

        Correlation regime changes signal risk.
        """
        correlation_shift = stock_treasury_correlation_60d - stock_treasury_correlation_historical

        if abs(correlation_shift) < 0.3:
            return None

        if correlation_shift > 0.3:
            desc = f"RATE REGIME: {ticker} now POSITIVELY correlated with rates (was {stock_treasury_correlation_historical:.2f}, now {stock_treasury_correlation_60d:.2f})"
            direction = "uncertain"
            confidence = 0.60
        else:
            desc = f"RATE REGIME: {ticker} correlation to rates inverted (was {stock_treasury_correlation_historical:.2f}, now {stock_treasury_correlation_60d:.2f})"
            direction = "uncertain"
            confidence = 0.58

        return CrossAssetSignal(
            signal_id=f"rate_{hashlib.sha256(f'{ticker}rate'.encode()).hexdigest()[:8]}",
            signal_type="treasury_equity_regime",
            primary_asset=ticker,
            related_assets=["TLT", "IEF", "UST10Y"],
            direction=direction,
            confidence=confidence,
            description=desc,
            divergence_data={"corr_current": stock_treasury_correlation_60d, "corr_historical": stock_treasury_correlation_historical},
        )

    def vix_term_structure_equity_divergence(
        self,
        vix_spot: float,
        vix_1m_future: float,
        vix_3m_future: float,
        spx_return_5d: float,
    ) -> Optional[CrossAssetSignal]:
        """VIX term structure tells the fear story.

        Inverted (backwardation) = immediate fear
        Steep contango + equity rally = complacency building
        """
        term_structure = (vix_3m_future - vix_spot) / vix_spot if vix_spot > 0 else 0

        if term_structure < -0.05 and spx_return_5d > 0.02:
            # Inverted VIX but market rallying = fear fading
            direction = "bullish"
            confidence = 0.62
            desc = f"VIX FEAR FADING: Term structure inverted but SPX +{spx_return_5d:.1%} in 5d"
        elif term_structure > 0.15 and spx_return_5d > 0.03:
            # Steep contango + rally = complacency
            direction = "bearish"
            confidence = 0.58
            desc = f"VIX COMPLACENCY: Steep contango ({term_structure:.0%}) during rally - correction risk"
        elif vix_spot > 30 and term_structure > 0:
            # High VIX but normal structure = near bottom
            direction = "bullish"
            confidence = 0.60
            desc = f"VIX CAPITULATION: VIX {vix_spot:.0f} but structure normalizing - fear peak"
        else:
            return None

        return CrossAssetSignal(
            signal_id=f"vix_{hashlib.sha256(b'vixterm').hexdigest()[:8]}",
            signal_type="vix_term_structure",
            primary_asset="SPY",
            related_assets=["VIX", "VX1", "VX3"],
            direction=direction,
            confidence=confidence,
            description=desc,
            divergence_data={"vix": vix_spot, "term_structure": term_structure, "spx_5d": spx_return_5d},
        )

    def sector_rotation_leading(
        self,
        leading_sectors: Dict[str, float],  # sector -> 10d return
        lagging_sectors: Dict[str, float],
        market_return_10d: float,
    ) -> List[CrossAssetSignal]:
        """Sector rotation patterns predict market direction.

        Defensive leadership = late cycle
        Cyclical leadership = early cycle
        """
        signals = []

        defensive = ["XLU", "XLP", "XLV", "XLRE"]  # Utilities, Staples, Healthcare, Real Estate
        cyclical = ["XLF", "XLI", "XLB", "XLE"]  # Financials, Industrials, Materials, Energy

        defensive_avg = sum(leading_sectors.get(s, 0) for s in defensive) / len(defensive)
        cyclical_avg = sum(leading_sectors.get(s, 0) for s in cyclical) / len(cyclical)

        rotation = cyclical_avg - defensive_avg

        if rotation > 0.03:
            signals.append(CrossAssetSignal(
                signal_id=f"rot_{hashlib.sha256(b'rot_cyc').hexdigest()[:8]}",
                signal_type="sector_rotation",
                primary_asset="SPY",
                related_assets=cyclical,
                direction="bullish",
                confidence=0.62,
                description=f"CYCLICAL ROTATION: Cyclicals +{cyclical_avg:.1%} vs Defensive +{defensive_avg:.1%} - risk-on",
                divergence_data={"cyclical": cyclical_avg, "defensive": defensive_avg, "spread": rotation},
            ))
        elif rotation < -0.03:
            signals.append(CrossAssetSignal(
                signal_id=f"rot_{hashlib.sha256(b'rot_def').hexdigest()[:8]}",
                signal_type="sector_rotation",
                primary_asset="SPY",
                related_assets=defensive,
                direction="bearish",
                confidence=0.60,
                description=f"DEFENSIVE ROTATION: Defensive +{defensive_avg:.1%} vs Cyclicals +{cyclical_avg:.1%} - risk-off",
                divergence_data={"cyclical": cyclical_avg, "defensive": defensive_avg, "spread": rotation},
            ))

        return signals

    def factor_crowding_detector(
        self,
        factor_name: str,  # "momentum", "value", "quality", "low_vol"
        factor_return_ytd: float,
        factor_valuation_z: float,  # Z-score of factor valuation
        short_interest_top_decile: float,
    ) -> Optional[CrossAssetSignal]:
        """Factor crowding creates unwind risk.

        Strong performance + extreme valuation + high SI = crowded
        """
        crowding_score = (
            (factor_return_ytd / 0.20) * 0.3 +
            (factor_valuation_z / 2) * 0.4 +
            (short_interest_top_decile / 0.20) * 0.3
        )

        if crowding_score < 0.7:
            return None

        return CrossAssetSignal(
            signal_id=f"crowd_{hashlib.sha256(f'{factor_name}'.encode()).hexdigest()[:8]}",
            signal_type="factor_crowding",
            primary_asset=f"FACTOR_{factor_name.upper()}",
            related_assets=[],
            direction="bearish",
            confidence=min(0.55 + crowding_score * 0.15, 0.75),
            description=f"FACTOR CROWDED: {factor_name} factor crowded (score {crowding_score:.2f}) - unwind risk",
            divergence_data={"return": factor_return_ytd, "valuation_z": factor_valuation_z, "crowding": crowding_score},
        )

    def liquidity_regime_shift(
        self,
        bid_ask_spread_change: float,
        market_depth_change: float,
        etf_premium_discount: float,
        correlation_to_spy: float,
    ) -> Optional[CrossAssetSignal]:
        """Liquidity regime shifts precede volatility.

        Widening spreads + declining depth + ETF discounts = stress
        """
        liquidity_stress = (
            max(0, bid_ask_spread_change) * 0.35 +
            max(0, -market_depth_change) * 0.35 +
            max(0, -etf_premium_discount) * 0.30
        )

        if liquidity_stress < 0.10:
            return None

        if liquidity_stress > 0.25:
            severity = "CRITICAL"
            confidence = 0.72
        elif liquidity_stress > 0.15:
            severity = "HIGH"
            confidence = 0.65
        else:
            severity = "ELEVATED"
            confidence = 0.58

        return CrossAssetSignal(
            signal_id=f"liq_{hashlib.sha256(b'liquidity').hexdigest()[:8]}",
            signal_type="liquidity_regime",
            primary_asset="MARKET",
            related_assets=["SPY", "HYG", "LQD"],
            direction="bearish",
            confidence=confidence,
            description=f"{severity} LIQUIDITY STRESS: Spreads +{bid_ask_spread_change:.0%}, depth {market_depth_change:.0%}",
            divergence_data={"spread_chg": bid_ask_spread_change, "depth_chg": market_depth_change, "stress": liquidity_stress},
        )

    def get_stats(self) -> Dict[str, Any]:
        return {"total_signals": len(self.signals_detected)}


