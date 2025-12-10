"""================================================================================
SUPPLY CHAIN & REAL ECONOMY SIGNALS
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

Track the physical economy before it hits financial statements.
================================================================================
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SupplyChainSignal:
    """Signal from supply chain/real economy."""

    signal_id: str
    signal_type: str
    ticker: str
    direction: str
    confidence: float
    description: str
    physical_evidence: str
    lead_time_days: int  # How far ahead this signal typically leads
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {k: v.isoformat() if isinstance(v, datetime) else v for k, v in self.__dict__.items()}


class SupplyChainSignals:
    """SUPPLY CHAIN & REAL ECONOMY SIGNALS

    1. Credit Card Transaction Velocity - Real-time revenue proxy
    2. Web Traffic Second Derivative - App Annie/SimilarWeb
    3. Supply Chain Disruption Propagator - Tier-2 supplier stress
    4. Container Ship Tracking - Import/export real-time
    5. Utility Usage Industrial Proxy - Power = production
    6. Supplier Payment Terms Change - Cash flow stress indicator
    7. Inventory Channel Check - Distributor inventory levels
    8. Hiring Freeze Detection - LinkedIn job posting gaps
    9. Customer Concentration Risk - Revenue concentration changes
    10. Capex Leading Indicator - Equipment orders vs guidance
    """

    def __init__(self):
        self.signals_detected: List[SupplyChainSignal] = []

    def credit_card_transaction_velocity(
        self,
        ticker: str,
        current_transaction_growth: float,
        previous_transaction_growth: float,
        same_store_sales_reported: float = None,
    ) -> Optional[SupplyChainSignal]:
        """Credit card data is real-time revenue proxy.

        Divergence from reported same-store-sales = early warning.
        """
        velocity_change = current_transaction_growth - previous_transaction_growth

        if same_store_sales_reported is not None:
            divergence = current_transaction_growth - same_store_sales_reported

            if abs(divergence) > 0.05:
                direction = "bullish" if divergence > 0 else "bearish"

                return SupplyChainSignal(
                    signal_id=f"cc_{hashlib.sha256(f'{ticker}cc'.encode()).hexdigest()[:8]}",
                    signal_type="credit_card_divergence",
                    ticker=ticker,
                    direction=direction,
                    confidence=0.72,
                    description=f"CC DIVERGENCE: {ticker} card data shows {current_transaction_growth:+.1%} vs reported {same_store_sales_reported:+.1%}",
                    physical_evidence=f"Divergence: {divergence:+.1%}",
                    lead_time_days=30,
                )

        if abs(velocity_change) > 0.10:
            direction = "bullish" if velocity_change > 0 else "bearish"

            return SupplyChainSignal(
                signal_id=f"cc_{hashlib.sha256(f'{ticker}vel'.encode()).hexdigest()[:8]}",
                signal_type="credit_card_velocity",
                ticker=ticker,
                direction=direction,
                confidence=0.65,
                description=f"CC VELOCITY: {ticker} transaction growth {'accelerating' if velocity_change > 0 else 'decelerating'} ({velocity_change:+.1%})",
                physical_evidence=f"Growth: {previous_transaction_growth:.1%} → {current_transaction_growth:.1%}",
                lead_time_days=21,
            )

        return None

    def web_traffic_second_derivative(
        self,
        ticker: str,
        current_traffic: float,
        traffic_30d_ago: float,
        traffic_60d_ago: float,
    ) -> Optional[SupplyChainSignal]:
        """Web traffic second derivative = acceleration/deceleration.

        More leading than traffic level itself.
        """
        if traffic_60d_ago == 0 or traffic_30d_ago == 0:
            return None

        first_deriv_recent = (current_traffic - traffic_30d_ago) / traffic_30d_ago
        first_deriv_prior = (traffic_30d_ago - traffic_60d_ago) / traffic_60d_ago
        second_deriv = first_deriv_recent - first_deriv_prior

        if abs(second_deriv) < 0.05:
            return None

        direction = "bullish" if second_deriv > 0 else "bearish"

        return SupplyChainSignal(
            signal_id=f"web_{hashlib.sha256(f'{ticker}web'.encode()).hexdigest()[:8]}",
            signal_type="web_traffic_acceleration",
            ticker=ticker,
            direction=direction,
            confidence=0.62,
            description=f"WEB TRAFFIC: {ticker} traffic growth {'ACCELERATING' if second_deriv > 0 else 'DECELERATING'} (2nd deriv: {second_deriv:+.1%})",
            physical_evidence=f"First deriv: {first_deriv_prior:.1%} → {first_deriv_recent:.1%}",
            lead_time_days=45,
        )

    def supply_chain_disruption_propagator(
        self,
        ticker: str,
        tier1_supplier_stress: List[Dict],
        tier2_supplier_stress: List[Dict],
    ) -> Optional[SupplyChainSignal]:
        """Track tier-2 supplier stress before it propagates.

        Tier 2 stress today = Tier 1 stress in 30 days = Company stress in 60 days.
        """
        tier2_stress_count = len([s for s in tier2_supplier_stress if s.get("stress_level", "") == "high"])
        tier1_stress_count = len([s for s in tier1_supplier_stress if s.get("stress_level", "") == "high"])

        if tier2_stress_count >= 3 and tier1_stress_count < 2:
            return SupplyChainSignal(
                signal_id=f"sc_{hashlib.sha256(f'{ticker}supply'.encode()).hexdigest()[:8]}",
                signal_type="supply_chain_propagation",
                ticker=ticker,
                direction="bearish",
                confidence=0.68,
                description=f"SUPPLY RISK: {ticker} has {tier2_stress_count} stressed Tier-2 suppliers - propagation coming",
                physical_evidence=f"Tier-2 stressed: {tier2_stress_count}, Tier-1 stressed: {tier1_stress_count}",
                lead_time_days=60,
            )

        return None

    def container_ship_tracking(
        self,
        ticker: str,
        inbound_containers_change: float,
        outbound_containers_change: float,
        is_importer: bool = True,
    ) -> Optional[SupplyChainSignal]:
        """Container tracking = real-time import/export signal.
        """
        if is_importer:
            relevant_change = inbound_containers_change
            container_type = "inbound"
        else:
            relevant_change = outbound_containers_change
            container_type = "outbound"

        if abs(relevant_change) < 0.15:
            return None

        direction = "bullish" if relevant_change > 0 else "bearish"

        return SupplyChainSignal(
            signal_id=f"ship_{hashlib.sha256(f'{ticker}ship'.encode()).hexdigest()[:8]}",
            signal_type="container_tracking",
            ticker=ticker,
            direction=direction,
            confidence=0.65,
            description=f"CONTAINER: {ticker} {container_type} containers {'+' if relevant_change > 0 else ''}{relevant_change:.0%}",
            physical_evidence=f"In: {inbound_containers_change:+.0%}, Out: {outbound_containers_change:+.0%}",
            lead_time_days=45,
        )

    def utility_usage_industrial_proxy(
        self,
        ticker: str,
        facility_power_usage: float,
        baseline_power_usage: float,
        is_manufacturing: bool = True,
    ) -> Optional[SupplyChainSignal]:
        """Power consumption = production activity.

        More accurate than reported production numbers.
        """
        if not is_manufacturing:
            return None

        if baseline_power_usage == 0:
            return None

        usage_ratio = facility_power_usage / baseline_power_usage
        deviation = usage_ratio - 1.0

        if abs(deviation) < 0.10:
            return None

        direction = "bullish" if deviation > 0 else "bearish"
        activity = "ramping" if deviation > 0 else "slowing"

        return SupplyChainSignal(
            signal_id=f"util_{hashlib.sha256(f'{ticker}util'.encode()).hexdigest()[:8]}",
            signal_type="utility_usage_proxy",
            ticker=ticker,
            direction=direction,
            confidence=0.70,
            description=f"UTILITY SIGNAL: {ticker} facilities {activity} ({deviation:+.0%} vs baseline)",
            physical_evidence=f"Power usage: {usage_ratio:.0%} of normal",
            lead_time_days=30,
        )

    def supplier_payment_terms_change(
        self,
        ticker: str,
        current_days_payable: int,
        previous_days_payable: int,
        industry_average: int,
    ) -> Optional[SupplyChainSignal]:
        """Extending payment terms = cash flow stress.
        Shortening = either flush with cash or suppliers demanding.
        """
        change = current_days_payable - previous_days_payable
        vs_industry = current_days_payable - industry_average

        if abs(change) < 5:
            return None

        if change > 10:
            direction = "bearish"
            confidence = 0.68
            desc = f"PAYMENT STRETCH: {ticker} extending supplier payments by {change} days"
            evidence = "Cash flow stress indicator"
        elif change < -10 and vs_industry < -5:
            direction = "bullish"
            confidence = 0.58
            desc = f"PAYMENT SHRINK: {ticker} paying suppliers {abs(change)} days faster"
            evidence = "Strong cash position or supplier relationship"
        else:
            return None

        return SupplyChainSignal(
            signal_id=f"pay_{hashlib.sha256(f'{ticker}pay'.encode()).hexdigest()[:8]}",
            signal_type="supplier_payment_terms",
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            description=desc,
            physical_evidence=evidence + f" (DPO: {previous_days_payable} → {current_days_payable} days)",
            lead_time_days=60,
        )

    def inventory_channel_check(
        self,
        ticker: str,
        distributor_inventory_weeks: float,
        normal_inventory_weeks: float,
        sell_through_rate: float,
    ) -> Optional[SupplyChainSignal]:
        """Distributor inventory levels reveal demand reality.

        High inventory + low sell-through = demand problem
        Low inventory + high sell-through = under-shipping demand
        """
        inventory_ratio = distributor_inventory_weeks / normal_inventory_weeks if normal_inventory_weeks > 0 else 1

        if inventory_ratio > 1.5 and sell_through_rate < 0.8:
            direction = "bearish"
            confidence = 0.72
            desc = f"CHANNEL STUFFING: {ticker} distributor inventory {inventory_ratio:.1f}x normal, weak sell-through"
        elif inventory_ratio < 0.6 and sell_through_rate > 1.1:
            direction = "bullish"
            confidence = 0.68
            desc = f"CHANNEL LEAN: {ticker} inventory tight ({inventory_ratio:.1f}x normal), strong sell-through"
        else:
            return None

        return SupplyChainSignal(
            signal_id=f"inv_{hashlib.sha256(f'{ticker}inv'.encode()).hexdigest()[:8]}",
            signal_type="inventory_channel_check",
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            description=desc,
            physical_evidence=f"Inventory weeks: {distributor_inventory_weeks:.1f} (normal: {normal_inventory_weeks:.1f})",
            lead_time_days=45,
        )

    def hiring_freeze_detection(
        self,
        ticker: str,
        current_job_postings: int,
        avg_job_postings: int,
        key_departments_affected: List[str],
    ) -> Optional[SupplyChainSignal]:
        """Gap in job postings = hiring freeze.

        Which departments matter: Sales = revenue worry, R&D = innovation stall.
        """
        if avg_job_postings == 0:
            return None

        posting_ratio = current_job_postings / avg_job_postings

        if posting_ratio > 0.5:
            return None  # Not a significant drop

        critical_depts = {"sales", "engineering", "r&d", "research", "finance"}
        critical_affected = [d for d in key_departments_affected if d.lower() in critical_depts]

        confidence = 0.60 + len(critical_affected) * 0.05

        return SupplyChainSignal(
            signal_id=f"hire_{hashlib.sha256(f'{ticker}hire'.encode()).hexdigest()[:8]}",
            signal_type="hiring_freeze",
            ticker=ticker,
            direction="bearish",
            confidence=min(confidence, 0.78),
            description=f"HIRING FREEZE: {ticker} postings down {(1-posting_ratio):.0%}, affecting {', '.join(key_departments_affected)}",
            physical_evidence=f"Postings: {current_job_postings} vs normal {avg_job_postings}",
            lead_time_days=60,
        )

    def capex_leading_indicator(
        self,
        ticker: str,
        equipment_orders: float,
        capex_guidance: float,
        previous_equipment_orders: float,
    ) -> Optional[SupplyChainSignal]:
        """Equipment orders vs capex guidance reveals true investment plans.

        Orders >> Guidance = sandbagging
        Orders << Guidance = capex cut coming
        """
        if capex_guidance == 0:
            return None

        order_to_guidance = equipment_orders / capex_guidance
        order_change = (equipment_orders - previous_equipment_orders) / previous_equipment_orders if previous_equipment_orders > 0 else 0

        if order_to_guidance > 1.3:
            direction = "bullish"
            confidence = 0.65
            desc = f"CAPEX SANDBAG: {ticker} equipment orders {order_to_guidance:.0%} of guidance - upside to capex"
        elif order_to_guidance < 0.6:
            direction = "bearish"
            confidence = 0.70
            desc = f"CAPEX CUT COMING: {ticker} equipment orders only {order_to_guidance:.0%} of guidance"
        else:
            return None

        return SupplyChainSignal(
            signal_id=f"capx_{hashlib.sha256(f'{ticker}capx'.encode()).hexdigest()[:8]}",
            signal_type="capex_leading",
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            description=desc,
            physical_evidence=f"Orders: ${equipment_orders/1e6:.1f}M vs Guidance: ${capex_guidance/1e6:.1f}M",
            lead_time_days=90,
        )

    def get_stats(self) -> Dict[str, Any]:
        return {"total_signals": len(self.signals_detected)}


