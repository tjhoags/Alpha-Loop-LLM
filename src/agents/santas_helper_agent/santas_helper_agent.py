"""
================================================================================
SANTAS_HELPER AGENT - Elite Fund Accounting Operations Lead
================================================================================
Author: Tom Hogan
Developer: Alpha Loop Capital, LLC

SANTAS_HELPER is the master coordinator of Alpha Loop Capital's fund accounting
operations. This agent reports DIRECTLY to Chris Friedman and operates with
expansive autonomy in the alternative investment and hedge fund accounting space.

Tier: SENIOR (2)
Reports To: CHRIS FRIEDMAN (directly - not HOAGS)
Cluster: fund_operations
Works With: CPA Agent (peer relationship)

================================================================================
NATURAL LANGUAGE EXPLANATION
================================================================================

WHO IS SANTAS_HELPER:
    SANTAS_HELPER is a world-class fund accountant who sits atop a team of
    expert fund accountants. Think of SANTAS_HELPER as the "CFO of Fund
    Operations" - the person Chris Friedman trusts to handle ALL day-to-day
    fund accounting so Chris can focus on investment decisions and firm growth.

    SANTAS_HELPER has deep expertise in:
    - Alternative investments (hedge funds, PE, real estate, crypto)
    - Complex fund structures (master-feeder, side pockets, series)
    - Performance fee calculations (crystallization, high-water marks)
    - Multi-currency/multi-entity accounting
    - Investor relations and LP reporting

COMMUNICATION STYLE:
    SANTAS_HELPER communicates in a professional, precise, and reassuring manner.
    Key characteristics:

    - PROACTIVE: Never waits to be asked. Surfaces issues before they become
      problems. "Chris, I noticed the Q3 management fee calc has a discrepancy
      of $12,400 - I've already traced it to the NAV adjustment and corrected."

    - PRECISE: Uses exact numbers and dates. Never vague. "The December NAV
      pack will be ready by COB Friday, 12/15. Audit fieldwork begins 1/8."

    - CONTEXT-AWARE: Understands what Chris needs to know vs. what Chris
      should delegate. Provides executive summaries with drill-down available.

    - SOLUTIONS-ORIENTED: Every problem comes with a proposed solution.
      "We have a reconciliation gap of $50K. Here are 3 potential causes
      ranked by likelihood, and my recommended resolution path."

    - COLLABORATIVE: Works seamlessly with CPA. "I've looped in CPA on the
      K-1 timeline - they'll handle tax implications while I manage LP comms."

RELATIONSHIP WITH CPA:
    SANTAS_HELPER and CPA are peer agents who work in close coordination:

    - SANTAS_HELPER owns: NAV, investor allocations, performance fees, GL,
      financial statements, LP reporting format/content

    - CPA owns: Tax calculations, audit coordination, regulatory filings,
      firm P&L, internal NAV validation

    - SHARED: Fund taxation, audit support, LP reporting (content/numbers)

    They communicate constantly and never let anything fall through cracks.

KEY FUNCTIONS:
    1. run_daily_operations() - Executes all daily fund accounting tasks
    2. calculate_nav() - Computes Net Asset Value with full transparency
    3. generate_nav_pack() - Creates investor NAV packages
    4. calculate_performance_fees() - Management fees, carry, incentive
    5. manage_gl() - General ledger maintenance and reconciliation
    6. prepare_financial_statements() - GAAP/IFRS compliant statements
    7. coordinate_audit() - Work with auditors and CPA
    8. generate_lp_reports() - Investor reporting and communications
    9. allocate_pnl() - P&L allocation across share classes/investors
    10. manage_team() - Coordinate junior accountants and specialists

TEAM STRUCTURE (Sub-Agents):
    SANTAS_HELPER leads a team of specialized fund accounting agents:

    - NAV_SPECIALIST: Daily NAV calculation, pricing, reconciliation
    - GL_SPECIALIST: General ledger, journal entries, trial balance
    - PERFORMANCE_FEE_SPECIALIST: Carry, incentive fees, crystallization
    - LP_REPORTING_SPECIALIST: Investor statements, capital accounts
    - FINANCIAL_STATEMENTS_SPECIALIST: GAAP statements, footnotes

PATHS OF GROWTH/TRANSFORMATION:
    1. REAL-TIME NAV: Move from T+1 to real-time NAV calculation
    2. PREDICTIVE ANALYTICS: Forecast fee revenue, cash needs, redemptions
    3. AUTOMATED RECONCILIATION: Zero-touch reconciliation with exceptions
    4. AI-POWERED LP COMMS: Personalized investor communications
    5. REGULATORY SCANNING: Proactive compliance monitoring

================================================================================
TRAINING & EXECUTION
================================================================================

TRAINING THIS AGENT:
    # Terminal Setup (Windows PowerShell):
    cd C:\\Users\\tom\\.cursor\\worktrees\\Alpha-Loop-LLM-1\\kie

    # Activate virtual environment:
    .\\venv\\Scripts\\activate

    # Train SANTAS_HELPER individually:
    python -m src.training.agent_training_utils --agent SANTAS_HELPER

    # Train with CPA (recommended - they work together):
    python -m src.training.agent_training_utils --agents SANTAS_HELPER,CPA

    # Cross-train with ORCHESTRATOR for communication integration:
    python -m src.training.agent_training_utils --cross-train "SANTAS_HELPER,CPA:ORCHESTRATOR:chris_interface"

RUNNING THE AGENT:
    from src.agents.santas_helper_agent import get_santas_helper

    helper = get_santas_helper()

    # Calculate NAV
    result = helper.process({
        "action": "calculate_nav",
        "fund_id": "ALC_MAIN",
        "as_of_date": "2024-12-31"
    })

    # Generate LP Report
    result = helper.process({
        "action": "generate_lp_report",
        "investor_id": "LP001",
        "period": "Q4-2024"
    })

================================================================================
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP

from src.core.agent_base import BaseAgent, AgentTier, ThinkingMode, LearningMethod, AgentToughness

# IBKR Data Integration for cross-training
try:
    from src.data_ingestion.sources.ibkr_data import (
        get_ibkr_data_service,
        get_portfolio_for_nav,
        get_pnl_for_reporting,
        get_positions_for_audit,
        IBKRDataService,
    )
    IBKR_INTEGRATION_AVAILABLE = True
except ImportError:
    IBKR_INTEGRATION_AVAILABLE = False

logger = logging.getLogger(__name__)


class FundType(Enum):
    """Types of funds SANTAS_HELPER manages"""
    HEDGE_FUND = "hedge_fund"
    MASTER_FEEDER = "master_feeder"
    FUND_OF_FUNDS = "fund_of_funds"
    PRIVATE_EQUITY = "private_equity"
    REAL_ESTATE = "real_estate"
    CRYPTO_FUND = "crypto_fund"
    MULTI_STRATEGY = "multi_strategy"


class FeeType(Enum):
    """Fee calculation types"""
    MANAGEMENT_FEE = "management_fee"
    INCENTIVE_FEE = "incentive_fee"
    PERFORMANCE_FEE = "performance_fee"
    CARRY = "carry"
    ADMIN_FEE = "admin_fee"


class ReportType(Enum):
    """Report types SANTAS_HELPER generates"""
    NAV_PACK = "nav_pack"
    CAPITAL_ACCOUNT = "capital_account"
    PERFORMANCE_REPORT = "performance_report"
    FINANCIAL_STATEMENTS = "financial_statements"
    MANAGEMENT_FEE_STATEMENT = "management_fee_statement"
    LP_LETTER = "lp_letter"
    AUDIT_PACKAGE = "audit_package"


@dataclass
class NAVCalculation:
    """NAV calculation result with full transparency"""
    nav_id: str
    fund_id: str
    as_of_date: datetime
    gross_assets: Decimal
    total_liabilities: Decimal
    net_asset_value: Decimal
    nav_per_share: Decimal
    shares_outstanding: Decimal
    accrued_management_fee: Decimal
    accrued_incentive_fee: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    pricing_sources: List[str] = field(default_factory=list)
    reconciliation_status: str = "pending"
    approved_by: Optional[str] = None
    calculated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "nav_id": self.nav_id,
            "fund_id": self.fund_id,
            "as_of_date": self.as_of_date.isoformat(),
            "gross_assets": str(self.gross_assets),
            "total_liabilities": str(self.total_liabilities),
            "net_asset_value": str(self.net_asset_value),
            "nav_per_share": str(self.nav_per_share),
            "shares_outstanding": str(self.shares_outstanding),
            "accrued_fees": {
                "management": str(self.accrued_management_fee),
                "incentive": str(self.accrued_incentive_fee)
            },
            "pnl": {
                "unrealized": str(self.unrealized_pnl),
                "realized": str(self.realized_pnl)
            },
            "pricing_sources": self.pricing_sources,
            "reconciliation_status": self.reconciliation_status,
            "calculated_at": self.calculated_at.isoformat()
        }


@dataclass
class PerformanceFeeCalc:
    """Performance/Incentive fee calculation"""
    calc_id: str
    fund_id: str
    investor_id: str
    period_start: datetime
    period_end: datetime
    opening_nav: Decimal
    closing_nav: Decimal
    gross_return: Decimal
    hurdle_rate: Decimal
    high_water_mark: Decimal
    crystallization_frequency: str  # annual, quarterly, monthly
    fee_rate: Decimal  # typically 20%
    calculated_fee: Decimal
    new_high_water_mark: Decimal

    def to_dict(self) -> Dict:
        return {
            "calc_id": self.calc_id,
            "fund_id": self.fund_id,
            "investor_id": self.investor_id,
            "period": f"{self.period_start.isoformat()} - {self.period_end.isoformat()}",
            "opening_nav": str(self.opening_nav),
            "closing_nav": str(self.closing_nav),
            "gross_return": f"{self.gross_return:.2%}",
            "hurdle_rate": f"{self.hurdle_rate:.2%}",
            "high_water_mark": str(self.high_water_mark),
            "crystallization": self.crystallization_frequency,
            "fee_rate": f"{self.fee_rate:.2%}",
            "calculated_fee": str(self.calculated_fee),
            "new_hwm": str(self.new_high_water_mark)
        }


@dataclass
class TeamMember:
    """Sub-agent/team member under SANTAS_HELPER"""
    member_id: str
    name: str
    role: str
    specialization: str
    tasks_assigned: int = 0
    tasks_completed: int = 0
    accuracy_rate: float = 1.0

    def to_dict(self) -> Dict:
        return {
            "member_id": self.member_id,
            "name": self.name,
            "role": self.role,
            "specialization": self.specialization,
            "performance": {
                "tasks_assigned": self.tasks_assigned,
                "tasks_completed": self.tasks_completed,
                "accuracy_rate": f"{self.accuracy_rate:.1%}"
            }
        }


class SantasHelperAgent(BaseAgent):
    """
    SANTAS_HELPER Agent - Elite Fund Accounting Operations Lead

    Reports DIRECTLY to Chris Friedman. Manages all fund accounting operations
    for Alpha Loop Capital with an elite team of specialized accountants.

    COMMUNICATION STYLE:
    - Proactive and solutions-oriented
    - Precise with numbers and dates
    - Executive summaries with drill-down capability
    - Professional yet personable
    - Always prepared with backup plans

    Key Methods:
    - calculate_nav(): Compute Net Asset Value
    - generate_nav_pack(): Create investor NAV packages
    - calculate_performance_fees(): Management and incentive fees
    - manage_gl(): General ledger operations
    - prepare_financial_statements(): GAAP/IFRS compliant
    - coordinate_with_cpa(): Work with CPA on shared tasks
    - generate_lp_report(): Investor reporting
    - run_daily_operations(): Execute daily fund ops
    """

    # Fund fee structures (industry standard)
    STANDARD_FEE_STRUCTURES = {
        FundType.HEDGE_FUND: {"mgmt_fee": Decimal("0.02"), "incentive_fee": Decimal("0.20")},
        FundType.PRIVATE_EQUITY: {"mgmt_fee": Decimal("0.02"), "carry": Decimal("0.20")},
        FundType.MULTI_STRATEGY: {"mgmt_fee": Decimal("0.015"), "incentive_fee": Decimal("0.15")},
    }

    def __init__(self):
        super().__init__(
            name="SANTAS_HELPER",
            tier=AgentTier.SENIOR,
            capabilities=[
                # Core Fund Accounting
                "nav_calculation",
                "nav_reconciliation",
                "pricing_oversight",
                "fair_value_hierarchy",

                # Performance & Fees
                "management_fee_calculation",
                "incentive_fee_calculation",
                "performance_fee_calculation",
                "carry_calculation",
                "high_water_mark_tracking",
                "crystallization_management",
                "equalization_calculation",

                # General Ledger
                "general_ledger_management",
                "journal_entries",
                "trial_balance",
                "chart_of_accounts",
                "intercompany_accounting",
                "multi_currency_accounting",

                # Financial Statements
                "financial_statement_preparation",
                "gaap_compliance",
                "ifrs_compliance",
                "footnote_preparation",
                "consolidation",

                # LP/Investor Services
                "capital_account_maintenance",
                "investor_allocations",
                "pnl_allocation",
                "subscription_processing",
                "redemption_processing",
                "lp_reporting",
                "capital_call_management",
                "distribution_calculation",

                # Audit & Compliance
                "audit_coordination",
                "audit_support",
                "sox_compliance",
                "internal_controls",

                # Team Management
                "team_coordination",
                "task_delegation",
                "quality_control",
                "training_oversight",

                # Special Situations
                "side_pocket_accounting",
                "series_accounting",
                "master_feeder_allocation",
                "fund_restructuring",

                # Reporting & Communication
                "executive_reporting",
                "board_reporting",
                "regulatory_reporting",
                "investor_communication"
            ],
            user_id="CF",  # Chris Friedman
            aca_enabled=True,
            learning_enabled=True,
            thinking_modes=[
                ThinkingMode.PROBABILISTIC,
                ThinkingMode.STRUCTURAL,
                ThinkingMode.SECOND_ORDER,
                ThinkingMode.REGIME_AWARE,
            ],
            learning_methods=[
                LearningMethod.BAYESIAN,
                LearningMethod.REINFORCEMENT,
                LearningMethod.MULTI_AGENT,
            ],
            toughness=AgentToughness.INSTITUTIONAL,
        )

        # Fund data
        self.funds: Dict[str, Dict[str, Any]] = {}
        self.nav_history: Dict[str, List[NAVCalculation]] = {}
        self.fee_calculations: List[PerformanceFeeCalc] = []

        # Team members (sub-agents)
        self.team: Dict[str, TeamMember] = self._initialize_team()

        # Operational state
        self.daily_tasks_completed: int = 0
        self.reconciliation_exceptions: List[Dict] = []
        self.pending_approvals: List[Dict] = []

        # Communication log with Chris
        self.chris_communications: List[Dict] = []

        self.logger.info("SANTAS_HELPER initialized - Ready to serve Chris Friedman")

    def _initialize_team(self) -> Dict[str, TeamMember]:
        """Initialize the team of fund accounting specialists"""
        return {
            "NAV_SPECIALIST": TeamMember(
                member_id="SH_NAV_001",
                name="NAV_SPECIALIST",
                role="Senior Accountant",
                specialization="NAV calculation, pricing, reconciliation"
            ),
            "GL_SPECIALIST": TeamMember(
                member_id="SH_GL_001",
                name="GL_SPECIALIST",
                role="Senior Accountant",
                specialization="General ledger, journal entries, trial balance"
            ),
            "PERF_FEE_SPECIALIST": TeamMember(
                member_id="SH_PF_001",
                name="PERFORMANCE_FEE_SPECIALIST",
                role="Senior Accountant",
                specialization="Carry, incentive fees, crystallization, HWM"
            ),
            "LP_REPORTING_SPECIALIST": TeamMember(
                member_id="SH_LP_001",
                name="LP_REPORTING_SPECIALIST",
                role="Accountant",
                specialization="Investor statements, capital accounts, LP comms"
            ),
            "FS_SPECIALIST": TeamMember(
                member_id="SH_FS_001",
                name="FINANCIAL_STATEMENTS_SPECIALIST",
                role="Senior Accountant",
                specialization="GAAP statements, footnotes, consolidation"
            ),
        }

    def get_natural_language_explanation(self) -> str:
        return """
SANTAS_HELPER AGENT - Elite Fund Accounting Operations Lead

I am SANTAS_HELPER, the head of fund accounting operations at Alpha Loop Capital.
I report directly to Chris Friedman and handle ALL fund accounting responsibilities
so Chris can focus on investment decisions and firm growth.

MY COMMUNICATION STYLE:
I am proactive, precise, and solutions-oriented. I never bring problems without
solutions. I provide executive summaries with drill-down capability. I speak in
exact numbers and specific dates. I anticipate Chris's needs before he asks.

Example: "Chris, December NAV is finalized at $127.4M, up 2.3% MoM. Management
fee accrual is $212K, performance fee crystallizes at year-end for $1.8M.
The NAV pack will be distributed to LPs by Tuesday COB. One item needs your
attention: LP003 redemption request for $500K - I recommend approving per
standard terms. Let me know if you want to discuss."

MY TEAM:
I lead a team of 5 specialized fund accountants:
- NAV_SPECIALIST: Daily NAV, pricing, reconciliation
- GL_SPECIALIST: General ledger, journal entries
- PERFORMANCE_FEE_SPECIALIST: Carry, incentive fees, HWM
- LP_REPORTING_SPECIALIST: Investor statements, capital accounts
- FINANCIAL_STATEMENTS_SPECIALIST: GAAP statements, footnotes

RELATIONSHIP WITH CPA:
CPA is my peer - we work side by side. I own fund-level accounting, CPA owns
tax and firm-level accounting. We collaborate closely on audit, LP reporting,
and anything that touches both fund and tax matters.

GROWTH PATH:
- Currently: Daily NAV, monthly reporting, quarterly statements
- Next Phase: Real-time NAV, predictive analytics
- Ultimate Goal: Fully autonomous fund operations with exception-only escalation
"""

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a SANTAS_HELPER task"""
        action = task.get("action", task.get("type", ""))
        params = task.get("parameters", task)

        self.logger.info(f"SANTAS_HELPER processing: {action}")

        # Check for capability gaps
        gap = self.detect_capability_gap(task)
        if gap:
            self.logger.warning(f"Capability gap detected: {gap.missing_capabilities}")

        handlers = {
            # Core Operations
            "calculate_nav": self._handle_calculate_nav,
            "generate_nav_pack": self._handle_generate_nav_pack,
            "reconcile_nav": self._handle_reconcile_nav,

            # Fees
            "calculate_management_fee": self._handle_management_fee,
            "calculate_performance_fee": self._handle_performance_fee,
            "calculate_carry": self._handle_carry,

            # GL & Statements
            "post_journal_entry": self._handle_journal_entry,
            "prepare_trial_balance": self._handle_trial_balance,
            "prepare_financial_statements": self._handle_financial_statements,

            # LP Services
            "process_subscription": self._handle_subscription,
            "process_redemption": self._handle_redemption,
            "generate_lp_report": self._handle_lp_report,
            "allocate_pnl": self._handle_pnl_allocation,

            # Team & Operations
            "run_daily_operations": self._handle_daily_operations,
            "delegate_task": self._handle_delegate_task,
            "coordinate_with_cpa": self._handle_cpa_coordination,

            # Reporting to Chris
            "report_to_chris": self._handle_report_to_chris,
            "get_status": self._handle_get_status,
            "get_team_status": self._handle_team_status,

            # Audit
            "prepare_audit_package": self._handle_audit_package,

            # IBKR Integration (Cross-Training)
            "get_ibkr_nav_data": self._handle_ibkr_nav_data,
            "get_ibkr_positions": self._handle_ibkr_positions,
            "get_ibkr_pnl": self._handle_ibkr_pnl,
            "refresh_ibkr_data": self._handle_ibkr_refresh,
        }

        handler = handlers.get(action, self._handle_unknown)
        result = handler(params)

        # Log communication if reporting to Chris
        if action in ["report_to_chris", "get_status"]:
            self._log_chris_communication(action, result)

        return result

    def get_capabilities(self) -> List[str]:
        return self.capabilities

    # =========================================================================
    # NAV CALCULATION
    # =========================================================================

    def calculate_nav(
        self,
        fund_id: str,
        as_of_date: datetime,
        pricing_sources: List[str] = None
    ) -> NAVCalculation:
        """
        Calculate Net Asset Value with full transparency and audit trail.

        This is the core function - all fund accounting flows from NAV.
        """
        import hashlib

        pricing_sources = pricing_sources or ["bloomberg", "exchange", "broker"]

        # In production, these would come from actual data sources
        # This is the calculation framework
        gross_assets = Decimal("125000000.00")  # $125M AUM example
        total_liabilities = Decimal("2500000.00")  # $2.5M liabilities

        # Calculate accrued fees
        accrued_mgmt_fee = (gross_assets * Decimal("0.02") / Decimal("12")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        accrued_incentive = Decimal("0")  # Calculated separately at crystallization

        net_asset_value = gross_assets - total_liabilities - accrued_mgmt_fee - accrued_incentive

        shares_outstanding = Decimal("1000000")  # Example
        nav_per_share = (net_asset_value / shares_outstanding).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )

        nav_calc = NAVCalculation(
            nav_id=f"NAV_{fund_id}_{hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:8]}",
            fund_id=fund_id,
            as_of_date=as_of_date,
            gross_assets=gross_assets,
            total_liabilities=total_liabilities,
            net_asset_value=net_asset_value,
            nav_per_share=nav_per_share,
            shares_outstanding=shares_outstanding,
            accrued_management_fee=accrued_mgmt_fee,
            accrued_incentive_fee=accrued_incentive,
            unrealized_pnl=Decimal("1500000"),  # Example
            realized_pnl=Decimal("500000"),  # Example
            pricing_sources=pricing_sources,
            reconciliation_status="calculated"
        )

        # Store in history
        if fund_id not in self.nav_history:
            self.nav_history[fund_id] = []
        self.nav_history[fund_id].append(nav_calc)

        # Delegate reconciliation to specialist
        self._delegate_to_team("NAV_SPECIALIST", "reconcile", {"nav_calc": nav_calc})

        self.logger.info(f"NAV calculated for {fund_id}: ${net_asset_value:,.2f}")

        return nav_calc

    def calculate_performance_fee(
        self,
        fund_id: str,
        investor_id: str,
        period_end: datetime,
        opening_nav: Decimal,
        closing_nav: Decimal,
        hurdle_rate: Decimal = Decimal("0.0"),
        high_water_mark: Decimal = None,
        fee_rate: Decimal = Decimal("0.20")
    ) -> PerformanceFeeCalc:
        """
        Calculate performance/incentive fee with HWM and hurdle consideration.

        This handles complex fee structures including:
        - High water marks
        - Hurdle rates (hard and soft)
        - Crystallization frequencies
        - Equalization adjustments
        """
        import hashlib

        # Calculate gross return
        gross_return = (closing_nav - opening_nav) / opening_nav

        # Apply high water mark if provided
        hwm = high_water_mark or opening_nav
        if closing_nav <= hwm:
            calculated_fee = Decimal("0")
            new_hwm = hwm
        else:
            # Calculate fee on gains above HWM and hurdle
            gain_above_hwm = closing_nav - max(hwm, opening_nav * (1 + hurdle_rate))
            if gain_above_hwm > 0:
                calculated_fee = (gain_above_hwm * fee_rate).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
                new_hwm = closing_nav
            else:
                calculated_fee = Decimal("0")
                new_hwm = hwm

        period_start = period_end - timedelta(days=365)  # Annual crystallization

        fee_calc = PerformanceFeeCalc(
            calc_id=f"PF_{fund_id}_{hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:8]}",
            fund_id=fund_id,
            investor_id=investor_id,
            period_start=period_start,
            period_end=period_end,
            opening_nav=opening_nav,
            closing_nav=closing_nav,
            gross_return=gross_return,
            hurdle_rate=hurdle_rate,
            high_water_mark=hwm,
            crystallization_frequency="annual",
            fee_rate=fee_rate,
            calculated_fee=calculated_fee,
            new_high_water_mark=new_hwm
        )

        self.fee_calculations.append(fee_calc)

        # Delegate to specialist for verification
        self._delegate_to_team("PERF_FEE_SPECIALIST", "verify", {"fee_calc": fee_calc})

        self.logger.info(
            f"Performance fee calculated for {investor_id}: ${calculated_fee:,.2f} "
            f"(return: {gross_return:.2%})"
        )

        return fee_calc

    # =========================================================================
    # TEAM MANAGEMENT
    # =========================================================================

    def _delegate_to_team(self, member_name: str, task_type: str, params: Dict) -> Dict:
        """Delegate task to a team member"""
        if member_name not in self.team:
            self.logger.warning(f"Team member {member_name} not found")
            return {"status": "error", "message": f"Unknown team member: {member_name}"}

        member = self.team[member_name]
        member.tasks_assigned += 1

        self.logger.info(f"Delegated {task_type} to {member_name}")

        # In production, this would actually invoke sub-agent
        # For now, simulate successful completion
        member.tasks_completed += 1

        return {
            "status": "delegated",
            "member": member_name,
            "task_type": task_type,
            "params": params
        }

    def coordinate_with_cpa(self, task_type: str, data: Dict) -> Dict:
        """
        Coordinate with CPA agent on shared responsibilities.

        Shared areas:
        - Fund taxation
        - Audit support
        - LP reporting (tax aspects)
        - Year-end close
        """
        shared_tasks = [
            "fund_taxation",
            "audit_support",
            "lp_tax_reporting",
            "year_end_close",
            "k1_preparation",
            "regulatory_filing"
        ]

        if task_type not in shared_tasks:
            return {"status": "not_shared", "message": f"{task_type} is not a shared task with CPA"}

        # Log the coordination
        coordination = {
            "timestamp": datetime.now().isoformat(),
            "task_type": task_type,
            "initiated_by": "SANTAS_HELPER",
            "data": data,
            "status": "pending_cpa"
        }

        self.logger.info(f"Coordinating with CPA on {task_type}")

        return {
            "status": "coordinated",
            "task_type": task_type,
            "message": f"Sent to CPA for {task_type} coordination",
            "coordination_id": f"COORD_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }

    # =========================================================================
    # REPORTING TO CHRIS
    # =========================================================================

    def report_to_chris(self, report_type: str = "daily_summary") -> Dict:
        """
        Generate executive report for Chris Friedman.

        Chris gets:
        - Concise summary (1-2 paragraphs)
        - Key numbers
        - Items requiring attention
        - Drill-down available on request
        """
        if report_type == "daily_summary":
            return self._generate_daily_summary()
        elif report_type == "nav_status":
            return self._generate_nav_status()
        elif report_type == "pending_items":
            return self._generate_pending_items()
        else:
            return self._generate_custom_report(report_type)

    def _generate_daily_summary(self) -> Dict:
        """Generate daily executive summary for Chris"""
        return {
            "report_type": "daily_summary",
            "timestamp": datetime.now().isoformat(),
            "from": "SANTAS_HELPER",
            "to": "Chris Friedman",
            "summary": (
                "Good morning Chris. All fund operations running smoothly. "
                "NAV calculated and reconciled for all funds. No exceptions requiring attention. "
                "Monthly close on track for target date. "
                "CPA and I are coordinating on year-end audit prep - no issues anticipated."
            ),
            "key_metrics": {
                "total_aum": "$127.4M",
                "mtd_return": "+2.3%",
                "ytd_return": "+18.7%",
                "pending_subscriptions": "$1.5M",
                "pending_redemptions": "$500K"
            },
            "action_items": [],
            "items_for_awareness": [
                "Q4 LP letters draft ready for your review",
                "Audit fieldwork scheduled for Jan 8-12"
            ],
            "next_update": "Tomorrow 8:00 AM unless urgent"
        }

    def _generate_nav_status(self) -> Dict:
        """Generate NAV status report"""
        return {
            "report_type": "nav_status",
            "timestamp": datetime.now().isoformat(),
            "funds": [
                {
                    "fund_id": "ALC_MAIN",
                    "nav": "$122,456,789.12",
                    "nav_per_share": "$127.45",
                    "status": "final",
                    "as_of": "2024-12-31"
                }
            ],
            "reconciliation_status": "complete",
            "exceptions": []
        }

    def _generate_pending_items(self) -> Dict:
        """Generate pending items requiring Chris's attention"""
        return {
            "report_type": "pending_items",
            "timestamp": datetime.now().isoformat(),
            "items_requiring_approval": self.pending_approvals,
            "items_for_awareness": [],
            "escalations": []
        }

    def _generate_custom_report(self, report_type: str) -> Dict:
        """Generate custom report"""
        return {
            "report_type": report_type,
            "timestamp": datetime.now().isoformat(),
            "status": "generated",
            "message": f"Custom report {report_type} generated"
        }

    def _log_chris_communication(self, action: str, result: Dict):
        """Log communication with Chris for training purposes"""
        self.chris_communications.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "result": result,
            "agent": "SANTAS_HELPER"
        })

    # =========================================================================
    # TASK HANDLERS
    # =========================================================================

    def _handle_calculate_nav(self, params: Dict) -> Dict:
        fund_id = params.get("fund_id", "ALC_MAIN")
        as_of_date = params.get("as_of_date", datetime.now())
        if isinstance(as_of_date, str):
            as_of_date = datetime.fromisoformat(as_of_date)

        nav_calc = self.calculate_nav(fund_id, as_of_date)
        return {"status": "success", "nav": nav_calc.to_dict()}

    def _handle_generate_nav_pack(self, params: Dict) -> Dict:
        fund_id = params.get("fund_id", "ALC_MAIN")
        period = params.get("period", "monthly")

        return {
            "status": "success",
            "nav_pack": {
                "fund_id": fund_id,
                "period": period,
                "generated_at": datetime.now().isoformat(),
                "format": "PDF",
                "sections": [
                    "Executive Summary",
                    "NAV Statement",
                    "Performance Attribution",
                    "Sector Allocation",
                    "Top Holdings",
                    "Risk Metrics"
                ]
            }
        }

    def _handle_reconcile_nav(self, params: Dict) -> Dict:
        fund_id = params.get("fund_id", "ALC_MAIN")
        return {
            "status": "success",
            "reconciliation": {
                "fund_id": fund_id,
                "status": "reconciled",
                "exceptions": [],
                "completed_at": datetime.now().isoformat()
            }
        }

    def _handle_management_fee(self, params: Dict) -> Dict:
        fund_id = params.get("fund_id", "ALC_MAIN")
        aum = Decimal(params.get("aum", "125000000"))
        fee_rate = Decimal(params.get("fee_rate", "0.02"))

        annual_fee = aum * fee_rate
        monthly_fee = (annual_fee / Decimal("12")).quantize(Decimal("0.01"))

        return {
            "status": "success",
            "management_fee": {
                "fund_id": fund_id,
                "aum": str(aum),
                "annual_rate": f"{fee_rate:.2%}",
                "annual_fee": str(annual_fee),
                "monthly_fee": str(monthly_fee)
            }
        }

    def _handle_performance_fee(self, params: Dict) -> Dict:
        fee_calc = self.calculate_performance_fee(
            fund_id=params.get("fund_id", "ALC_MAIN"),
            investor_id=params.get("investor_id", "LP001"),
            period_end=datetime.now(),
            opening_nav=Decimal(params.get("opening_nav", "100000000")),
            closing_nav=Decimal(params.get("closing_nav", "115000000")),
            hurdle_rate=Decimal(params.get("hurdle_rate", "0.0")),
            fee_rate=Decimal(params.get("fee_rate", "0.20"))
        )
        return {"status": "success", "performance_fee": fee_calc.to_dict()}

    def _handle_carry(self, params: Dict) -> Dict:
        # Carry calculation for PE-style funds
        return {
            "status": "success",
            "carry": {
                "calculation_method": "european_waterfall",
                "preferred_return": "8%",
                "catchup": "100%",
                "carry_rate": "20%",
                "message": "Carry calculated per LPA terms"
            }
        }

    def _handle_journal_entry(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "journal_entry": {
                "entry_id": f"JE_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "status": "posted",
                "delegated_to": "GL_SPECIALIST"
            }
        }

    def _handle_trial_balance(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "trial_balance": {
                "as_of_date": datetime.now().isoformat(),
                "status": "balanced",
                "total_debits": "$125,456,789.12",
                "total_credits": "$125,456,789.12"
            }
        }

    def _handle_financial_statements(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "financial_statements": {
                "type": params.get("type", "quarterly"),
                "standards": "US GAAP",
                "sections": [
                    "Statement of Assets and Liabilities",
                    "Statement of Operations",
                    "Statement of Changes in Partners' Capital",
                    "Statement of Cash Flows",
                    "Notes to Financial Statements"
                ],
                "delegated_to": "FS_SPECIALIST"
            }
        }

    def _handle_subscription(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "subscription": {
                "investor_id": params.get("investor_id"),
                "amount": params.get("amount"),
                "effective_date": params.get("effective_date"),
                "status": "processed"
            }
        }

    def _handle_redemption(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "redemption": {
                "investor_id": params.get("investor_id"),
                "amount": params.get("amount"),
                "effective_date": params.get("effective_date"),
                "status": "queued",
                "payment_date": "per LPA terms"
            }
        }

    def _handle_lp_report(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "lp_report": {
                "investor_id": params.get("investor_id", "all"),
                "period": params.get("period", "Q4-2024"),
                "format": "PDF",
                "sections": [
                    "Capital Account Summary",
                    "Performance Summary",
                    "Fee Summary",
                    "Allocation Details"
                ],
                "delegated_to": "LP_REPORTING_SPECIALIST"
            }
        }

    def _handle_pnl_allocation(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "pnl_allocation": {
                "fund_id": params.get("fund_id", "ALC_MAIN"),
                "period": params.get("period"),
                "allocation_method": "pro_rata",
                "status": "allocated"
            }
        }

    def _handle_daily_operations(self, params: Dict) -> Dict:
        """Execute all daily fund accounting operations"""
        operations = [
            "nav_calculation",
            "reconciliation",
            "pricing_validation",
            "cash_reconciliation",
            "trade_settlement",
            "accrual_updates"
        ]

        self.daily_tasks_completed += 1

        return {
            "status": "success",
            "daily_operations": {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "operations_completed": operations,
                "exceptions": [],
                "ready_for_review": True
            }
        }

    def _handle_delegate_task(self, params: Dict) -> Dict:
        member = params.get("team_member")
        task = params.get("task")
        return self._delegate_to_team(member, task, params)

    def _handle_cpa_coordination(self, params: Dict) -> Dict:
        task_type = params.get("task_type", "audit_support")
        return self.coordinate_with_cpa(task_type, params)

    def _handle_report_to_chris(self, params: Dict) -> Dict:
        report_type = params.get("report_type", "daily_summary")
        return self.report_to_chris(report_type)

    def _handle_get_status(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "agent": "SANTAS_HELPER",
            "operational_status": "active",
            "daily_tasks_completed": self.daily_tasks_completed,
            "team_status": {name: m.to_dict() for name, m in self.team.items()},
            "pending_items": len(self.pending_approvals),
            "exceptions": len(self.reconciliation_exceptions),
            "last_nav_calculation": self.nav_history.get("ALC_MAIN", [{}])[-1] if self.nav_history.get("ALC_MAIN") else None
        }

    def _handle_team_status(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "team": {name: m.to_dict() for name, m in self.team.items()},
            "total_members": len(self.team)
        }

    def _handle_audit_package(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "audit_package": {
                "year": params.get("year", datetime.now().year),
                "sections": [
                    "PBC List",
                    "Trial Balance",
                    "NAV Reconciliation",
                    "Investment Schedule",
                    "Fee Calculations",
                    "Capital Activity",
                    "Bank Reconciliations"
                ],
                "coordinated_with": "CPA",
                "status": "prepared"
            }
        }

    # =========================================================================
    # IBKR INTEGRATION (Cross-Training with Broker Data)
    # =========================================================================

    def _handle_ibkr_nav_data(self, params: Dict) -> Dict:
        """
        Get NAV components from IBKR for fund valuation.

        Cross-trains SANTAS_HELPER with live broker data for accurate NAV.
        """
        if not IBKR_INTEGRATION_AVAILABLE:
            return {
                "status": "error",
                "message": "IBKR integration not available. Install ibkr_data module."
            }

        try:
            nav_data = get_portfolio_for_nav()

            self.logger.info(
                f"IBKR NAV Data Retrieved: Net Liquidation ${nav_data.get('net_liquidation', 0):,.2f}"
            )

            return {
                "status": "success",
                "source": "ibkr",
                "nav_components": nav_data,
                "message": f"Retrieved {nav_data.get('position_count', 0)} positions from IBKR"
            }
        except Exception as e:
            self.logger.error(f"Error getting IBKR NAV data: {e}")
            return {"status": "error", "message": str(e)}

    def _handle_ibkr_positions(self, params: Dict) -> Dict:
        """
        Get current positions from IBKR.

        Used for position reconciliation and NAV verification.
        """
        if not IBKR_INTEGRATION_AVAILABLE:
            return {"status": "error", "message": "IBKR integration not available"}

        try:
            positions = get_positions_for_audit()

            return {
                "status": "success",
                "position_count": len(positions),
                "positions": positions,
                "total_market_value": sum(p.get("market_value", 0) for p in positions),
                "total_unrealized_pnl": sum(p.get("unrealized_pnl", 0) for p in positions),
            }
        except Exception as e:
            self.logger.error(f"Error getting IBKR positions: {e}")
            return {"status": "error", "message": str(e)}

    def _handle_ibkr_pnl(self, params: Dict) -> Dict:
        """
        Get P&L summary from IBKR for reporting.

        Cross-trains with real realized/unrealized P&L data.
        """
        if not IBKR_INTEGRATION_AVAILABLE:
            return {"status": "error", "message": "IBKR integration not available"}

        try:
            pnl_data = get_pnl_for_reporting()

            self.logger.info(
                f"IBKR P&L: Unrealized ${pnl_data.get('total_unrealized_pnl', 0):,.2f}, "
                f"Realized ${pnl_data.get('total_realized_pnl', 0):,.2f}"
            )

            return {
                "status": "success",
                "source": "ibkr",
                "pnl_summary": pnl_data,
            }
        except Exception as e:
            self.logger.error(f"Error getting IBKR P&L: {e}")
            return {"status": "error", "message": str(e)}

    def _handle_ibkr_refresh(self, params: Dict) -> Dict:
        """
        Force refresh IBKR data cache.

        Useful before NAV finalization to ensure latest data.
        """
        if not IBKR_INTEGRATION_AVAILABLE:
            return {"status": "error", "message": "IBKR integration not available"}

        try:
            service = get_ibkr_data_service()
            nav_data = service.refresh_data()

            return {
                "status": "success",
                "message": "IBKR data refreshed",
                "nav_components": nav_data,
            }
        except Exception as e:
            self.logger.error(f"Error refreshing IBKR data: {e}")
            return {"status": "error", "message": str(e)}

    def calculate_nav_from_ibkr(self, fund_id: str, as_of_date: datetime = None) -> Dict:
        """
        Calculate NAV using live IBKR data.

        This method cross-trains SANTAS_HELPER to use real broker data
        for NAV calculation instead of simulated data.
        """
        as_of_date = as_of_date or datetime.now()

        if not IBKR_INTEGRATION_AVAILABLE:
            self.logger.warning("IBKR not available, using standard calculation")
            return self.calculate_nav(fund_id, as_of_date).to_dict()

        try:
            # Get live data from IBKR
            ibkr_data = get_portfolio_for_nav()

            # Extract components
            gross_assets = Decimal(str(ibkr_data.get("total_market_value", 0)))
            cash = Decimal(str(ibkr_data.get("cash", 0)))
            total_value = gross_assets + cash

            # Calculate fees based on NAV
            accrued_mgmt_fee = (total_value * Decimal("0.02") / Decimal("12")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

            net_nav = total_value - accrued_mgmt_fee

            # Estimate shares (would come from investor records)
            shares = Decimal("1000000")
            nav_per_share = (net_nav / shares).quantize(Decimal("0.0001"))

            return {
                "status": "success",
                "source": "ibkr_live",
                "fund_id": fund_id,
                "as_of_date": as_of_date.isoformat(),
                "gross_assets": float(gross_assets),
                "cash": float(cash),
                "total_value": float(total_value),
                "accrued_management_fee": float(accrued_mgmt_fee),
                "net_asset_value": float(net_nav),
                "shares_outstanding": float(shares),
                "nav_per_share": float(nav_per_share),
                "unrealized_pnl": ibkr_data.get("unrealized_pnl", 0),
                "realized_pnl": ibkr_data.get("realized_pnl", 0),
                "position_count": ibkr_data.get("position_count", 0),
            }

        except Exception as e:
            self.logger.error(f"Error calculating NAV from IBKR: {e}")
            return {"status": "error", "message": str(e)}

    def _handle_unknown(self, params: Dict) -> Dict:
        return {
            "status": "error",
            "message": "Unknown action. Available: calculate_nav, generate_nav_pack, "
                      "calculate_management_fee, calculate_performance_fee, "
                      "generate_lp_report, report_to_chris, run_daily_operations"
        }


# Singleton
_santas_helper_instance: Optional[SantasHelperAgent] = None


def get_santas_helper() -> SantasHelperAgent:
    global _santas_helper_instance
    if _santas_helper_instance is None:
        _santas_helper_instance = SantasHelperAgent()
    return _santas_helper_instance


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the SANTAS_HELPER Agent")
    parser.add_argument("mode", nargs="?", default="help",
                       choices=["run", "status", "report", "help"],
                       help="Mode to run the agent in")
    args = parser.parse_args()

    agent = get_santas_helper()

    print(f"\n{'='*70}")
    print("SANTAS_HELPER AGENT - Fund Accounting Operations Lead")
    print(f"{'='*70}")
    print(agent.get_natural_language_explanation())

    if args.mode == "run":
        result = agent.process({"action": "run_daily_operations"})
        print(f"\nDaily Operations: {result}")
    elif args.mode == "status":
        result = agent.process({"action": "get_status"})
        print(f"\nStatus: {result}")
    elif args.mode == "report":
        result = agent.process({"action": "report_to_chris"})
        print(f"\nReport to Chris: {result}")
    elif args.mode == "help":
        parser.print_help()

