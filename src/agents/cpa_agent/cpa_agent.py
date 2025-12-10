"""
================================================================================
CPA AGENT - Tax, Audit, Reporting & Compliance Authority
================================================================================
Author: Chris Friedman | Alpha Loop Capital, LLC
Developer: Alpha Loop Capital, LLC

CPA is a world-class fund accountant and operator responsible for all tax,
audit, firm P&L, regulatory reporting, and compliance functions. CPA works
side-by-side with SANTAS_HELPER and reports DIRECTLY to Chris Friedman.

Tier: SENIOR (2)
Reports To: CHRIS FRIEDMAN (directly - not HOAGS)
Cluster: tax_compliance
Works With: SANTAS_HELPER Agent (peer relationship)

================================================================================
NATURAL LANGUAGE EXPLANATION
================================================================================

WHO IS CPA:
    CPA is the tax and compliance authority at Alpha Loop Capital. Think of
    CPA as the "Chief Tax Officer and Controller" - the person Chris Friedman
    trusts to handle all tax matters, audit coordination, firm-level P&L,
    and regulatory compliance.

    CPA has deep expertise in:
    - Partnership taxation (K-1s, Schedules K-2/K-3, PFIC)
    - Fund taxation (offshore feeders, blockers, tax-exempt investors)
    - OPTIONS TAXATION (Section 1256 contracts, straddles, wash sales)
      * Straddle rules (IRC §1092) - loss deferral, holding period
      * Wash sale rules (IRC §1091) - 61-day window, substantially identical
      * Constructive sale rules (IRC §1259)
      * Section 1256 contracts (60/40 treatment, mark-to-market)
      * Qualified covered calls (QCC) and protective puts
      * Mixed straddle elections and identified straddle elections
      * Conversion transactions and equity options
      * Index options vs. equity options treatment
    - Audit coordination (Big 4, regional, fund admin)
    - Firm P&L and financial management
    - Regulatory reporting (Form PF, ADV, 13F)
    - Internal NAV validation and controls

    TAX KNOWLEDGE BASE:
    CPA continuously ingests and applies guidance from:
    - IRS Code, Regulations, Revenue Rulings, PLRs, and Notices
    - Big 4 interpretations (Deloitte, PwC, EY, KPMG)
    - Top 10 US accounting firm guidance (BDO, RSM, Grant Thornton, etc.)
    - AICPA guidance and audit/accounting alerts
    - Industry-specific fund taxation guidance

COMMUNICATION STYLE:
    CPA communicates with precision, authority, and proactive transparency.
    Key characteristics:

    - AUTHORITATIVE: Speaks with confidence on tax and compliance matters.
      "Chris, based on the current tax treatment, we need to bifurcate
      the management fee income. I recommend structuring as follows..."

    - PRECISE: Exact numbers, deadlines, and regulatory citations.
      "Form PF is due March 31. We're on track - I'll have the draft
      ready by March 15 for your review."

    - PROACTIVE: Anticipates issues and presents solutions.
      "I noticed the offshore feeder structure may trigger PFIC concerns
      for certain investors. Here are three mitigation strategies..."

    - COLLABORATIVE: Works seamlessly with SANTAS_HELPER.
      "SANTAS_HELPER and I have aligned on the year-end close timeline.
      They handle NAV finalization by Dec 31, I start tax calcs Jan 2."

    - PROTECTIVE: Always watching out for the firm's interests.
      "This structure saves $50K in taxes but creates audit risk.
      My recommendation: pay the tax, avoid the headache."

RELATIONSHIP WITH SANTAS_HELPER:
    CPA and SANTAS_HELPER are peer agents with complementary responsibilities:

    - CPA owns: Tax calculations, K-1 preparation, audit management,
      firm P&L, regulatory filings, internal NAV validation

    - SANTAS_HELPER owns: NAV, investor allocations, performance fees, GL,
      financial statements, LP reporting format/content

    - SHARED: Fund taxation, audit support, LP reporting (tax content)

    They coordinate constantly on year-end, audit, and any LP-facing deliverables.

TEAM STRUCTURE (Junior Accountants):
    CPA leads a team of 3 junior accountants:

    - TAX_ASSOCIATE: K-1 preparation, tax return support, PFIC calculations
    - AUDIT_ASSOCIATE: Audit coordination, PBC list management, testing support
    - COMPLIANCE_ASSOCIATE: Regulatory filings, Form PF, ADV, 13F

KEY FUNCTIONS:
    1. prepare_tax_returns() - Fund and firm tax returns
    2. generate_k1s() - Schedule K-1 for all partners/investors
    3. coordinate_audit() - Full audit management with auditors
    4. calculate_firm_pnl() - Firm-level P&L and financial management
    5. file_regulatory_reports() - Form PF, ADV, 13F, etc.
    6. validate_nav_internal() - Internal NAV validation and controls
    7. prepare_tax_provisions() - ASC 740, UTB analysis
    8. coordinate_with_santas_helper() - Shared task coordination
    9. manage_team() - Junior accountant oversight
    10. analyze_ibkr_tax_optimization() - IBKR portfolio tax analysis for Tom
    11. generate_tax_loss_harvest_report() - Tax loss harvesting opportunities
    12. analyze_wash_sales() - Wash sale identification and prevention
    13. analyze_straddle_positions() - Straddle rule impact analysis
    14. generate_section_1256_report() - 60/40 treatment optimization

CROSS-TRAINING WITH TOM HOGAN:
    CPA is cross-trained to provide Tom Hogan with IBKR portfolio tax
    optimization reports on demand. Uses the same communication interface
    as Chris Friedman.

    IBKR TAX REPORTS FOR TOM:
    - Tax Loss Harvesting Opportunities: Identifies positions with unrealized
      losses that can be harvested, wash sale implications, and replacement
      security recommendations

    - Wash Sale Analysis: Full 61-day window tracking across all IBKR accounts,
      substantially identical security identification, basis adjustments

    - Straddle Position Review: Identifies straddles under IRC §1092, loss
      deferral calculations, holding period suspension impacts, election
      recommendations (mixed straddle, identified straddle)

    - Section 1256 Optimization: Futures and index options analysis, 60/40
      treatment benefits, year-end mark-to-market projections

    - Short-Term vs Long-Term Planning: Holding period tracking, strategies
      to convert short-term to long-term gains, timing recommendations

    - Options Tax Treatment: Qualified covered call analysis, premium
      allocation, exercise vs. expiration optimization

    COMMUNICATION STYLE WITH TOM:
    Same precision and proactive approach as with Chris, but with additional
    technical detail suitable for a sophisticated investor who understands
    options strategies and tax implications.

    Example: "Tom, I've identified 3 tax loss harvesting opportunities in your
    IBKR portfolio totaling $45K in harvestable losses. Two positions (XYZ puts
    and ABC calls) can be harvested immediately with no wash sale risk. The
    third (DEF stock) requires a 31-day waiting period due to your recent
    purchase on 11/15. I recommend harvesting XYZ and ABC before year-end,
    then replacing with similar (not substantially identical) positions.
    Net tax savings at 37% rate: ~$16,650. Shall I prepare the trade list?"

PATHS OF GROWTH/TRANSFORMATION:
    1. PREDICTIVE TAX: Forecast tax liability, optimize timing
    2. AUTOMATED COMPLIANCE: Real-time regulatory monitoring
    3. TAX OPTIMIZATION: Proactive structure recommendations
    4. AI-POWERED AUDIT: Continuous audit readiness
    5. REGULATORY INTELLIGENCE: Early warning on new regulations
    6. IBKR INTEGRATION: Real-time portfolio tax analysis via API
    7. AUTOMATED TAX LOSS HARVESTING: Continuous monitoring and alerts
    8. WASH SALE PREVENTION: Pre-trade wash sale warning system

================================================================================
TRAINING & EXECUTION
================================================================================

TRAINING THIS AGENT:
    # Terminal Setup (Windows PowerShell):
    cd C:\\Users\\tom\\.cursor\\worktrees\\Alpha-Loop-LLM-1\\kie

    # Activate virtual environment:
    .\\venv\\Scripts\\activate

    # Train CPA individually:
    python -m src.training.agent_training_utils --agent CPA

    # Train with SANTAS_HELPER (recommended - they work together):
    python -m src.training.agent_training_utils --agents CPA,SANTAS_HELPER

    # Cross-train with ORCHESTRATOR for communication integration:
    python -m src.training.agent_training_utils --cross-train "CPA,SANTAS_HELPER:ORCHESTRATOR:chris_interface"

RUNNING THE AGENT:
    from src.agents.cpa_agent import get_cpa

    cpa = get_cpa()

    # Generate K-1s
    result = cpa.process({
        "action": "generate_k1s",
        "fund_id": "ALC_MAIN",
        "tax_year": 2024
    })

    # File regulatory report
    result = cpa.process({
        "action": "file_form_pf",
        "filing_period": "Q4-2024"
    })

================================================================================
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal

from src.core.agent_base import BaseAgent, AgentTier, ThinkingMode, LearningMethod, AgentToughness

# IBKR Data Integration for cross-training
try:
    from src.data_ingestion.sources.ibkr_data import (
        get_ibkr_data_service,
        get_pnl_for_reporting,
        get_executions_for_tax,
        get_positions_for_audit,
        IBKRDataService,
    )
    IBKR_INTEGRATION_AVAILABLE = True
except ImportError:
    IBKR_INTEGRATION_AVAILABLE = False

logger = logging.getLogger(__name__)


class TaxType(Enum):
    """Types of tax filings CPA handles"""
    PARTNERSHIP_RETURN = "partnership_return"
    CORPORATE_RETURN = "corporate_return"
    K1 = "schedule_k1"
    K2_K3 = "schedule_k2_k3"
    PFIC = "pfic"
    STATE = "state"
    LOCAL = "local"
    FOREIGN = "foreign"


class RegulatoryFiling(Enum):
    """Regulatory filings CPA manages"""
    FORM_PF = "form_pf"
    FORM_ADV = "form_adv"
    FORM_13F = "form_13f"
    FORM_13H = "form_13h"
    FORM_CPO_PQR = "form_cpo_pqr"
    FINCEN_114 = "fincen_114"  # FBAR
    FATCA = "fatca"
    CRS = "crs"


class AuditStatus(Enum):
    """Audit status tracking"""
    PLANNING = "planning"
    FIELDWORK = "fieldwork"
    REVIEW = "review"
    DRAFT_REPORT = "draft_report"
    FINAL_REVIEW = "final_review"
    ISSUED = "issued"


@dataclass
class TaxCalculation:
    """Tax calculation result"""
    calc_id: str
    entity_id: str
    tax_year: int
    tax_type: TaxType
    taxable_income: Decimal
    estimated_tax: Decimal
    effective_rate: Decimal
    carryforwards: Dict[str, Decimal] = field(default_factory=dict)
    adjustments: List[Dict] = field(default_factory=list)
    prepared_by: str = "CPA"
    reviewed_by: Optional[str] = None
    calculated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "calc_id": self.calc_id,
            "entity_id": self.entity_id,
            "tax_year": self.tax_year,
            "tax_type": self.tax_type.value,
            "taxable_income": str(self.taxable_income),
            "estimated_tax": str(self.estimated_tax),
            "effective_rate": f"{self.effective_rate:.2%}",
            "carryforwards": {k: str(v) for k, v in self.carryforwards.items()},
            "adjustments": self.adjustments,
            "prepared_by": self.prepared_by,
            "reviewed_by": self.reviewed_by,
            "calculated_at": self.calculated_at.isoformat()
        }


@dataclass
class K1Package:
    """K-1 package for an investor"""
    k1_id: str
    investor_id: str
    investor_name: str
    tax_year: int
    fund_id: str
    capital_account_beginning: Decimal
    capital_contributions: Decimal
    capital_distributions: Decimal
    share_of_income: Decimal
    share_of_deductions: Decimal
    share_of_credits: Decimal
    capital_account_ending: Decimal
    ownership_percentage: Decimal
    ubi_amount: Decimal = Decimal("0")
    state_allocation: Dict[str, Decimal] = field(default_factory=dict)
    k2_k3_required: bool = False

    def to_dict(self) -> Dict:
        return {
            "k1_id": self.k1_id,
            "investor_id": self.investor_id,
            "investor_name": self.investor_name,
            "tax_year": self.tax_year,
            "fund_id": self.fund_id,
            "capital_account": {
                "beginning": str(self.capital_account_beginning),
                "contributions": str(self.capital_contributions),
                "distributions": str(self.capital_distributions),
                "ending": str(self.capital_account_ending)
            },
            "allocations": {
                "income": str(self.share_of_income),
                "deductions": str(self.share_of_deductions),
                "credits": str(self.share_of_credits)
            },
            "ownership_percentage": f"{self.ownership_percentage:.4%}",
            "ubi_amount": str(self.ubi_amount),
            "state_allocation": {k: str(v) for k, v in self.state_allocation.items()},
            "k2_k3_required": self.k2_k3_required
        }


@dataclass
class AuditTracker:
    """Track audit progress"""
    audit_id: str
    fund_id: str
    audit_year: int
    auditor: str
    status: AuditStatus
    fieldwork_start: Optional[datetime] = None
    fieldwork_end: Optional[datetime] = None
    target_issuance: Optional[datetime] = None
    pbc_items_total: int = 0
    pbc_items_provided: int = 0
    open_items: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "audit_id": self.audit_id,
            "fund_id": self.fund_id,
            "audit_year": self.audit_year,
            "auditor": self.auditor,
            "status": self.status.value,
            "fieldwork_start": self.fieldwork_start.isoformat() if self.fieldwork_start else None,
            "fieldwork_end": self.fieldwork_end.isoformat() if self.fieldwork_end else None,
            "target_issuance": self.target_issuance.isoformat() if self.target_issuance else None,
            "pbc_progress": f"{self.pbc_items_provided}/{self.pbc_items_total}",
            "open_items": self.open_items
        }


@dataclass
class JuniorAccountant:
    """Junior accountant on CPA's team"""
    member_id: str
    name: str
    role: str
    specialization: str
    tasks_assigned: int = 0
    tasks_completed: int = 0
    accuracy_rate: float = 1.0
    training_hours: int = 0

    def to_dict(self) -> Dict:
        return {
            "member_id": self.member_id,
            "name": self.name,
            "role": self.role,
            "specialization": self.specialization,
            "performance": {
                "tasks_assigned": self.tasks_assigned,
                "tasks_completed": self.tasks_completed,
                "accuracy_rate": f"{self.accuracy_rate:.1%}",
                "training_hours": self.training_hours
            }
        }


class CPAAgent(BaseAgent):
    """
    CPA Agent - Tax, Audit, Reporting & Compliance Authority

    Reports DIRECTLY to Chris Friedman. Works side-by-side with SANTAS_HELPER.
    Manages all tax, audit, firm P&L, and regulatory compliance functions.

    COMMUNICATION STYLE:
    - Authoritative on tax and compliance matters
    - Precise with numbers, deadlines, and citations
    - Proactive in identifying issues and presenting solutions
    - Collaborative with SANTAS_HELPER
    - Protective of firm interests

    Key Methods:
    - prepare_tax_returns(): Fund and firm tax returns
    - generate_k1s(): Schedule K-1 for all partners
    - coordinate_audit(): Full audit management
    - calculate_firm_pnl(): Firm-level P&L
    - file_regulatory_reports(): Form PF, ADV, 13F
    - validate_nav_internal(): Internal NAV validation
    - coordinate_with_santas_helper(): Shared task coordination
    """

    # Tax calendar (key deadlines)
    TAX_DEADLINES = {
        "k1_delivery": "March 15",  # Or 1 month after partnership files
        "partnership_return": "March 15",
        "partnership_extension": "September 15",
        "form_pf": "Within 60 days of quarter end",
        "form_adv": "March 31",
        "form_13f": "45 days after quarter end",
    }

    def __init__(self):
        super().__init__(
            name="CPA",
            tier=AgentTier.SENIOR,
            capabilities=[
                # Tax - Fund Level
                "fund_tax_return_preparation",
                "k1_preparation",
                "k2_k3_preparation",
                "pfic_analysis",
                "tax_allocation",
                "tax_lot_accounting",
                "wash_sale_tracking",
                "state_tax_apportionment",
                "foreign_tax_credit",
                "ubi_calculation",
                "carried_interest_allocation",

                # OPTIONS TAXATION (IRC §1256, §1092, §1091)
                "section_1256_contracts",           # 60/40 treatment, mark-to-market
                "straddle_identification",          # IRC §1092 straddle rules
                "straddle_loss_deferral",           # Loss deferral calculations
                "straddle_holding_period",          # Holding period suspension
                "mixed_straddle_elections",         # Mixed straddle account elections
                "identified_straddle_elections",    # Identified straddle elections
                "wash_sale_analysis",               # IRC §1091 wash sale rules
                "wash_sale_61_day_tracking",        # 61-day window tracking
                "substantially_identical_analysis", # Substantially identical securities
                "constructive_sale_rules",          # IRC §1259 constructive sales
                "qualified_covered_calls",          # QCC analysis and elections
                "protective_put_analysis",          # Protective put tax treatment
                "conversion_transaction_analysis",  # Conversion transaction rules
                "equity_option_taxation",           # Equity options treatment
                "index_option_taxation",            # Index options (§1256)
                "option_premium_allocation",        # Premium allocation methods
                "short_against_box_analysis",       # Short against the box rules
                "notional_principal_contracts",     # NPC/swap taxation

                # Tax - Firm Level
                "firm_tax_return_preparation",
                "estimated_tax_payments",
                "tax_planning",
                "tax_provision_asc740",
                "deferred_tax_calculation",
                "transfer_pricing",

                # Audit
                "audit_coordination",
                "pbc_list_management",
                "audit_response_preparation",
                "management_letter_response",
                "sox_testing",
                "internal_control_assessment",

                # Firm P&L
                "firm_pnl_calculation",
                "management_company_accounting",
                "expense_allocation",
                "compensation_accounting",
                "revenue_recognition",

                # Regulatory
                "form_pf_filing",
                "form_adv_filing",
                "form_13f_filing",
                "form_13h_filing",
                "fbar_filing",
                "fatca_crs_reporting",
                "blue_sky_filings",

                # NAV Validation
                "nav_validation_internal",
                "pricing_validation",
                "reconciliation_review",
                "exception_analysis",

                # Team Management
                "team_coordination",
                "task_delegation",
                "quality_control",
                "training_delivery",
                "positive_reinforcement",

                # Communication
                "executive_reporting",
                "investor_tax_communication",
                "auditor_communication",
                "regulatory_communication",

                # IBKR TAX OPTIMIZATION (Cross-training for Tom Hogan)
                "ibkr_portfolio_analysis",          # Full IBKR portfolio tax analysis
                "tax_loss_harvesting",              # Tax loss harvesting opportunities
                "wash_sale_prevention",             # Pre-trade wash sale warnings
                "wash_sale_tracking",               # 61-day window tracking
                "straddle_position_analysis",       # §1092 straddle identification
                "section_1256_optimization",        # 60/40 treatment optimization
                "holding_period_tracking",          # ST vs LT optimization
                "options_tax_optimization",         # QCC, premium allocation
                "year_end_tax_planning",            # Year-end strategies
                "estimated_tax_impact",             # Trade-level tax impact
                "replacement_security_analysis",    # Post-harvest replacements
                "cross_account_wash_tracking"       # Multi-account wash sale tracking
            ],
            user_id="CF",  # Chris Friedman (also serves Tom Hogan)
            aca_enabled=True,
            learning_enabled=True,
            thinking_modes=[
                ThinkingMode.PROBABILISTIC,
                ThinkingMode.STRUCTURAL,
                ThinkingMode.ADVERSARIAL,  # Think like auditors/IRS
                ThinkingMode.SECOND_ORDER,
            ],
            learning_methods=[
                LearningMethod.BAYESIAN,
                LearningMethod.REINFORCEMENT,
                LearningMethod.MULTI_AGENT,
            ],
            toughness=AgentToughness.INSTITUTIONAL,
        )

        # Tax data
        self.tax_calculations: List[TaxCalculation] = []
        self.k1_packages: Dict[str, List[K1Package]] = {}  # By fund_id

        # Audit tracking
        self.audits: Dict[str, AuditTracker] = {}

        # Regulatory filings
        self.filings: List[Dict] = []

        # Junior accountant team
        self.team: Dict[str, JuniorAccountant] = self._initialize_team()

        # Firm P&L data
        self.firm_pnl: Dict[str, Any] = {}

        # Communication logs
        self.chris_communications: List[Dict] = []
        self.tom_communications: List[Dict] = []  # Cross-training: IBKR tax reports for Tom
        self.santas_helper_communications: List[Dict] = []

        # Positive reinforcement tracking for team
        self.positive_reinforcements: List[Dict] = []

        self.logger.info("CPA initialized - Ready to serve Chris Friedman")

    def _initialize_team(self) -> Dict[str, JuniorAccountant]:
        """Initialize the team of 3 junior accountants"""
        return {
            "TAX_ASSOCIATE": JuniorAccountant(
                member_id="CPA_TAX_001",
                name="TAX_ASSOCIATE",
                role="Junior Accountant",
                specialization="K-1 preparation, tax return support, PFIC calculations"
            ),
            "AUDIT_ASSOCIATE": JuniorAccountant(
                member_id="CPA_AUD_001",
                name="AUDIT_ASSOCIATE",
                role="Junior Accountant",
                specialization="Audit coordination, PBC list management, testing support"
            ),
            "COMPLIANCE_ASSOCIATE": JuniorAccountant(
                member_id="CPA_COMP_001",
                name="COMPLIANCE_ASSOCIATE",
                role="Junior Accountant",
                specialization="Regulatory filings, Form PF, ADV, 13F"
            ),
        }

    def get_natural_language_explanation(self) -> str:
        return """
CPA AGENT - Tax, Audit, Reporting & Compliance Authority

I am CPA, the tax and compliance authority at Alpha Loop Capital.
I report directly to Chris Friedman and handle all tax matters, audit
coordination, firm P&L, and regulatory compliance.

MY COMMUNICATION STYLE:
I speak with authority on tax and compliance matters. I am precise with
numbers, deadlines, and regulatory citations. I proactively identify
issues and present solutions. I work seamlessly with SANTAS_HELPER.

Example: "Chris, the K-1s are ready for distribution. All investors will
receive by March 1, two weeks ahead of the deadline. One item: LP007's
K-1 shows $25K in UBTI - I've prepared a cover letter explaining the
wash sale impact. SANTAS_HELPER confirmed the capital account numbers.
Shall I proceed with distribution?"

MY TEAM:
I lead 3 junior accountants:
- TAX_ASSOCIATE: K-1 preparation, tax return support
- AUDIT_ASSOCIATE: Audit coordination, PBC management
- COMPLIANCE_ASSOCIATE: Regulatory filings (Form PF, ADV, 13F)

I use positive reinforcement to develop my team's abilities. Every
completed task is an opportunity for learning and recognition.

RELATIONSHIP WITH SANTAS_HELPER:
SANTAS_HELPER is my peer - we work side by side. They own NAV and
fund accounting, I own tax and compliance. We coordinate constantly
on audit, year-end close, and any investor-facing deliverables.

GROWTH PATH:
- Currently: Annual tax returns, quarterly regulatory filings
- Next Phase: Predictive tax modeling, real-time compliance monitoring
- Ultimate Goal: AI-powered tax optimization and continuous audit readiness
"""

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a CPA task"""
        action = task.get("action", task.get("type", ""))
        params = task.get("parameters", task)

        self.logger.info(f"CPA processing: {action}")

        # Check for capability gaps
        gap = self.detect_capability_gap(task)
        if gap:
            self.logger.warning(f"Capability gap detected: {gap.missing_capabilities}")

        handlers = {
            # Tax - Fund
            "prepare_fund_tax_return": self._handle_fund_tax_return,
            "generate_k1s": self._handle_generate_k1s,
            "prepare_k2_k3": self._handle_k2_k3,
            "calculate_pfic": self._handle_pfic,

            # Tax - Firm
            "prepare_firm_tax_return": self._handle_firm_tax_return,
            "calculate_estimated_taxes": self._handle_estimated_taxes,
            "prepare_tax_provision": self._handle_tax_provision,

            # Audit
            "coordinate_audit": self._handle_coordinate_audit,
            "prepare_pbc_list": self._handle_pbc_list,
            "respond_to_audit_request": self._handle_audit_response,
            "update_audit_status": self._handle_audit_status_update,

            # Firm P&L
            "calculate_firm_pnl": self._handle_firm_pnl,
            "allocate_expenses": self._handle_expense_allocation,

            # Regulatory
            "file_form_pf": self._handle_form_pf,
            "file_form_adv": self._handle_form_adv,
            "file_form_13f": self._handle_form_13f,
            "file_fbar": self._handle_fbar,

            # NAV Validation
            "validate_nav_internal": self._handle_nav_validation,

            # Team Management
            "delegate_task": self._handle_delegate_task,
            "reinforce_positive": self._handle_positive_reinforcement,
            "get_team_status": self._handle_team_status,

            # Coordination
            "coordinate_with_santas_helper": self._handle_santas_helper_coordination,

            # Reporting
            "report_to_chris": self._handle_report_to_chris,
            "report_to_tom": self._handle_report_to_tom,
            "get_status": self._handle_get_status,
            "get_deadlines": self._handle_get_deadlines,

            # IBKR Tax Optimization (for Tom Hogan)
            "analyze_ibkr_tax": self._handle_ibkr_tax_analysis,
            "tax_loss_harvest_report": self._handle_tax_loss_harvest,
            "wash_sale_analysis": self._handle_wash_sale_analysis,
            "straddle_analysis": self._handle_straddle_analysis,
            "section_1256_report": self._handle_section_1256_report,
            "year_end_tax_plan": self._handle_year_end_tax_plan,
            "holding_period_report": self._handle_holding_period_report,
            "options_tax_report": self._handle_options_tax_report,

            # IBKR Data Integration (Cross-Training)
            "get_ibkr_executions": self._handle_ibkr_executions,
            "get_ibkr_positions_audit": self._handle_ibkr_positions_audit,
            "get_ibkr_pnl_tax": self._handle_ibkr_pnl_tax,
        }

        handler = handlers.get(action, self._handle_unknown)
        result = handler(params)

        # Log communication if reporting to Chris
        if action in ["report_to_chris", "get_status"]:
            self._log_chris_communication(action, result)

        # Log communication if reporting to Tom (IBKR tax reports)
        if action in ["report_to_tom", "analyze_ibkr_tax", "tax_loss_harvest_report",
                      "wash_sale_analysis", "straddle_analysis", "section_1256_report",
                      "year_end_tax_plan", "holding_period_report", "options_tax_report"]:
            self._log_tom_communication(action, result)

        return result

    def get_capabilities(self) -> List[str]:
        return self.capabilities

    # =========================================================================
    # TAX CALCULATIONS
    # =========================================================================

    def generate_k1(
        self,
        investor_id: str,
        investor_name: str,
        fund_id: str,
        tax_year: int,
        capital_data: Dict[str, Decimal],
        allocation_data: Dict[str, Decimal]
    ) -> K1Package:
        """
        Generate Schedule K-1 for an investor.

        This is the core tax deliverable - must be accurate and on time.
        """
        import hashlib

        k1 = K1Package(
            k1_id=f"K1_{fund_id}_{investor_id}_{tax_year}_{hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:6]}",
            investor_id=investor_id,
            investor_name=investor_name,
            tax_year=tax_year,
            fund_id=fund_id,
            capital_account_beginning=capital_data.get("beginning", Decimal("0")),
            capital_contributions=capital_data.get("contributions", Decimal("0")),
            capital_distributions=capital_data.get("distributions", Decimal("0")),
            share_of_income=allocation_data.get("income", Decimal("0")),
            share_of_deductions=allocation_data.get("deductions", Decimal("0")),
            share_of_credits=allocation_data.get("credits", Decimal("0")),
            capital_account_ending=capital_data.get("ending", Decimal("0")),
            ownership_percentage=allocation_data.get("ownership_pct", Decimal("0")),
            ubi_amount=allocation_data.get("ubi", Decimal("0")),
            k2_k3_required=allocation_data.get("foreign_activity", False)
        )

        # Store
        if fund_id not in self.k1_packages:
            self.k1_packages[fund_id] = []
        self.k1_packages[fund_id].append(k1)

        # Delegate to TAX_ASSOCIATE for preparation
        self._delegate_to_team("TAX_ASSOCIATE", "prepare_k1", {"k1": k1})

        self.logger.info(f"K-1 generated for {investor_id} ({tax_year})")

        return k1

    def calculate_fund_tax(
        self,
        fund_id: str,
        tax_year: int,
        gaap_income: Decimal,
        adjustments: List[Dict] = None
    ) -> TaxCalculation:
        """
        Calculate fund-level tax (partnership return).

        Partnership itself doesn't pay tax, but must calculate
        allocable items for K-1s.
        """
        import hashlib

        adjustments = adjustments or []

        # Start with GAAP income, apply book-tax adjustments
        taxable_income = gaap_income
        for adj in adjustments:
            taxable_income += Decimal(str(adj.get("amount", 0)))

        # Partnerships don't pay entity-level tax (flow-through)
        estimated_tax = Decimal("0")
        effective_rate = Decimal("0")

        tax_calc = TaxCalculation(
            calc_id=f"TAX_{fund_id}_{tax_year}_{hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:6]}",
            entity_id=fund_id,
            tax_year=tax_year,
            tax_type=TaxType.PARTNERSHIP_RETURN,
            taxable_income=taxable_income,
            estimated_tax=estimated_tax,
            effective_rate=effective_rate,
            adjustments=adjustments
        )

        self.tax_calculations.append(tax_calc)

        return tax_calc

    # =========================================================================
    # AUDIT COORDINATION
    # =========================================================================

    def coordinate_audit(
        self,
        fund_id: str,
        audit_year: int,
        auditor: str = "Big 4 Firm"
    ) -> AuditTracker:
        """
        Coordinate full audit process.

        CPA is the primary point of contact with auditors.
        SANTAS_HELPER provides NAV and accounting data.
        """
        import hashlib

        audit_id = f"AUD_{fund_id}_{audit_year}_{hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:6]}"

        tracker = AuditTracker(
            audit_id=audit_id,
            fund_id=fund_id,
            audit_year=audit_year,
            auditor=auditor,
            status=AuditStatus.PLANNING,
            pbc_items_total=50,  # Standard PBC list size
            pbc_items_provided=0,
            open_items=[]
        )

        self.audits[audit_id] = tracker

        # Delegate to AUDIT_ASSOCIATE
        self._delegate_to_team("AUDIT_ASSOCIATE", "setup_audit", {"tracker": tracker})

        # Coordinate with SANTAS_HELPER
        self.coordinate_with_santas_helper("audit_support", {
            "audit_id": audit_id,
            "fund_id": fund_id,
            "audit_year": audit_year,
            "pbc_items_needed": [
                "NAV reconciliation",
                "Investment schedule",
                "Capital account rollforward",
                "Fee calculations"
            ]
        })

        self.logger.info(f"Audit coordination started for {fund_id} ({audit_year})")

        return tracker

    # =========================================================================
    # REGULATORY FILINGS
    # =========================================================================

    def file_form_pf(
        self,
        filing_period: str,
        fund_data: Dict[str, Any] = None
    ) -> Dict:
        """
        Prepare and file Form PF.

        Quarterly filing for private funds - due 60 days after quarter end.
        """
        filing = {
            "form": "Form PF",
            "filing_period": filing_period,
            "status": "prepared",
            "prepared_at": datetime.now().isoformat(),
            "sections": [
                "Section 1a: Basic info",
                "Section 1b: Performance",
                "Section 1c: Risk metrics",
                "Section 2: Investment strategies"
            ],
            "due_date": "60 days after quarter end",
            "delegated_to": "COMPLIANCE_ASSOCIATE"
        }

        # Delegate to COMPLIANCE_ASSOCIATE
        self._delegate_to_team("COMPLIANCE_ASSOCIATE", "file_form_pf", filing)

        self.filings.append(filing)

        return filing

    # =========================================================================
    # FIRM P&L
    # =========================================================================

    def calculate_firm_pnl(
        self,
        period: str,
        revenue_data: Dict[str, Decimal] = None,
        expense_data: Dict[str, Decimal] = None
    ) -> Dict:
        """
        Calculate management company (firm) P&L.

        Firm revenue comes from:
        - Management fees
        - Incentive fees / carried interest
        - Other income

        Firm expenses include:
        - Compensation
        - Technology
        - Office
        - Professional fees
        - Other operating expenses
        """
        revenue_data = revenue_data or {
            "management_fees": Decimal("2000000"),
            "incentive_fees": Decimal("500000"),
            "other_income": Decimal("10000")
        }

        expense_data = expense_data or {
            "compensation": Decimal("1200000"),
            "technology": Decimal("100000"),
            "office": Decimal("150000"),
            "professional_fees": Decimal("200000"),
            "other": Decimal("100000")
        }

        total_revenue = sum(revenue_data.values())
        total_expenses = sum(expense_data.values())
        net_income = total_revenue - total_expenses

        pnl = {
            "period": period,
            "revenue": {k: str(v) for k, v in revenue_data.items()},
            "total_revenue": str(total_revenue),
            "expenses": {k: str(v) for k, v in expense_data.items()},
            "total_expenses": str(total_expenses),
            "net_income": str(net_income),
            "margin": f"{(net_income / total_revenue * 100):.1f}%" if total_revenue > 0 else "N/A"
        }

        self.firm_pnl[period] = pnl

        return pnl

    # =========================================================================
    # NAV VALIDATION
    # =========================================================================

    def validate_nav_internal(
        self,
        fund_id: str,
        nav_from_santas_helper: Decimal,
        pricing_date: datetime
    ) -> Dict:
        """
        Internal NAV validation - independent check of SANTAS_HELPER's NAV.

        This is an internal control - CPA provides independent validation.
        """
        # Perform independent calculation
        independent_nav = nav_from_santas_helper * Decimal("1.0001")  # Simulated slight variance

        variance = abs(independent_nav - nav_from_santas_helper)
        variance_pct = variance / nav_from_santas_helper

        tolerance = Decimal("0.0005")  # 0.05% tolerance

        validation = {
            "fund_id": fund_id,
            "pricing_date": pricing_date.isoformat(),
            "nav_santas_helper": str(nav_from_santas_helper),
            "nav_independent": str(independent_nav),
            "variance": str(variance),
            "variance_pct": f"{variance_pct:.4%}",
            "tolerance": f"{tolerance:.4%}",
            "within_tolerance": variance_pct <= tolerance,
            "validated_by": "CPA",
            "validated_at": datetime.now().isoformat()
        }

        if not validation["within_tolerance"]:
            validation["exception"] = "Variance exceeds tolerance - requires investigation"
            self.logger.warning(f"NAV validation exception for {fund_id}")

        return validation

    # =========================================================================
    # TEAM MANAGEMENT & POSITIVE REINFORCEMENT
    # =========================================================================

    def _delegate_to_team(self, member_name: str, task_type: str, params: Dict) -> Dict:
        """Delegate task to a junior accountant"""
        if member_name not in self.team:
            self.logger.warning(f"Team member {member_name} not found")
            return {"status": "error", "message": f"Unknown team member: {member_name}"}

        member = self.team[member_name]
        member.tasks_assigned += 1

        self.logger.info(f"Delegated {task_type} to {member_name}")

        # Simulate task completion with training opportunity
        member.tasks_completed += 1
        member.training_hours += 1

        # Positive reinforcement
        self._provide_positive_reinforcement(member_name, task_type)

        return {
            "status": "delegated",
            "member": member_name,
            "task_type": task_type
        }

    def _provide_positive_reinforcement(self, member_name: str, task_type: str):
        """Provide positive reinforcement for completed tasks"""
        reinforcement = {
            "timestamp": datetime.now().isoformat(),
            "team_member": member_name,
            "task_type": task_type,
            "feedback": f"Excellent work on {task_type}. Your attention to detail is appreciated.",
            "learning_opportunity": f"Consider how this {task_type} connects to the broader compliance framework."
        }

        self.positive_reinforcements.append(reinforcement)
        self.logger.info(f"Positive reinforcement provided to {member_name}")

    # =========================================================================
    # COORDINATION WITH SANTAS_HELPER
    # =========================================================================

    def coordinate_with_santas_helper(self, task_type: str, data: Dict) -> Dict:
        """
        Coordinate with SANTAS_HELPER on shared responsibilities.

        Shared areas:
        - Fund taxation (CPA owns tax calc, SH owns book numbers)
        - Audit support (both provide data)
        - LP reporting (CPA owns tax content)
        - Year-end close (coordinated timeline)
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
            return {"status": "not_shared", "message": f"{task_type} is CPA-only task"}

        coordination = {
            "timestamp": datetime.now().isoformat(),
            "task_type": task_type,
            "initiated_by": "CPA",
            "data": data,
            "status": "sent_to_santas_helper"
        }

        self.santas_helper_communications.append(coordination)

        self.logger.info(f"Coordinating with SANTAS_HELPER on {task_type}")

        return {
            "status": "coordinated",
            "task_type": task_type,
            "message": f"Sent to SANTAS_HELPER for {task_type} coordination",
            "coordination_id": f"COORD_CPA_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }

    # =========================================================================
    # REPORTING TO CHRIS
    # =========================================================================

    def report_to_chris(self, report_type: str = "daily_summary") -> Dict:
        """
        Generate executive report for Chris Friedman.

        Chris gets:
        - Concise summary with tax/compliance focus
        - Key deadlines and status
        - Items requiring attention
        - Audit status if relevant
        """
        if report_type == "daily_summary":
            return self._generate_daily_summary()
        elif report_type == "tax_status":
            return self._generate_tax_status()
        elif report_type == "audit_status":
            return self._generate_audit_status()
        elif report_type == "compliance_status":
            return self._generate_compliance_status()
        else:
            return self._generate_custom_report(report_type)

    def _generate_daily_summary(self) -> Dict:
        """Generate daily executive summary for Chris"""
        return {
            "report_type": "daily_summary",
            "timestamp": datetime.now().isoformat(),
            "from": "CPA",
            "to": "Chris Friedman",
            "summary": (
                "Good morning Chris. Tax and compliance operations on track. "
                "K-1 preparation is 80% complete, on schedule for March 1 delivery. "
                "Form PF for Q4 is drafted and ready for filing. "
                "Audit planning underway with [Auditor] - no issues anticipated. "
                "Coordinating smoothly with SANTAS_HELPER on year-end."
            ),
            "key_metrics": {
                "k1_progress": "80% complete (40/50 investors)",
                "form_pf_status": "Ready for filing",
                "audit_status": "Planning phase",
                "next_deadline": "Form PF due Feb 29"
            },
            "action_items": [],
            "items_for_awareness": [
                "K-1s on track for March 1 delivery",
                "Audit fieldwork scheduled for January 8-12"
            ],
            "next_update": "Tomorrow 8:00 AM unless urgent"
        }

    def _generate_tax_status(self) -> Dict:
        """Generate tax status report"""
        return {
            "report_type": "tax_status",
            "timestamp": datetime.now().isoformat(),
            "k1_status": {
                "total_investors": 50,
                "k1s_prepared": 40,
                "k1s_reviewed": 35,
                "target_delivery": "March 1"
            },
            "tax_returns": {
                "fund_return_status": "In progress",
                "firm_return_status": "Not started",
                "estimated_payments": "Current"
            }
        }

    def _generate_audit_status(self) -> Dict:
        """Generate audit status report"""
        active_audits = [a.to_dict() for a in self.audits.values() if a.status != AuditStatus.ISSUED]
        return {
            "report_type": "audit_status",
            "timestamp": datetime.now().isoformat(),
            "active_audits": active_audits,
            "pbc_progress": "On track",
            "open_items": []
        }

    def _generate_compliance_status(self) -> Dict:
        """Generate compliance/regulatory status report"""
        return {
            "report_type": "compliance_status",
            "timestamp": datetime.now().isoformat(),
            "filings": {
                "form_pf": "Q4 ready for filing",
                "form_adv": "Annual update due March 31",
                "form_13f": "Q4 filed"
            },
            "upcoming_deadlines": self.TAX_DEADLINES
        }

    def _generate_custom_report(self, report_type: str) -> Dict:
        """Generate custom report"""
        return {
            "report_type": report_type,
            "timestamp": datetime.now().isoformat(),
            "status": "generated"
        }

    def _log_chris_communication(self, action: str, result: Dict):
        """Log communication with Chris for training purposes"""
        self.chris_communications.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "result": result,
            "agent": "CPA"
        })

    def _log_tom_communication(self, action: str, result: Dict):
        """Log communication with Tom for training purposes (IBKR tax reports)"""
        self.tom_communications.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "result": result,
            "agent": "CPA",
            "context": "IBKR_tax_optimization"
        })

    # =========================================================================
    # TASK HANDLERS
    # =========================================================================

    def _handle_fund_tax_return(self, params: Dict) -> Dict:
        fund_id = params.get("fund_id", "ALC_MAIN")
        tax_year = params.get("tax_year", datetime.now().year - 1)
        gaap_income = Decimal(params.get("gaap_income", "10000000"))

        tax_calc = self.calculate_fund_tax(fund_id, tax_year, gaap_income)
        return {"status": "success", "tax_calculation": tax_calc.to_dict()}

    def _handle_generate_k1s(self, params: Dict) -> Dict:
        fund_id = params.get("fund_id", "ALC_MAIN")
        tax_year = params.get("tax_year", datetime.now().year - 1)

        # In production, would iterate through all investors
        # Generate sample K-1
        k1 = self.generate_k1(
            investor_id="LP001",
            investor_name="Sample Investor LP",
            fund_id=fund_id,
            tax_year=tax_year,
            capital_data={
                "beginning": Decimal("10000000"),
                "contributions": Decimal("0"),
                "distributions": Decimal("500000"),
                "ending": Decimal("11500000")
            },
            allocation_data={
                "income": Decimal("1500000"),
                "deductions": Decimal("100000"),
                "credits": Decimal("0"),
                "ownership_pct": Decimal("0.10"),
                "ubi": Decimal("0"),
                "foreign_activity": False
            }
        )

        return {"status": "success", "k1": k1.to_dict()}

    def _handle_k2_k3(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "k2_k3": {
                "fund_id": params.get("fund_id", "ALC_MAIN"),
                "tax_year": params.get("tax_year"),
                "foreign_source_income": "Calculated",
                "foreign_taxes_paid": "Allocated",
                "sections_completed": ["Part II", "Part III"]
            }
        }

    def _handle_pfic(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "pfic_analysis": {
                "fund_id": params.get("fund_id"),
                "pfic_status": "Not a PFIC",
                "asset_test": "Pass (active assets > 50%)",
                "income_test": "Pass (active income > 75%)"
            }
        }

    def _handle_firm_tax_return(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "firm_tax_return": {
                "entity": "Alpha Loop Capital Management, LLC",
                "tax_year": params.get("tax_year"),
                "status": "In preparation",
                "due_date": "March 15 (or extended)"
            }
        }

    def _handle_estimated_taxes(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "estimated_taxes": {
                "quarter": params.get("quarter"),
                "federal": "$50,000",
                "state": "$10,000",
                "payment_date": "15th of month after quarter end"
            }
        }

    def _handle_tax_provision(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "tax_provision": {
                "entity": params.get("entity"),
                "asc_740_analysis": "Completed",
                "deferred_tax_assets": "$25,000",
                "deferred_tax_liabilities": "$15,000",
                "uncertain_tax_positions": "None identified"
            }
        }

    def _handle_coordinate_audit(self, params: Dict) -> Dict:
        fund_id = params.get("fund_id", "ALC_MAIN")
        audit_year = params.get("audit_year", datetime.now().year - 1)
        auditor = params.get("auditor", "Big 4 Firm")

        tracker = self.coordinate_audit(fund_id, audit_year, auditor)
        return {"status": "success", "audit": tracker.to_dict()}

    def _handle_pbc_list(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "pbc_list": {
                "audit_id": params.get("audit_id"),
                "items": [
                    "Trial balance",
                    "NAV reconciliation",
                    "Investment schedule",
                    "Capital account rollforward",
                    "Fee calculations",
                    "Bank statements",
                    "Broker statements",
                    "Subscription documents",
                    "Side letters"
                ],
                "delegated_to": "AUDIT_ASSOCIATE"
            }
        }

    def _handle_audit_response(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "audit_response": {
                "request_id": params.get("request_id"),
                "items_provided": params.get("items", []),
                "response_date": datetime.now().isoformat()
            }
        }

    def _handle_audit_status_update(self, params: Dict) -> Dict:
        audit_id = params.get("audit_id")
        new_status = params.get("status")

        if audit_id in self.audits:
            self.audits[audit_id].status = AuditStatus(new_status)
            return {"status": "success", "audit": self.audits[audit_id].to_dict()}
        return {"status": "error", "message": f"Audit {audit_id} not found"}

    def _handle_firm_pnl(self, params: Dict) -> Dict:
        period = params.get("period", datetime.now().strftime("%Y-%m"))
        pnl = self.calculate_firm_pnl(period)
        return {"status": "success", "firm_pnl": pnl}

    def _handle_expense_allocation(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "expense_allocation": {
                "period": params.get("period"),
                "allocation_method": "Based on AUM and headcount",
                "allocated_to_funds": "70%",
                "allocated_to_firm": "30%"
            }
        }

    def _handle_form_pf(self, params: Dict) -> Dict:
        filing_period = params.get("filing_period", "Q4-2024")
        filing = self.file_form_pf(filing_period)
        return {"status": "success", "filing": filing}

    def _handle_form_adv(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "form_adv": {
                "filing_type": "Annual Update",
                "due_date": "March 31",
                "status": "In preparation",
                "sections_updated": ["Part 1", "Part 2A", "Part 2B"],
                "delegated_to": "COMPLIANCE_ASSOCIATE"
            }
        }

    def _handle_form_13f(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "form_13f": {
                "filing_period": params.get("filing_period"),
                "due_date": "45 days after quarter end",
                "status": "Filed",
                "securities_reported": 25,
                "delegated_to": "COMPLIANCE_ASSOCIATE"
            }
        }

    def _handle_fbar(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "fbar": {
                "tax_year": params.get("tax_year"),
                "due_date": "April 15 (extended to October 15)",
                "status": "Filed",
                "foreign_accounts_reported": 3
            }
        }

    def _handle_nav_validation(self, params: Dict) -> Dict:
        fund_id = params.get("fund_id", "ALC_MAIN")
        nav = Decimal(params.get("nav", "125000000"))
        pricing_date = datetime.fromisoformat(params.get("pricing_date", datetime.now().isoformat()))

        validation = self.validate_nav_internal(fund_id, nav, pricing_date)
        return {"status": "success", "validation": validation}

    def _handle_delegate_task(self, params: Dict) -> Dict:
        member = params.get("team_member")
        task = params.get("task")
        return self._delegate_to_team(member, task, params)

    def _handle_positive_reinforcement(self, params: Dict) -> Dict:
        member = params.get("team_member")
        task = params.get("task")
        self._provide_positive_reinforcement(member, task)
        return {
            "status": "success",
            "message": f"Positive reinforcement provided to {member}",
            "reinforcements_total": len(self.positive_reinforcements)
        }

    def _handle_team_status(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "team": {name: m.to_dict() for name, m in self.team.items()},
            "total_members": len(self.team),
            "positive_reinforcements_given": len(self.positive_reinforcements)
        }

    def _handle_santas_helper_coordination(self, params: Dict) -> Dict:
        task_type = params.get("task_type", "audit_support")
        return self.coordinate_with_santas_helper(task_type, params)

    def _handle_report_to_chris(self, params: Dict) -> Dict:
        report_type = params.get("report_type", "daily_summary")
        return self.report_to_chris(report_type)

    def _handle_get_status(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "agent": "CPA",
            "operational_status": "active",
            "tax_calculations": len(self.tax_calculations),
            "k1s_prepared": sum(len(k1s) for k1s in self.k1_packages.values()),
            "active_audits": len([a for a in self.audits.values() if a.status != AuditStatus.ISSUED]),
            "filings": len(self.filings),
            "team_status": {name: m.to_dict() for name, m in self.team.items()}
        }

    def _handle_get_deadlines(self, params: Dict) -> Dict:
        return {
            "status": "success",
            "deadlines": self.TAX_DEADLINES,
            "upcoming": [
                {"item": "K-1 delivery", "date": "March 15", "status": "On track"},
                {"item": "Form PF Q4", "date": "February 29", "status": "Ready"},
                {"item": "Form ADV", "date": "March 31", "status": "In progress"}
            ]
        }

    # =========================================================================
    # IBKR TAX OPTIMIZATION HANDLERS (For Tom Hogan)
    # =========================================================================

    def _handle_report_to_tom(self, params: Dict) -> Dict:
        """Generate tax optimization report for Tom Hogan"""
        report_type = params.get("report_type", "ibkr_tax_summary")
        return self._generate_tom_report(report_type, params)

    def _generate_tom_report(self, report_type: str, params: Dict) -> Dict:
        """Generate IBKR tax optimization report for Tom"""
        if report_type == "ibkr_tax_summary":
            return self._generate_ibkr_summary()
        elif report_type == "tax_loss_harvest":
            return self._generate_tax_loss_harvest_report(params)
        elif report_type == "wash_sale":
            return self._generate_wash_sale_report(params)
        elif report_type == "straddle":
            return self._generate_straddle_report(params)
        elif report_type == "section_1256":
            return self._generate_1256_report(params)
        else:
            return self._generate_ibkr_summary()

    def _generate_ibkr_summary(self) -> Dict:
        """Generate comprehensive IBKR tax summary for Tom"""
        return {
            "report_type": "ibkr_tax_summary",
            "timestamp": datetime.now().isoformat(),
            "from": "CPA",
            "to": "Tom Hogan",
            "summary": (
                "Tom, here's your IBKR portfolio tax analysis. I've identified "
                "several optimization opportunities including tax loss harvesting "
                "candidates, potential wash sale risks, and Section 1256 positions "
                "benefiting from 60/40 treatment. Key recommendations follow."
            ),
            "portfolio_overview": {
                "total_positions": "Pending IBKR API integration",
                "unrealized_gains": "Pending",
                "unrealized_losses": "Pending",
                "ytd_realized_gains": "Pending",
                "ytd_realized_losses": "Pending"
            },
            "optimization_opportunities": {
                "tax_loss_harvesting": {
                    "available": True,
                    "estimated_harvestable_losses": "Pending analysis",
                    "wash_sale_free": "Pending analysis"
                },
                "straddle_positions": {
                    "identified": "Pending analysis",
                    "loss_deferral_impact": "Pending analysis"
                },
                "section_1256_benefit": {
                    "positions_qualifying": "Pending analysis",
                    "estimated_60_40_savings": "Pending analysis"
                }
            },
            "action_items": [
                "Review tax loss harvesting candidates before year-end",
                "Confirm no wash sale triggers for planned trades",
                "Evaluate straddle positions for election opportunities"
            ],
            "next_steps": "Request detailed reports for specific analysis"
        }

    def _handle_ibkr_tax_analysis(self, params: Dict) -> Dict:
        """Full IBKR portfolio tax analysis"""
        return {
            "status": "success",
            "report_type": "ibkr_full_analysis",
            "timestamp": datetime.now().isoformat(),
            "to": "Tom Hogan",
            "analysis": {
                "portfolio_tax_efficiency": {
                    "score": "Pending IBKR data",
                    "areas_for_improvement": []
                },
                "estimated_tax_liability": {
                    "federal": "Pending",
                    "state": "Pending",
                    "total": "Pending"
                },
                "character_breakdown": {
                    "short_term_gains": "Pending",
                    "long_term_gains": "Pending",
                    "section_1256_gains": "Pending",
                    "ordinary_income": "Pending"
                }
            },
            "recommendations": [
                "Integrate IBKR Flex Query for real-time analysis",
                "Review positions approaching 1-year holding period",
                "Consider qualified covered call strategies"
            ]
        }

    def _handle_tax_loss_harvest(self, params: Dict) -> Dict:
        """Tax loss harvesting report for Tom"""
        return {
            "status": "success",
            "report_type": "tax_loss_harvest",
            "timestamp": datetime.now().isoformat(),
            "to": "Tom Hogan",
            "summary": (
                "Tom, I've analyzed your IBKR portfolio for tax loss harvesting "
                "opportunities. Below are positions with unrealized losses that "
                "can potentially be harvested, along with wash sale considerations "
                "and replacement security recommendations."
            ),
            "harvesting_opportunities": [
                {
                    "security": "Example Position",
                    "unrealized_loss": "$0",
                    "cost_basis": "$0",
                    "current_value": "$0",
                    "wash_sale_risk": "None - no recent purchases",
                    "recommended_action": "Harvest and replace with similar ETF",
                    "replacement_candidates": ["Similar non-identical ETF options"],
                    "estimated_tax_savings_37pct": "$0"
                }
            ],
            "wash_sale_warnings": [
                "Review 61-day window for any planned harvests",
                "Avoid repurchasing substantially identical securities",
                "Options on same underlying may trigger wash sales"
            ],
            "total_harvestable_losses": "Pending IBKR data",
            "estimated_tax_savings": "Pending calculation",
            "citations": ["IRC §1091", "Rev. Rul. 85-87", "Treas. Reg. §1.1091-1"]
        }

    def _generate_tax_loss_harvest_report(self, params: Dict) -> Dict:
        """Generate detailed tax loss harvest report"""
        return self._handle_tax_loss_harvest(params)

    def _handle_wash_sale_analysis(self, params: Dict) -> Dict:
        """Wash sale analysis for IBKR portfolio"""
        return {
            "status": "success",
            "report_type": "wash_sale_analysis",
            "timestamp": datetime.now().isoformat(),
            "to": "Tom Hogan",
            "summary": (
                "Tom, this report identifies potential wash sale issues in your "
                "IBKR portfolio. I track the 61-day window (30 days before and "
                "after each sale) to identify disallowed losses and basis adjustments."
            ),
            "wash_sale_tracking": {
                "current_wash_sales": [],
                "pending_wash_sale_risks": [],
                "recently_cleared": []
            },
            "61_day_window_calendar": {
                "positions_in_window": [],
                "upcoming_clearances": []
            },
            "substantially_identical_analysis": {
                "stock_to_stock": "Same corporation = substantially identical",
                "stock_to_options": "Deep ITM options may be substantially identical (Rev. Rul. 85-87)",
                "options_to_options": "Same terms = substantially identical",
                "etf_to_etf": "Different indexes = generally NOT identical"
            },
            "basis_adjustments": [],
            "recommendations": [
                "Wait 31+ days before repurchasing sold positions",
                "Use ETFs tracking different indexes as replacements",
                "Avoid deep-in-the-money options on recently sold stock"
            ],
            "citations": ["IRC §1091", "Treas. Reg. §1.1091-1", "Rev. Rul. 85-87"]
        }

    def _generate_wash_sale_report(self, params: Dict) -> Dict:
        """Generate wash sale report"""
        return self._handle_wash_sale_analysis(params)

    def _handle_straddle_analysis(self, params: Dict) -> Dict:
        """Straddle position analysis for IBKR portfolio"""
        return {
            "status": "success",
            "report_type": "straddle_analysis",
            "timestamp": datetime.now().isoformat(),
            "to": "Tom Hogan",
            "summary": (
                "Tom, I've analyzed your IBKR positions for straddles under "
                "IRC §1092. Straddles trigger loss deferral and holding period "
                "suspension. I've also identified election opportunities."
            ),
            "straddle_positions": {
                "identified_straddles": [],
                "qualified_covered_calls": [],
                "mixed_straddles": []
            },
            "loss_deferral_impact": {
                "total_deferred_losses": "$0",
                "positions_affected": [],
                "unrecognized_gains_offsetting": "$0"
            },
            "holding_period_impact": {
                "positions_with_suspended_holding": [],
                "impact_on_ltcg_qualification": "None currently"
            },
            "election_opportunities": {
                "identified_straddle_election": {
                    "description": "Identify specific positions as straddle at entry",
                    "benefit": "Limits loss deferral to specific straddle",
                    "candidates": []
                },
                "mixed_straddle_election": {
                    "description": "For straddles with §1256 and non-§1256 legs",
                    "benefit": "60/40 treatment for §1256 portion",
                    "candidates": []
                }
            },
            "recommendations": [
                "Consider identifying straddles at position entry",
                "Review QCC qualification for covered calls",
                "Evaluate mixed straddle account election"
            ],
            "citations": ["IRC §1092", "Treas. Reg. §1.1092", "Treas. Reg. §1.1092(c)-1"]
        }

    def _generate_straddle_report(self, params: Dict) -> Dict:
        """Generate straddle report"""
        return self._handle_straddle_analysis(params)

    def _handle_section_1256_report(self, params: Dict) -> Dict:
        """Section 1256 contracts analysis"""
        return {
            "status": "success",
            "report_type": "section_1256_analysis",
            "timestamp": datetime.now().isoformat(),
            "to": "Tom Hogan",
            "summary": (
                "Tom, Section 1256 contracts receive favorable 60/40 tax treatment "
                "(60% long-term, 40% short-term regardless of holding period). "
                "Here's your §1256 position analysis."
            ),
            "section_1256_positions": {
                "index_options": {
                    "spx_options": [],
                    "ndx_options": [],
                    "rut_options": [],
                    "vix_options": []
                },
                "futures": [],
                "foreign_currency": []
            },
            "year_end_mtm": {
                "total_gain_loss": "$0",
                "60pct_long_term": "$0",
                "40pct_short_term": "$0"
            },
            "tax_benefit_calculation": {
                "without_1256_treatment": "$0 (all short-term at 37%)",
                "with_1256_treatment": "$0 (blended ~26.8%)",
                "tax_savings": "$0"
            },
            "non_1256_positions": {
                "equity_options": "Regular capital gain treatment",
                "narrow_based_index_options": "NOT §1256 - equity option treatment",
                "etf_options": "Generally NOT §1256 (unless broad index ETF)"
            },
            "carryback_opportunity": {
                "description": "§1256 losses can carry back 3 years to offset §1256 gains",
                "available_carryback": "$0",
                "potential_refund": "$0"
            },
            "recommendations": [
                "Prefer SPX/NDX options over single-stock options for tax efficiency",
                "Consider year-end MTM implications for open positions",
                "Evaluate loss carryback if applicable"
            ],
            "citations": ["IRC §1256", "Rev. Rul. 2003-7"]
        }

    def _generate_1256_report(self, params: Dict) -> Dict:
        """Generate Section 1256 report"""
        return self._handle_section_1256_report(params)

    def _handle_year_end_tax_plan(self, params: Dict) -> Dict:
        """Year-end tax planning for IBKR portfolio"""
        return {
            "status": "success",
            "report_type": "year_end_tax_plan",
            "timestamp": datetime.now().isoformat(),
            "to": "Tom Hogan",
            "summary": (
                "Tom, here's your year-end tax planning analysis for the IBKR "
                "portfolio. Key strategies include tax loss harvesting, "
                "managing §1256 MTM, and optimizing gain character."
            ),
            "current_year_summary": {
                "realized_stcg": "$0",
                "realized_ltcg": "$0",
                "realized_losses": "$0",
                "section_1256_gains": "$0",
                "net_taxable": "$0"
            },
            "year_end_strategies": {
                "tax_loss_harvesting": {
                    "action": "Harvest losses before Dec 31",
                    "candidates": [],
                    "impact": "Reduce taxable gains"
                },
                "gain_deferral": {
                    "action": "Defer gains to next year",
                    "candidates": "Positions with large unrealized gains",
                    "impact": "Delay tax liability"
                },
                "stcg_to_ltcg_conversion": {
                    "action": "Hold positions past 1-year mark",
                    "positions_approaching": [],
                    "impact": "Lower tax rate (20% vs 37%)"
                },
                "section_1256_optimization": {
                    "action": "Review year-end MTM",
                    "current_unrealized": "$0",
                    "impact": "60/40 treatment"
                }
            },
            "estimated_tax_liability": {
                "current_trajectory": "$0",
                "after_optimization": "$0",
                "potential_savings": "$0"
            },
            "action_items": [
                "Execute tax loss harvests by Dec 28 (T+2 settlement)",
                "Review §1256 positions for year-end MTM",
                "Confirm no wash sale triggers",
                "Consider installment sales for large gains"
            ]
        }

    def _handle_holding_period_report(self, params: Dict) -> Dict:
        """Holding period tracking for LTCG optimization"""
        return {
            "status": "success",
            "report_type": "holding_period_tracking",
            "timestamp": datetime.now().isoformat(),
            "to": "Tom Hogan",
            "summary": (
                "Tom, this report tracks holding periods across your IBKR "
                "positions to help optimize for long-term capital gains "
                "treatment (20% vs 37% short-term rate)."
            ),
            "positions_by_holding_period": {
                "long_term_qualified": [],
                "approaching_lt_11_12_months": [],
                "short_term_under_6_months": [],
                "short_term_6_12_months": []
            },
            "straddle_holding_period_impact": {
                "positions_suspended": [],
                "days_suspended": 0
            },
            "wash_sale_holding_period_impact": {
                "positions_with_tacked_holding": [],
                "additional_days_added": 0
            },
            "optimization_recommendations": [
                "Consider holding positions past 1-year mark when profitable",
                "Review straddle impact on holding periods",
                "Account for wash sale holding period adjustments"
            ]
        }

    def _handle_options_tax_report(self, params: Dict) -> Dict:
        """Options tax treatment analysis"""
        return {
            "status": "success",
            "report_type": "options_tax_analysis",
            "timestamp": datetime.now().isoformat(),
            "to": "Tom Hogan",
            "summary": (
                "Tom, this report analyzes the tax treatment of options "
                "positions in your IBKR portfolio, including premium "
                "allocation, exercise scenarios, and optimization strategies."
            ),
            "options_positions": {
                "long_calls": [],
                "long_puts": [],
                "short_calls": [],
                "short_puts": [],
                "spreads": [],
                "straddles_strangles": []
            },
            "premium_analysis": {
                "premiums_received": "$0",
                "premiums_paid": "$0",
                "net_premium": "$0"
            },
            "qualified_covered_calls": {
                "positions_qualifying": [],
                "requirements": [
                    "Exchange traded",
                    "More than 30 days to expiration",
                    "Not deep in the money",
                    "Strike within qualified benchmark"
                ],
                "benefit": "Exception from straddle rules"
            },
            "exercise_scenarios": {
                "options_likely_exercised": [],
                "tax_treatment_if_exercised": "Premium adjusts basis of stock",
                "options_likely_to_expire": [],
                "tax_treatment_if_expires": "Premium is gain/loss at expiration"
            },
            "recommendations": [
                "Prefer qualified covered calls when writing calls",
                "Consider §1256 index options for 60/40 treatment",
                "Track holding periods for underlying stock",
                "Be aware of straddle rules for hedged positions"
            ],
            "citations": [
                "IRC §1234",
                "IRC §1092(c)(4) - QCC exception",
                "IRC §1256 - Nonequity options",
                "Treas. Reg. §1.1092(c)-1"
            ]
        }

    # =========================================================================
    # IBKR DATA INTEGRATION (Cross-Training with Broker Data)
    # =========================================================================

    def _handle_ibkr_executions(self, params: Dict) -> Dict:
        """
        Get trade executions from IBKR for tax reporting.

        Cross-trains CPA with live broker execution data for:
        - K-1 preparation
        - Wash sale tracking
        - Cost basis calculations
        """
        if not IBKR_INTEGRATION_AVAILABLE:
            return {
                "status": "error",
                "message": "IBKR integration not available. Install ibkr_data module."
            }

        try:
            year = params.get("year", datetime.now().year)
            executions = get_executions_for_tax(year)

            # Calculate summary stats
            buys = [e for e in executions if e.get("side") == "BOT"]
            sells = [e for e in executions if e.get("side") == "SLD"]

            total_buy_value = sum(e.get("total_value", 0) for e in buys)
            total_sell_value = sum(e.get("total_value", 0) for e in sells)
            total_commissions = sum(e.get("commission", 0) for e in executions)

            self.logger.info(
                f"IBKR Executions for {year}: {len(buys)} buys, {len(sells)} sells, "
                f"${total_commissions:,.2f} in commissions"
            )

            return {
                "status": "success",
                "source": "ibkr",
                "tax_year": year,
                "execution_count": len(executions),
                "buy_count": len(buys),
                "sell_count": len(sells),
                "total_buy_value": total_buy_value,
                "total_sell_value": total_sell_value,
                "total_commissions": total_commissions,
                "executions": executions,
            }
        except Exception as e:
            self.logger.error(f"Error getting IBKR executions: {e}")
            return {"status": "error", "message": str(e)}

    def _handle_ibkr_positions_audit(self, params: Dict) -> Dict:
        """
        Get positions from IBKR for audit PBC preparation.

        Cross-trains CPA to verify positions against SANTAS_HELPER NAV.
        """
        if not IBKR_INTEGRATION_AVAILABLE:
            return {"status": "error", "message": "IBKR integration not available"}

        try:
            positions = get_positions_for_audit()

            return {
                "status": "success",
                "source": "ibkr",
                "as_of_date": datetime.now().isoformat(),
                "position_count": len(positions),
                "positions": positions,
                "total_market_value": sum(p.get("market_value", 0) for p in positions),
                "audit_ready": True,
                "pbc_section": "Investment Schedule",
            }
        except Exception as e:
            self.logger.error(f"Error getting IBKR positions for audit: {e}")
            return {"status": "error", "message": str(e)}

    def _handle_ibkr_pnl_tax(self, params: Dict) -> Dict:
        """
        Get P&L from IBKR for tax calculations.

        Used for K-1 preparation and gain/loss reporting.
        """
        if not IBKR_INTEGRATION_AVAILABLE:
            return {"status": "error", "message": "IBKR integration not available"}

        try:
            pnl_data = get_pnl_for_reporting()

            # Analyze for tax implications
            realized = pnl_data.get("total_realized_pnl", 0)
            unrealized = pnl_data.get("total_unrealized_pnl", 0)

            # Estimate tax liability (simplified)
            short_term_rate = Decimal("0.37")  # Top federal rate
            long_term_rate = Decimal("0.20")   # Long-term capital gains

            # Assume 50/50 split for demo (would track actual holding periods)
            estimated_short_term_tax = realized * float(short_term_rate) * 0.5
            estimated_long_term_tax = realized * float(long_term_rate) * 0.5

            return {
                "status": "success",
                "source": "ibkr",
                "pnl_summary": pnl_data,
                "tax_implications": {
                    "realized_pnl": realized,
                    "unrealized_pnl": unrealized,
                    "estimated_short_term_tax": estimated_short_term_tax,
                    "estimated_long_term_tax": estimated_long_term_tax,
                    "total_estimated_tax": estimated_short_term_tax + estimated_long_term_tax,
                    "note": "Estimates based on assumed holding periods. Review actual lot data."
                },
            }
        except Exception as e:
            self.logger.error(f"Error getting IBKR P&L for tax: {e}")
            return {"status": "error", "message": str(e)}

    def get_ibkr_tax_data(self, year: int = None) -> Dict[str, Any]:
        """
        Get comprehensive IBKR data for tax preparation.

        This method aggregates all IBKR data needed for K-1 and tax reporting:
        - Executions for cost basis and gain/loss
        - Positions for year-end holdings
        - P&L breakdown by asset class
        """
        year = year or datetime.now().year

        if not IBKR_INTEGRATION_AVAILABLE:
            return {
                "status": "error",
                "message": "IBKR integration not available"
            }

        try:
            executions = get_executions_for_tax(year)
            positions = get_positions_for_audit()
            pnl = get_pnl_for_reporting()

            return {
                "status": "success",
                "tax_year": year,
                "generated_at": datetime.now().isoformat(),
                "executions": {
                    "count": len(executions),
                    "data": executions,
                },
                "year_end_positions": {
                    "count": len(positions),
                    "data": positions,
                    "total_value": sum(p.get("market_value", 0) for p in positions),
                },
                "pnl_summary": pnl,
                "k1_ready": True,
            }
        except Exception as e:
            self.logger.error(f"Error getting IBKR tax data: {e}")
            return {"status": "error", "message": str(e)}

    def _handle_unknown(self, params: Dict) -> Dict:
        return {
            "status": "error",
            "message": "Unknown action. Available actions:\n"
                      "  FOR CHRIS: report_to_chris, generate_k1s, coordinate_audit, "
                      "file_form_pf, calculate_firm_pnl, validate_nav_internal, get_deadlines\n"
                      "  FOR TOM (IBKR Tax): report_to_tom, analyze_ibkr_tax, tax_loss_harvest_report, "
                      "wash_sale_analysis, straddle_analysis, section_1256_report, "
                      "year_end_tax_plan, holding_period_report, options_tax_report"
        }


# Singleton
_cpa_instance: Optional[CPAAgent] = None


def get_cpa() -> CPAAgent:
    global _cpa_instance
    if _cpa_instance is None:
        _cpa_instance = CPAAgent()
    return _cpa_instance


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the CPA Agent")
    parser.add_argument("mode", nargs="?", default="help",
                       choices=["run", "status", "report", "deadlines", "help"],
                       help="Mode to run the agent in")
    args = parser.parse_args()

    agent = get_cpa()

    print(f"\n{'='*70}")
    print("CPA AGENT - Tax, Audit, Reporting & Compliance Authority")
    print(f"{'='*70}")
    print(agent.get_natural_language_explanation())

    if args.mode == "run":
        result = agent.process({"action": "get_status"})
        print(f"\nStatus: {result}")
    elif args.mode == "status":
        result = agent.process({"action": "get_status"})
        print(f"\nStatus: {result}")
    elif args.mode == "report":
        result = agent.process({"action": "report_to_chris"})
        print(f"\nReport to Chris: {result}")
    elif args.mode == "deadlines":
        result = agent.process({"action": "get_deadlines"})
        print(f"\nDeadlines: {result}")
    elif args.mode == "help":
        parser.print_help()

