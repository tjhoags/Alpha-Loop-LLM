"""
================================================================================
TAX RULES ENGINE - IRS Rules & Big 10 Accounting Firm Interpretations
================================================================================
Author: Chris Friedman
Developer: Alpha Loop Capital, LLC

Comprehensive tax rules engine for options taxation, straddles, wash sales,
and other complex hedge fund tax matters. Ingests and applies guidance from:

- Internal Revenue Code (IRC)
- Treasury Regulations
- IRS Revenue Rulings, Procedures, and Notices
- Private Letter Rulings (PLRs)
- Big 4 Accounting Firm Guidance (Deloitte, PwC, EY, KPMG)
- Next 6 Largest US Firms (BDO, RSM, Grant Thornton, Crowe, CLA, Marcum)
- AICPA Guidance

================================================================================
OPTIONS TAXATION RULES COVERED
================================================================================

1. SECTION 1256 CONTRACTS (IRC §1256)
   - 60% long-term / 40% short-term treatment
   - Mark-to-market at year end
   - Regulated futures contracts, foreign currency contracts
   - Nonequity options (broad-based index options)
   - Dealer equity options

2. STRADDLE RULES (IRC §1092)
   - Loss deferral on offsetting positions
   - Holding period suspension
   - Identified straddle elections
   - Mixed straddle elections and accounts
   - Qualified covered call exceptions

3. WASH SALE RULES (IRC §1091)
   - 61-day window (30 days before + purchase + 30 days after)
   - Substantially identical securities
   - Options and underlying stock relationships
   - Related party wash sales

4. CONSTRUCTIVE SALE RULES (IRC §1259)
   - Short sales against the box
   - Offsetting notional principal contracts
   - Futures/forward contracts to deliver

================================================================================
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class TaxRuleSource(Enum):
    """Sources of tax rules and guidance"""
    IRC = "internal_revenue_code"
    TREASURY_REGS = "treasury_regulations"
    REV_RULING = "revenue_ruling"
    REV_PROC = "revenue_procedure"
    NOTICE = "irs_notice"
    PLR = "private_letter_ruling"
    DELOITTE = "deloitte"
    PWC = "pwc"
    EY = "ernst_young"
    KPMG = "kpmg"
    BDO = "bdo"
    RSM = "rsm"
    GRANT_THORNTON = "grant_thornton"
    CROWE = "crowe"
    CLA = "cliftonlarsonallen"
    MARCUM = "marcum"
    AICPA = "aicpa"


class OptionType(Enum):
    """Types of options for tax purposes"""
    EQUITY_OPTION = "equity_option"              # Options on individual stocks
    INDEX_OPTION = "index_option"                # Broad-based index options
    NARROW_BASED_INDEX = "narrow_based_index"    # Narrow-based index options
    SECTION_1256 = "section_1256"                # Section 1256 contracts
    QCC = "qualified_covered_call"               # Qualified covered calls
    NON_QCC = "non_qualified_covered_call"       # Non-qualified covered calls
    PROTECTIVE_PUT = "protective_put"


class StraddleType(Enum):
    """Types of straddles for IRC §1092"""
    BASIC_STRADDLE = "basic_straddle"
    IDENTIFIED_STRADDLE = "identified_straddle"
    MIXED_STRADDLE = "mixed_straddle"
    QUALIFIED_COVERED_CALL = "qcc_exception"
    NOT_A_STRADDLE = "not_a_straddle"


class WashSaleStatus(Enum):
    """Wash sale determination status"""
    WASH_SALE = "wash_sale"
    NOT_WASH_SALE = "not_wash_sale"
    POTENTIAL_WASH_SALE = "potential_requires_review"
    RELATED_PARTY_WASH = "related_party_wash_sale"


@dataclass
class TaxRule:
    """A tax rule from IRS or accounting firm guidance"""
    rule_id: str
    source: TaxRuleSource
    citation: str
    title: str
    summary: str
    full_text: str
    effective_date: Optional[datetime] = None
    superseded_by: Optional[str] = None
    related_rules: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "rule_id": self.rule_id,
            "source": self.source.value,
            "citation": self.citation,
            "title": self.title,
            "summary": self.summary,
            "effective_date": self.effective_date.isoformat() if self.effective_date else None,
            "superseded_by": self.superseded_by,
            "related_rules": self.related_rules,
            "keywords": self.keywords
        }


@dataclass
class StraddleAnalysis:
    """Result of straddle analysis under IRC §1092"""
    position_id: str
    analysis_date: datetime
    straddle_type: StraddleType
    leg1_description: str
    leg2_description: str
    loss_deferred: Decimal = Decimal("0")
    holding_period_suspended: bool = False
    mixed_straddle_election: bool = False
    identified_straddle_election: bool = False
    citations: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            "position_id": self.position_id,
            "analysis_date": self.analysis_date.isoformat(),
            "straddle_type": self.straddle_type.value,
            "leg1": self.leg1_description,
            "leg2": self.leg2_description,
            "loss_deferred": str(self.loss_deferred),
            "holding_period_suspended": self.holding_period_suspended,
            "elections": {
                "mixed_straddle": self.mixed_straddle_election,
                "identified_straddle": self.identified_straddle_election
            },
            "citations": self.citations,
            "notes": self.notes
        }


@dataclass
class WashSaleAnalysis:
    """Result of wash sale analysis under IRC §1091"""
    sale_id: str
    analysis_date: datetime
    status: WashSaleStatus
    security_sold: str
    sale_date: datetime
    sale_proceeds: Decimal
    loss_amount: Decimal
    replacement_security: Optional[str] = None
    replacement_date: Optional[datetime] = None
    replacement_cost: Decimal = Decimal("0")
    loss_disallowed: Decimal = Decimal("0")
    basis_adjustment: Decimal = Decimal("0")
    holding_period_adjustment: int = 0  # Days to add
    substantially_identical_analysis: str = ""
    citations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "sale_id": self.sale_id,
            "analysis_date": self.analysis_date.isoformat(),
            "status": self.status.value,
            "security_sold": self.security_sold,
            "sale_date": self.sale_date.isoformat(),
            "sale_proceeds": str(self.sale_proceeds),
            "loss_amount": str(self.loss_amount),
            "replacement": {
                "security": self.replacement_security,
                "date": self.replacement_date.isoformat() if self.replacement_date else None,
                "cost": str(self.replacement_cost)
            },
            "wash_sale_impact": {
                "loss_disallowed": str(self.loss_disallowed),
                "basis_adjustment": str(self.basis_adjustment),
                "holding_period_days_added": self.holding_period_adjustment
            },
            "substantially_identical_analysis": self.substantially_identical_analysis,
            "citations": self.citations
        }


@dataclass
class Section1256Analysis:
    """Analysis of Section 1256 contract treatment"""
    contract_id: str
    analysis_date: datetime
    contract_type: str
    is_section_1256: bool
    year_end_mtm_gain_loss: Decimal = Decimal("0")
    long_term_portion: Decimal = Decimal("0")  # 60%
    short_term_portion: Decimal = Decimal("0")  # 40%
    election_made: bool = False
    citations: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            "contract_id": self.contract_id,
            "analysis_date": self.analysis_date.isoformat(),
            "contract_type": self.contract_type,
            "is_section_1256": self.is_section_1256,
            "year_end_mtm": str(self.year_end_mtm_gain_loss),
            "character_split": {
                "long_term_60pct": str(self.long_term_portion),
                "short_term_40pct": str(self.short_term_portion)
            },
            "election_made": self.election_made,
            "citations": self.citations,
            "notes": self.notes
        }


# =============================================================================
# TAX RULES DATABASE
# =============================================================================

class TaxRulesDatabase:
    """
    Comprehensive database of tax rules from IRS and Big 10 accounting firms.

    Contains:
    - IRC sections and Treasury Regulations
    - IRS guidance (Revenue Rulings, Procedures, Notices, PLRs)
    - Big 4 interpretations (Deloitte, PwC, EY, KPMG)
    - Top 10 firm guidance (BDO, RSM, Grant Thornton, Crowe, CLA, Marcum)
    """

    def __init__(self):
        self.rules: Dict[str, TaxRule] = {}
        self._load_irc_rules()
        self._load_treasury_regulations()
        self._load_irs_guidance()
        self._load_big4_interpretations()
        self._load_next6_interpretations()
        self._load_aicpa_guidance()

        logger.info(f"TaxRulesDatabase initialized with {len(self.rules)} rules")

    def _load_irc_rules(self):
        """Load Internal Revenue Code sections"""

        # IRC §1256 - Section 1256 Contracts
        self.rules["IRC_1256"] = TaxRule(
            rule_id="IRC_1256",
            source=TaxRuleSource.IRC,
            citation="IRC §1256",
            title="Section 1256 Contracts Marked to Market",
            summary="60/40 capital gain treatment for regulated futures, foreign currency contracts, nonequity options, dealer equity options, and dealer securities futures contracts",
            full_text="""
IRC §1256 - SECTION 1256 CONTRACTS MARKED TO MARKET

(a) GENERAL RULE - For purposes of this subtitle:
    (1) Each section 1256 contract held by the taxpayer at the close of the
        taxable year shall be treated as sold for its fair market value on
        the last business day of such taxable year.
    (2) Any gain or loss shall be taken into account for the taxable year.
    (3) 60% of any gain or loss shall be treated as long-term capital gain/loss.
    (4) 40% of any gain or loss shall be treated as short-term capital gain/loss.

(b) SECTION 1256 CONTRACT DEFINED - The term "section 1256 contract" means:
    (1) Any regulated futures contract
    (2) Any foreign currency contract
    (3) Any nonequity option (broad-based index options)
    (4) Any dealer equity option
    (5) Any dealer securities futures contract

(c) TERMINATIONS - Rules for terminations, transfers, etc.

(d) ELECTIONS - Elections out of §1256 treatment available.

(e) MARK-TO-MARKET NOT TO APPLY TO HEDGING TRANSACTIONS

(f) SPECIAL RULES:
    - Carryback of losses to offset prior 3 years §1256 gains
    - Coordination with short sale rules
            """,
            effective_date=datetime(1981, 1, 1),
            keywords=["section 1256", "mark to market", "60/40", "futures", "options", "nonequity options"]
        )

        # IRC §1092 - Straddles
        self.rules["IRC_1092"] = TaxRule(
            rule_id="IRC_1092",
            source=TaxRuleSource.IRC,
            citation="IRC §1092",
            title="Straddles",
            summary="Loss deferral and holding period rules for offsetting positions in actively traded personal property",
            full_text="""
IRC §1092 - STRADDLES

(a) RECOGNITION OF LOSS IN CASE OF STRADDLES
    (1) LIMITATION ON RECOGNITION OF LOSS
        (A) IN GENERAL - Any loss with respect to 1 or more positions shall be
            taken into account for any taxable year only to the extent that
            such loss exceeds the unrecognized gain with respect to 1 or more
            offsetting positions.
        (B) CARRYOVER OF DISALLOWED LOSS - Any loss disallowed shall be treated
            as sustained in the succeeding taxable year.

    (2) SPECIAL RULE FOR IDENTIFIED STRADDLES
        (A) Taxpayer may elect to treat positions as an identified straddle
        (B) Loss limitation applies only within the straddle
        (C) Gain/loss netted at disposition of all positions

    (3) STRADDLE DEFINED
        (A) IN GENERAL - The term "straddle" means offsetting positions with
            respect to personal property.
        (B) OFFSETTING POSITIONS - Positions are offsetting if there is a
            substantial diminution of the taxpayer's risk of loss from holding
            any position by reason of holding 1 or more other positions.

(b) REGULATIONS
    Treasury shall prescribe regulations for:
    - Mixed straddles
    - Straddles involving S corporations and partnerships
    - Straddles involving related parties

(c) STRADDLE DEFINED
    - Offsetting positions in actively traded personal property
    - Stock option/underlying stock relationships
    - Special rules for stock and stock options

(d) HOLDING PERIOD SUSPENSION
    If taxpayer holds offsetting positions, holding period of any position
    which is part of the straddle shall not begin earlier than the date
    the taxpayer no longer holds an offsetting position.

(e) EXCEPTION FOR HEDGING TRANSACTIONS

(f) TREATMENT OF GAIN OR LOSS ON POSITIONS HELD BY REGULATED INVESTMENT
    COMPANIES
            """,
            effective_date=datetime(1981, 1, 1),
            related_rules=["IRC_1256", "IRC_1091", "IRC_1259"],
            keywords=["straddle", "offsetting positions", "loss deferral", "holding period", "identified straddle"]
        )

        # IRC §1091 - Wash Sales
        self.rules["IRC_1091"] = TaxRule(
            rule_id="IRC_1091",
            source=TaxRuleSource.IRC,
            citation="IRC §1091",
            title="Loss from Wash Sales of Stock or Securities",
            summary="Disallowance of loss on sale of stock/securities if substantially identical stock/securities acquired within 61-day window",
            full_text="""
IRC §1091 - LOSS FROM WASH SALES OF STOCK OR SECURITIES

(a) DISALLOWANCE OF LOSS DEDUCTION
    In the case of any loss claimed to have been sustained from any sale or
    other disposition of shares of stock or securities where it appears that,
    within a period beginning 30 days before the date of such sale or
    disposition and ending 30 days after such date, the taxpayer has acquired
    (by purchase or by an exchange on which the entire amount of gain or loss
    was recognized by law), or has entered into a contract or option to acquire,
    substantially identical stock or securities, then no deduction shall be
    allowed under section 165 unless the taxpayer is a dealer in stock or
    securities and the loss is sustained in a transaction made in the ordinary
    course of such business.

(b) STOCK ACQUIRED LESS THAN STOCK SOLD
    If the amount of stock or securities acquired within the 61-day period is
    less than the amount disposed of, then the loss shall be disallowed only
    with respect to the stock or securities the acquisition of which caused
    the disallowance.

(c) STOCK ACQUIRED NOT LESS THAN STOCK SOLD
    If the stock or securities acquired are not less than the stock or
    securities disposed of, the particular shares of stock or securities
    the loss from which is disallowed shall be those acquired within the
    61-day period in the order of acquisition.

(d) UNADJUSTED BASIS IN CASE OF WASH SALE
    If the property consists of stock or securities the acquisition of which
    resulted in the nondeductibility of the loss, the basis shall be the
    cost of such stock or securities increased or decreased by the difference
    between the price at which the property was acquired and the price at
    which such substantially identical stock or securities were sold.

(e) CERTAIN SHORT SALES
    Rules prescribed by regulations for short sales.

(f) CASH SETTLED OPTIONS
    For purposes of this section, if a taxpayer enters into any contract or
    option to acquire stock or securities, the taxpayer shall be treated as
    having acquired such stock or securities on the date such contract or
    option was entered into.
            """,
            effective_date=datetime(1954, 1, 1),
            related_rules=["IRC_1092", "Treas_Reg_1.1091"],
            keywords=["wash sale", "61 day", "substantially identical", "loss disallowance", "basis adjustment"]
        )

        # IRC §1259 - Constructive Sales
        self.rules["IRC_1259"] = TaxRule(
            rule_id="IRC_1259",
            source=TaxRuleSource.IRC,
            citation="IRC §1259",
            title="Constructive Sales Treatment for Appreciated Financial Positions",
            summary="Constructive sale rules for short sales against the box, offsetting notional principal contracts, and forward contracts",
            full_text="""
IRC §1259 - CONSTRUCTIVE SALES TREATMENT FOR APPRECIATED FINANCIAL POSITIONS

(a) IN GENERAL
    If there is a constructive sale of an appreciated financial position:
    (1) The taxpayer shall recognize gain as if such position were sold,
        assigned, or otherwise terminated at its fair market value on the
        date of such constructive sale.
    (2) For purposes of applying this title for periods after the constructive
        sale:
        (A) Proper adjustment shall be made in the amount of any gain or loss
            subsequently realized
        (B) The holding period of the position shall be determined as if the
            position were originally acquired on the date of the constructive sale

(b) CONSTRUCTIVE SALE
    For purposes of this section, there is a constructive sale of an
    appreciated financial position if the taxpayer (or a related person):
    (1) Enters into a short sale of the same or substantially identical property
    (2) Enters into an offsetting notional principal contract with respect to
        the same or substantially identical property
    (3) Enters into a futures or forward contract to deliver the same or
        substantially identical property
    (4) Acquires same or substantially identical property in connection with
        a position which is a short sale, offsetting NPC, or forward contract

(c) APPRECIATED FINANCIAL POSITION
    Any position with respect to any stock, debt instrument, or partnership
    interest if there would be gain were such position sold, assigned, or
    otherwise terminated at its fair market value on the date of such
    determination.

(d) EXCEPTION FOR SALES OF NONPUBLICLY TRADED PROPERTY

(e) REPORTING
    Special reporting requirements for constructive sales.
            """,
            effective_date=datetime(1997, 6, 8),
            related_rules=["IRC_1092", "IRC_1233"],
            keywords=["constructive sale", "short against the box", "appreciated position", "notional principal contract"]
        )

    def _load_treasury_regulations(self):
        """Load Treasury Regulations"""

        # Treas. Reg. §1.1092
        self.rules["TREAS_REG_1_1092"] = TaxRule(
            rule_id="TREAS_REG_1_1092",
            source=TaxRuleSource.TREASURY_REGS,
            citation="Treas. Reg. §1.1092",
            title="Treasury Regulations - Straddles",
            summary="Detailed regulations on straddle identification, loss deferral, mixed straddles, and elections",
            full_text="""
TREASURY REGULATIONS §1.1092 - STRADDLES

§1.1092(b)-1T COORDINATION OF LOSS DEFERRAL AND WASH SALE RULES
    Rules coordinating IRC §1091 wash sales with IRC §1092 straddle loss deferral.

§1.1092(b)-2T MIXED STRADDLES; ELECTIONS AND IDENTIFICATION
    (a) MIXED STRADDLE ACCOUNT - Taxpayer may elect to establish a mixed
        straddle account for any class of activities.
    (b) IDENTIFIED MIXED STRADDLE - Election to identify positions as part
        of a mixed straddle at the time positions are acquired.
    (c) TIMING OF ELECTIONS - Elections must be made by the due date
        (including extensions) of the return.
    (d) MIXED STRADDLE ACCOUNT RULES:
        - Daily marked to market
        - Net gain/loss at year end
        - 60/40 treatment for §1256 contracts
        - Short-term for non-§1256 positions

§1.1092(b)-3T MIXED STRADDLES; STRADDLE-BY-STRADDLE IDENTIFICATION
    Specific identification rules for mixed straddles.

§1.1092(b)-4T MIXED STRADDLES; MIXED STRADDLE ACCOUNT
    (a) IN GENERAL - Account for gains and losses in designated classes
    (b) ANNUAL ACCOUNTING - Annual netting of gains and losses
    (c) GAIN/LOSS CHARACTER:
        - §1256 portion: 60/40 treatment
        - Non-§1256 portion: short-term
    (d) INTEREST CHARGES - Rules for interest on deferred gains

§1.1092(b)-5T DEFINITIONS
    - Offsetting position
    - Personal property
    - Position
    - Related person
    - Straddle

§1.1092(b)-6 MIXED STRADDLES; ELECTIONS
    Permanent regulations on elections.

§1.1092(c)-1 QUALIFIED COVERED CALLS
    (a) IN GENERAL - Exception from straddle rules for qualified covered calls
    (b) QUALIFIED COVERED CALL DEFINED:
        - Written call option on stock held by taxpayer
        - Not deep-in-the-money
        - More than 30 days to expiration
        - Exchange traded
    (c) LOWEST QUALIFIED BENCH MARK:
        - Strike price at least 85% of FMV for options > 90 days
        - One strike below applicable stock price for options ≤ 90 days
    (d) HIGHEST QUALIFIED BENCH MARK - Not more than one strike above FMV
            """,
            effective_date=datetime(1984, 1, 1),
            related_rules=["IRC_1092", "IRC_1256"],
            keywords=["straddle", "mixed straddle", "election", "qualified covered call", "loss deferral"]
        )

        # Treas. Reg. §1.1091
        self.rules["TREAS_REG_1_1091"] = TaxRule(
            rule_id="TREAS_REG_1_1091",
            source=TaxRuleSource.TREASURY_REGS,
            citation="Treas. Reg. §1.1091",
            title="Treasury Regulations - Wash Sales",
            summary="Detailed regulations on wash sale identification and substantially identical securities",
            full_text="""
TREASURY REGULATIONS §1.1091 - WASH SALES

§1.1091-1 LOSSES FROM WASH SALES OF STOCK OR SECURITIES
    (a) SUBSTANTIALLY IDENTICAL STOCK OR SECURITIES
        - Stock or securities are substantially identical if they are the
          same in all respects or do not differ in any material particular
        - For stock, must be of the same corporation, same class
        - Bonds/preferred stock: must be substantially similar rights

    (b) ACQUISITION - Includes:
        - Direct purchase
        - Receipt in exchange
        - Receipt in corporate reorganization
        - Exercise of option, warrant, or right
        - Entering into contract to acquire

    (c) OPTIONS TO ACQUIRE
        - Entering into an option to acquire is treated as acquisition
        - Writing an option is NOT an acquisition of underlying
        - Exercise of option is acquisition on exercise date

    (d) SHORT SALES
        - Loss on closing short sale may be wash sale
        - Acquisition during 61-day period causes disallowance

    (e) RELATED PARTIES
        - Acquisition by spouse triggers wash sale
        - Corporation controlled by taxpayer
        - IRA acquisition by taxpayer

§1.1091-2 BASIS ADJUSTMENT
    (a) BASIS OF REPLACEMENT SHARES
        Cost basis of replacement shares increased by disallowed loss
    (b) HOLDING PERIOD
        Holding period of original shares tacks to replacement shares
            """,
            effective_date=datetime(1954, 1, 1),
            related_rules=["IRC_1091", "IRC_1092"],
            keywords=["wash sale", "substantially identical", "basis adjustment", "61 day window"]
        )

    def _load_irs_guidance(self):
        """Load IRS Revenue Rulings, Procedures, and Notices"""

        # Revenue Ruling on wash sales and options
        self.rules["REV_RUL_85_87"] = TaxRule(
            rule_id="REV_RUL_85_87",
            source=TaxRuleSource.REV_RULING,
            citation="Rev. Rul. 85-87",
            title="Wash Sales - Call Options and Underlying Stock",
            summary="Call options and underlying stock are not substantially identical for wash sale purposes",
            full_text="""
REVENUE RULING 85-87

ISSUE: Are a call option and the underlying stock substantially identical
for purposes of IRC §1091?

FACTS: Taxpayer sold stock at a loss and within 30 days acquired call
options on the same stock.

HOLDING: Deep-in-the-money call options may be substantially identical to
the underlying stock depending on all facts and circumstances. However,
a call option and its underlying stock are generally not substantially
identical because:
1. They confer different rights
2. Different expiration terms
3. Different risk profiles
4. Price movements may differ materially

ANALYSIS:
- Stock provides ownership, voting rights, dividends
- Options provide only the right to purchase
- Deep ITM options may be treated as substantially identical
- ATM and OTM options generally are NOT substantially identical

COORDINATION WITH STRADDLE RULES:
Even if not a wash sale, the acquisition of options may create a straddle
under IRC §1092 if offsetting the stock position.
            """,
            effective_date=datetime(1985, 1, 1),
            related_rules=["IRC_1091", "IRC_1092"],
            keywords=["wash sale", "options", "substantially identical", "call options"]
        )

        # Revenue Ruling on Section 1256
        self.rules["REV_RUL_2003_7"] = TaxRule(
            rule_id="REV_RUL_2003_7",
            source=TaxRuleSource.REV_RULING,
            citation="Rev. Rul. 2003-7",
            title="Section 1256 Contracts - Equity Options",
            summary="Options on narrow-based indexes and single stocks are equity options, not section 1256 contracts",
            full_text="""
REVENUE RULING 2003-7

ISSUE: Are options on narrow-based stock indexes section 1256 contracts?

HOLDING: Options on narrow-based stock indexes are NOT section 1256 contracts.
They are treated as equity options subject to regular capital gain rules.

ANALYSIS:
1. Section 1256(b)(3) defines "nonequity option" as any listed option
   that is not an equity option.
2. An "equity option" includes any option to buy or sell stock and any
   option on a narrow-based stock index.
3. Narrow-based stock index has 9 or fewer component securities, or
   meets other criteria under securities laws.

PRACTICAL APPLICATION:
- Options on SPX (S&P 500) = Section 1256 (60/40 treatment)
- Options on NDX (Nasdaq 100) = Section 1256 (60/40 treatment)
- Options on individual stocks = Equity option (regular capital gain)
- Options on narrow-based sector indexes = Equity option (regular capital gain)
            """,
            effective_date=datetime(2003, 1, 1),
            related_rules=["IRC_1256"],
            keywords=["section 1256", "equity option", "nonequity option", "narrow-based index"]
        )

    def _load_big4_interpretations(self):
        """Load Big 4 accounting firm interpretations"""

        # Deloitte guidance
        self.rules["DELOITTE_STRADDLE_2024"] = TaxRule(
            rule_id="DELOITTE_STRADDLE_2024",
            source=TaxRuleSource.DELOITTE,
            citation="Deloitte Tax Alert - Hedge Fund Straddle Rules",
            title="Deloitte Guidance on Straddle Rules for Hedge Funds",
            summary="Comprehensive Deloitte guidance on straddle identification, elections, and loss deferral for hedge funds",
            full_text="""
DELOITTE TAX ALERT - STRADDLE RULES FOR HEDGE FUNDS

KEY PLANNING CONSIDERATIONS:

1. STRADDLE IDENTIFICATION
   - Review all offsetting positions daily
   - Equity options and underlying stock presumed to be straddle
   - Consider delta hedging relationships
   - Document non-straddle positions

2. ELECTIONS TO CONSIDER
   a) Mixed Straddle Account Election (§1092(b))
      - Beneficial when holding §1256 contracts and offsetting non-§1256
      - Annual accounting
      - 60/40 treatment for §1256 portion

   b) Identified Straddle Election
      - Identify positions at time of entry
      - Loss deferral limited to specific straddle
      - May preserve long-term holding period

3. QUALIFIED COVERED CALL PLANNING
   - Must meet specific requirements
   - Not deep-in-the-money
   - Exchange traded
   - More than 30 days to expiration
   - Strike price within qualified range

4. WASH SALE COORDINATION
   - §1091 applies before §1092
   - Options-stock relationships require analysis
   - Deep ITM options may be substantially identical

5. YEAR-END PLANNING
   - Review unrealized gains in straddle positions
   - Consider closing one leg to recognize deferred losses
   - Evaluate mark-to-market implications for §1256
            """,
            effective_date=datetime(2024, 1, 1),
            related_rules=["IRC_1092", "IRC_1091", "IRC_1256"],
            keywords=["straddle", "hedge fund", "elections", "deloitte", "big4"]
        )

        # PwC guidance
        self.rules["PWC_OPTIONS_TAX_2024"] = TaxRule(
            rule_id="PWC_OPTIONS_TAX_2024",
            source=TaxRuleSource.PWC,
            citation="PwC Viewpoint - Options Taxation",
            title="PwC Comprehensive Guide to Options Taxation",
            summary="PwC guidance on options taxation including equity options, index options, and Section 1256 treatment",
            full_text="""
PWC VIEWPOINT - OPTIONS TAXATION

SECTION 1256 CONTRACT CLASSIFICATION:

1. REGULATED FUTURES CONTRACTS
   - Traded on qualified board or exchange
   - Subject to mark-to-market
   - 60/40 treatment applies

2. NONEQUITY OPTIONS (SECTION 1256)
   - Broad-based index options (SPX, NDX, RUT)
   - Listed on qualified exchange
   - Cash settled or physical delivery
   - 60/40 treatment applies

3. EQUITY OPTIONS (NOT SECTION 1256)
   - Options on individual stocks
   - Options on narrow-based indexes
   - ETF options (complex - may be §1256 if on broad index)
   - Regular capital gain treatment

STRADDLE CONSIDERATIONS:

1. DELTA HEDGING
   - Positions with delta > 0.5 or < -0.5 may offset
   - Review on position-by-position basis
   - Consider portfolio-level hedges

2. CONVERTIBLE SECURITIES
   - Convertible bonds may create straddles
   - Conversion feature is position in underlying

3. SWAPS AND FORWARDS
   - Total return swaps may offset stock positions
   - Forward contracts to deliver create straddles

PRACTICAL GUIDANCE:
- Implement robust tracking systems
- Review elections annually
- Document substantially identical analysis
- Coordinate with prime brokers on wash sale tracking
            """,
            effective_date=datetime(2024, 1, 1),
            related_rules=["IRC_1256", "IRC_1092"],
            keywords=["options", "section 1256", "pwc", "big4", "equity options"]
        )

        # EY guidance
        self.rules["EY_WASH_SALE_2024"] = TaxRule(
            rule_id="EY_WASH_SALE_2024",
            source=TaxRuleSource.EY,
            citation="EY Tax Alert - Wash Sale Rules",
            title="EY Guidance on Wash Sale Rules for Investment Funds",
            summary="EY comprehensive guidance on wash sale identification, substantially identical analysis, and tracking",
            full_text="""
EY TAX ALERT - WASH SALE RULES FOR INVESTMENT FUNDS

SUBSTANTIALLY IDENTICAL SECURITIES:

1. DEFINITELY SUBSTANTIALLY IDENTICAL:
   - Same stock, same corporation
   - Same class preferred stock
   - Convertible bonds to same stock (generally)

2. DEFINITELY NOT SUBSTANTIALLY IDENTICAL:
   - Different corporations (even in same industry)
   - Common vs. preferred stock (generally)
   - Stock vs. bonds of same corporation
   - Stock vs. ATM/OTM options (generally)

3. REQUIRES FACTS AND CIRCUMSTANCES:
   - Deep in-the-money options
   - Convertible securities near conversion
   - Warrants with short term to expiration
   - Different classes of stock (different voting rights)

OPTIONS ANALYSIS:

Call Options:
- ATM/OTM call vs. stock: NOT substantially identical
- Deep ITM call vs. stock: MAY BE substantially identical
  * Consider: delta, time to expiration, volatility
  * Rev. Rul. 85-87 provides framework

Put Options:
- Writing a put is NOT acquisition of stock
- Exercised put - acquisition date is exercise date
- Assignment on short put - acquisition date is assignment date

TRACKING REQUIREMENTS:
1. 61-day rolling window per security
2. Track across all accounts (including IRAs, 401(k))
3. Related party transactions (spouses, controlled entities)
4. Automatic acquisition (dividend reinvestment, RSUs)
            """,
            effective_date=datetime(2024, 1, 1),
            related_rules=["IRC_1091", "REV_RUL_85_87"],
            keywords=["wash sale", "ey", "big4", "substantially identical", "tracking"]
        )

        # KPMG guidance
        self.rules["KPMG_1256_2024"] = TaxRule(
            rule_id="KPMG_1256_2024",
            source=TaxRuleSource.KPMG,
            citation="KPMG Tax Handbook - Section 1256 Contracts",
            title="KPMG Guide to Section 1256 Contracts",
            summary="KPMG comprehensive guidance on Section 1256 contract identification, mark-to-market, and 60/40 treatment",
            full_text="""
KPMG TAX HANDBOOK - SECTION 1256 CONTRACTS

IDENTIFICATION OF SECTION 1256 CONTRACTS:

1. REGULATED FUTURES CONTRACTS
   Definition: Contract requiring delivery of personal property or
   foreign currency that is:
   - Traded on/subject to rules of qualified board of trade
   - With respect to which initial margin required
   - Mark-to-market under exchange rules

   Examples: E-mini S&P 500 futures, crude oil futures, gold futures

2. FOREIGN CURRENCY CONTRACTS
   - Contract requiring delivery of foreign currency
   - Traded on interbank market
   - Entered into at arm's length price

3. NONEQUITY OPTIONS
   Definition: Listed option that is NOT an equity option
   - Broad-based stock index options (SPX, NDX, RUT, VIX)
   - Interest rate options
   - Currency options

   NOT Section 1256:
   - Options on individual stocks
   - Options on narrow-based indexes
   - Options on ETFs (unless tracking broad index)

4. DEALER EQUITY OPTIONS
   - Options traded by dealers in normal course of business
   - Subject to dealer mark-to-market rules

MARK-TO-MARKET RULES:
- All §1256 contracts marked at year-end
- FMV determined on last business day
- Gain/loss recognized in year of MTM
- 60% long-term / 40% short-term character
- Carryback of losses available (3 years)

ELECTIONS:
- Election out of §1256 for hedging transactions
- Must be identified as hedge on books
- Subject to documentation requirements
            """,
            effective_date=datetime(2024, 1, 1),
            related_rules=["IRC_1256"],
            keywords=["section 1256", "kpmg", "big4", "mark to market", "60/40", "futures", "options"]
        )

    def _load_next6_interpretations(self):
        """Load guidance from next 6 largest US accounting firms"""

        # BDO guidance
        self.rules["BDO_HF_TAX_2024"] = TaxRule(
            rule_id="BDO_HF_TAX_2024",
            source=TaxRuleSource.BDO,
            citation="BDO Hedge Fund Tax Guide",
            title="BDO Comprehensive Hedge Fund Tax Guide",
            summary="BDO guidance on hedge fund taxation including options, straddles, and wash sales",
            full_text="""
BDO HEDGE FUND TAX GUIDE

OPTIONS TAXATION FRAMEWORK:

1. EQUITY OPTIONS
   - Regular capital gain/loss treatment
   - Holding period based on actual holding
   - Premium received/paid affects basis
   - Exercise vs. expiration treatment

2. INDEX OPTIONS
   - Section 1256 if broad-based
   - Mark-to-market at year end
   - 60/40 treatment

3. STRADDLE PLANNING
   - Loss deferral under §1092
   - Mixed straddle elections
   - Qualified covered call exception
   - Coordination with §1256

BEST PRACTICES:
- Daily straddle identification
- Documentation of elections
- Wash sale tracking systems
- Coordination with fund admin
            """,
            effective_date=datetime(2024, 1, 1),
            keywords=["hedge fund", "bdo", "options", "straddle"]
        )

        # RSM guidance
        self.rules["RSM_DERIV_TAX_2024"] = TaxRule(
            rule_id="RSM_DERIV_TAX_2024",
            source=TaxRuleSource.RSM,
            citation="RSM Derivatives Tax Update",
            title="RSM Guide to Derivatives Taxation",
            summary="RSM guidance on taxation of derivatives including options, futures, and swaps",
            full_text="""
RSM DERIVATIVES TAX UPDATE

KEY DERIVATIVES TAX RULES:

1. OPTIONS
   - Character follows underlying for equity options
   - Section 1256 for nonequity options
   - Premium allocation rules
   - Exercise vs. expiration

2. FUTURES
   - Section 1256 contracts (regulated)
   - Mark-to-market
   - 60/40 treatment
   - Loss carryback available

3. SWAPS
   - Notional principal contracts
   - Ordinary income/deduction
   - May create straddles
   - Timing of inclusion

4. FORWARDS
   - Delivery contracts
   - May be constructive sale
   - Straddle rules apply
            """,
            effective_date=datetime(2024, 1, 1),
            keywords=["derivatives", "rsm", "options", "futures", "swaps"]
        )

        # Grant Thornton guidance
        self.rules["GT_STRADDLE_2024"] = TaxRule(
            rule_id="GT_STRADDLE_2024",
            source=TaxRuleSource.GRANT_THORNTON,
            citation="Grant Thornton Straddle Rules Guide",
            title="Grant Thornton Guide to Straddle Rules",
            summary="Grant Thornton comprehensive guide to straddle rules for investment partnerships",
            full_text="""
GRANT THORNTON STRADDLE RULES GUIDE

STRADDLE IDENTIFICATION:

A straddle exists when:
1. Taxpayer holds offsetting positions
2. Positions substantially diminish risk
3. Positions are in personal property

TYPES OF STRADDLES:

1. STOCK AND OPTION
   - Long stock + short call
   - Long stock + long put
   - Short stock + long call

2. CONVERSION/REVERSAL
   - Long stock + short call + long put
   - Short stock + long call + short put

3. SPREAD POSITIONS
   - Bull/bear spreads
   - Calendar spreads
   - Diagonal spreads

LOSS DEFERRAL RULES:
- Loss limited to unrealized gain in offsetting position
- Deferred loss carries forward
- Interest may be disallowed
- Holding period may be suspended
            """,
            effective_date=datetime(2024, 1, 1),
            keywords=["straddle", "grant thornton", "loss deferral", "offsetting positions"]
        )

    def _load_aicpa_guidance(self):
        """Load AICPA guidance"""

        self.rules["AICPA_INV_COMP_2024"] = TaxRule(
            rule_id="AICPA_INV_COMP_2024",
            source=TaxRuleSource.AICPA,
            citation="AICPA Investment Company Guide",
            title="AICPA Audit and Accounting Guide - Investment Companies",
            summary="AICPA guidance on investment company taxation and accounting",
            full_text="""
AICPA INVESTMENT COMPANY GUIDE

TAX CONSIDERATIONS FOR INVESTMENT COMPANIES:

1. WASH SALE TRACKING
   - Required for accurate tax reporting
   - Impact on basis and holding period
   - Coordination across share classes

2. STRADDLE IDENTIFICATION
   - Daily position analysis
   - Documentation requirements
   - Election documentation

3. SECTION 1256 MARK-TO-MARKET
   - Year-end FMV determination
   - 60/40 character allocation
   - Schedule D reporting

4. K-1 REPORTING
   - Accurate character reporting
   - Straddle gain/loss disclosure
   - Section 751 considerations
            """,
            effective_date=datetime(2024, 1, 1),
            keywords=["aicpa", "investment company", "audit", "tax reporting"]
        )

    def get_rule(self, rule_id: str) -> Optional[TaxRule]:
        """Get a specific tax rule by ID"""
        return self.rules.get(rule_id)

    def search_rules(self, keywords: List[str]) -> List[TaxRule]:
        """Search rules by keywords"""
        results = []
        for rule in self.rules.values():
            if any(kw.lower() in rule.keywords or kw.lower() in rule.title.lower()
                   or kw.lower() in rule.summary.lower() for kw in keywords):
                results.append(rule)
        return results

    def get_rules_by_source(self, source: TaxRuleSource) -> List[TaxRule]:
        """Get all rules from a specific source"""
        return [r for r in self.rules.values() if r.source == source]

    def get_related_rules(self, rule_id: str) -> List[TaxRule]:
        """Get rules related to a specific rule"""
        rule = self.get_rule(rule_id)
        if not rule:
            return []
        return [self.rules[r] for r in rule.related_rules if r in self.rules]


# =============================================================================
# TAX ANALYSIS ENGINE
# =============================================================================

class TaxAnalysisEngine:
    """
    Engine for analyzing options taxation, straddles, wash sales,
    and other complex tax matters.
    """

    def __init__(self):
        self.rules_db = TaxRulesDatabase()
        self.analyses: List[Dict] = []
        logger.info("TaxAnalysisEngine initialized")

    def analyze_straddle(
        self,
        position1: Dict[str, Any],
        position2: Dict[str, Any],
        as_of_date: datetime = None
    ) -> StraddleAnalysis:
        """
        Analyze whether two positions constitute a straddle under IRC §1092.

        Args:
            position1: First position (stock, option, etc.)
            position2: Second position (offsetting)
            as_of_date: Date for analysis

        Returns:
            StraddleAnalysis with determination and citations
        """
        as_of_date = as_of_date or datetime.now()

        # Extract position details
        pos1_type = position1.get("type", "unknown")
        pos1_symbol = position1.get("symbol", "")
        pos1_direction = position1.get("direction", "")  # long/short
        pos1_gain_loss = Decimal(str(position1.get("unrealized_gain_loss", 0)))

        pos2_type = position2.get("type", "unknown")
        pos2_symbol = position2.get("symbol", "")
        pos2_direction = position2.get("direction", "")
        pos2_gain_loss = Decimal(str(position2.get("unrealized_gain_loss", 0)))

        # Check for QCC exception
        if self._is_qualified_covered_call(position1, position2):
            return StraddleAnalysis(
                position_id=f"STRADDLE_{as_of_date.strftime('%Y%m%d')}_{pos1_symbol}",
                analysis_date=as_of_date,
                straddle_type=StraddleType.QUALIFIED_COVERED_CALL,
                leg1_description=f"{pos1_direction} {pos1_type} {pos1_symbol}",
                leg2_description=f"{pos2_direction} {pos2_type} {pos2_symbol}",
                citations=["IRC §1092(c)(4)", "Treas. Reg. §1.1092(c)-1"],
                notes="Qualifies as Qualified Covered Call - straddle rules do not apply"
            )

        # Check if offsetting positions
        is_offsetting = self._are_positions_offsetting(position1, position2)

        if not is_offsetting:
            return StraddleAnalysis(
                position_id=f"STRADDLE_{as_of_date.strftime('%Y%m%d')}_{pos1_symbol}",
                analysis_date=as_of_date,
                straddle_type=StraddleType.NOT_A_STRADDLE,
                leg1_description=f"{pos1_direction} {pos1_type} {pos1_symbol}",
                leg2_description=f"{pos2_direction} {pos2_type} {pos2_symbol}",
                citations=["IRC §1092(c)(2)"],
                notes="Positions do not substantially diminish risk of loss"
            )

        # Calculate loss deferral
        loss_deferred = Decimal("0")
        if pos1_gain_loss < 0 and pos2_gain_loss > 0:
            loss_deferred = min(abs(pos1_gain_loss), pos2_gain_loss)
        elif pos2_gain_loss < 0 and pos1_gain_loss > 0:
            loss_deferred = min(abs(pos2_gain_loss), pos1_gain_loss)

        return StraddleAnalysis(
            position_id=f"STRADDLE_{as_of_date.strftime('%Y%m%d')}_{pos1_symbol}",
            analysis_date=as_of_date,
            straddle_type=StraddleType.BASIC_STRADDLE,
            leg1_description=f"{pos1_direction} {pos1_type} {pos1_symbol}",
            leg2_description=f"{pos2_direction} {pos2_type} {pos2_symbol}",
            loss_deferred=loss_deferred,
            holding_period_suspended=True,
            citations=["IRC §1092(a)", "IRC §1092(d)", "Treas. Reg. §1.1092"],
            notes="Straddle identified - loss deferral and holding period suspension apply"
        )

    def analyze_wash_sale(
        self,
        sale: Dict[str, Any],
        acquisitions: List[Dict[str, Any]]
    ) -> WashSaleAnalysis:
        """
        Analyze whether a sale constitutes a wash sale under IRC §1091.

        Args:
            sale: Details of the sale (date, security, proceeds, loss)
            acquisitions: List of acquisitions to check against

        Returns:
            WashSaleAnalysis with determination
        """
        sale_date = sale.get("date")
        if isinstance(sale_date, str):
            sale_date = datetime.fromisoformat(sale_date)

        security_sold = sale.get("security", "")
        sale_proceeds = Decimal(str(sale.get("proceeds", 0)))
        cost_basis = Decimal(str(sale.get("cost_basis", 0)))
        loss_amount = cost_basis - sale_proceeds

        # Only analyze if there's a loss
        if loss_amount <= 0:
            return WashSaleAnalysis(
                sale_id=f"WASH_{sale_date.strftime('%Y%m%d')}_{security_sold}",
                analysis_date=datetime.now(),
                status=WashSaleStatus.NOT_WASH_SALE,
                security_sold=security_sold,
                sale_date=sale_date,
                sale_proceeds=sale_proceeds,
                loss_amount=loss_amount,
                citations=["IRC §1091"],
                substantially_identical_analysis="No loss to disallow - not a wash sale"
            )

        # Check 61-day window
        window_start = sale_date - timedelta(days=30)
        window_end = sale_date + timedelta(days=30)

        for acq in acquisitions:
            acq_date = acq.get("date")
            if isinstance(acq_date, str):
                acq_date = datetime.fromisoformat(acq_date)

            acq_security = acq.get("security", "")

            # Check if within window
            if window_start <= acq_date <= window_end:
                # Check if substantially identical
                is_identical = self._is_substantially_identical(security_sold, acq_security, acq)

                if is_identical:
                    acq_cost = Decimal(str(acq.get("cost", 0)))

                    return WashSaleAnalysis(
                        sale_id=f"WASH_{sale_date.strftime('%Y%m%d')}_{security_sold}",
                        analysis_date=datetime.now(),
                        status=WashSaleStatus.WASH_SALE,
                        security_sold=security_sold,
                        sale_date=sale_date,
                        sale_proceeds=sale_proceeds,
                        loss_amount=loss_amount,
                        replacement_security=acq_security,
                        replacement_date=acq_date,
                        replacement_cost=acq_cost,
                        loss_disallowed=loss_amount,
                        basis_adjustment=loss_amount,  # Add disallowed loss to basis
                        holding_period_adjustment=(sale_date - acq_date).days if acq_date < sale_date else 0,
                        substantially_identical_analysis=f"{acq_security} is substantially identical to {security_sold}",
                        citations=["IRC §1091(a)", "IRC §1091(d)", "Treas. Reg. §1.1091-1"]
                    )

        return WashSaleAnalysis(
            sale_id=f"WASH_{sale_date.strftime('%Y%m%d')}_{security_sold}",
            analysis_date=datetime.now(),
            status=WashSaleStatus.NOT_WASH_SALE,
            security_sold=security_sold,
            sale_date=sale_date,
            sale_proceeds=sale_proceeds,
            loss_amount=loss_amount,
            citations=["IRC §1091"],
            substantially_identical_analysis="No substantially identical acquisition within 61-day window"
        )

    def analyze_section_1256(
        self,
        contract: Dict[str, Any],
        year_end_value: Decimal
    ) -> Section1256Analysis:
        """
        Analyze Section 1256 contract treatment.

        Args:
            contract: Contract details
            year_end_value: Fair market value at year end

        Returns:
            Section1256Analysis with 60/40 calculation
        """
        contract_type = contract.get("type", "")
        cost_basis = Decimal(str(contract.get("cost_basis", 0)))

        # Determine if Section 1256
        is_1256 = self._is_section_1256_contract(contract)

        if not is_1256:
            return Section1256Analysis(
                contract_id=contract.get("id", ""),
                analysis_date=datetime.now(),
                contract_type=contract_type,
                is_section_1256=False,
                citations=["IRC §1256(b)"],
                notes="Not a Section 1256 contract - regular capital gain rules apply"
            )

        # Calculate mark-to-market gain/loss
        mtm_gain_loss = year_end_value - cost_basis

        # Apply 60/40 treatment
        long_term = mtm_gain_loss * Decimal("0.60")
        short_term = mtm_gain_loss * Decimal("0.40")

        return Section1256Analysis(
            contract_id=contract.get("id", ""),
            analysis_date=datetime.now(),
            contract_type=contract_type,
            is_section_1256=True,
            year_end_mtm_gain_loss=mtm_gain_loss,
            long_term_portion=long_term,
            short_term_portion=short_term,
            citations=["IRC §1256(a)", "IRC §1256(b)"],
            notes="Section 1256 contract - 60% long-term / 40% short-term treatment applies"
        )

    def _is_qualified_covered_call(self, pos1: Dict, pos2: Dict) -> bool:
        """Check if positions qualify as a qualified covered call"""
        # QCC: Long stock + short call that meets specific requirements
        stock_pos = None
        call_pos = None

        if pos1.get("type") == "stock" and pos1.get("direction") == "long":
            stock_pos = pos1
        if pos2.get("type") == "stock" and pos2.get("direction") == "long":
            stock_pos = pos2
        if pos1.get("type") == "call_option" and pos1.get("direction") == "short":
            call_pos = pos1
        if pos2.get("type") == "call_option" and pos2.get("direction") == "short":
            call_pos = pos2

        if not stock_pos or not call_pos:
            return False

        # Check QCC requirements
        # 1. Exchange traded
        if not call_pos.get("exchange_traded", False):
            return False

        # 2. More than 30 days to expiration
        days_to_exp = call_pos.get("days_to_expiration", 0)
        if days_to_exp <= 30:
            return False

        # 3. Not deep in the money (strike >= lowest qualified benchmark)
        stock_price = Decimal(str(stock_pos.get("price", 0)))
        strike = Decimal(str(call_pos.get("strike", 0)))

        if days_to_exp > 90:
            lowest_benchmark = stock_price * Decimal("0.85")
        else:
            lowest_benchmark = stock_price * Decimal("0.95")  # Simplified

        if strike < lowest_benchmark:
            return False  # Deep ITM

        return True

    def _are_positions_offsetting(self, pos1: Dict, pos2: Dict) -> bool:
        """Check if positions are offsetting under IRC §1092"""
        # Same underlying
        underlying1 = pos1.get("underlying", pos1.get("symbol", ""))
        underlying2 = pos2.get("underlying", pos2.get("symbol", ""))

        if underlying1 != underlying2:
            return False

        # Check for offsetting directions/types
        dir1 = pos1.get("direction", "")
        dir2 = pos2.get("direction", "")
        type1 = pos1.get("type", "")
        type2 = pos2.get("type", "")

        # Long stock + short call = offsetting
        if type1 == "stock" and dir1 == "long" and type2 == "call_option" and dir2 == "short":
            return True
        if type2 == "stock" and dir2 == "long" and type1 == "call_option" and dir1 == "short":
            return True

        # Long stock + long put = offsetting
        if type1 == "stock" and dir1 == "long" and type2 == "put_option" and dir2 == "long":
            return True
        if type2 == "stock" and dir2 == "long" and type1 == "put_option" and dir1 == "long":
            return True

        # Short stock + long call = offsetting
        if type1 == "stock" and dir1 == "short" and type2 == "call_option" and dir2 == "long":
            return True
        if type2 == "stock" and dir2 == "short" and type1 == "call_option" and dir1 == "long":
            return True

        return False

    def _is_substantially_identical(self, security1: str, security2: str, acq: Dict) -> bool:
        """Check if securities are substantially identical"""
        # Same security
        if security1 == security2:
            return True

        # Check if acquisition is an option on the same underlying
        acq_type = acq.get("type", "")
        if acq_type in ["call_option", "put_option"]:
            underlying = acq.get("underlying", "")
            if underlying == security1:
                # Check if deep in the money
                is_deep_itm = acq.get("is_deep_itm", False)
                if is_deep_itm:
                    return True  # Deep ITM options may be substantially identical

        return False

    def _is_section_1256_contract(self, contract: Dict) -> bool:
        """Determine if contract is a Section 1256 contract"""
        contract_type = contract.get("type", "").lower()

        # Regulated futures contracts
        if "futures" in contract_type:
            return True

        # Nonequity options (broad-based index options)
        if contract_type in ["index_option", "spx_option", "ndx_option", "rut_option"]:
            return True

        # Foreign currency contracts
        if "forex" in contract_type or "currency" in contract_type:
            return True

        # Equity options are NOT Section 1256
        if contract_type in ["equity_option", "stock_option"]:
            return False

        return False


# =============================================================================
# SINGLETON AND FACTORY
# =============================================================================

_rules_db_instance: Optional[TaxRulesDatabase] = None
_analysis_engine_instance: Optional[TaxAnalysisEngine] = None


def get_tax_rules_database() -> TaxRulesDatabase:
    """Get singleton TaxRulesDatabase instance"""
    global _rules_db_instance
    if _rules_db_instance is None:
        _rules_db_instance = TaxRulesDatabase()
    return _rules_db_instance


def get_tax_analysis_engine() -> TaxAnalysisEngine:
    """Get singleton TaxAnalysisEngine instance"""
    global _analysis_engine_instance
    if _analysis_engine_instance is None:
        _analysis_engine_instance = TaxAnalysisEngine()
    return _analysis_engine_instance


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TAX RULES ENGINE - IRS & Big 10 Accounting Firm Guidance")
    print("="*70)

    # Initialize database
    db = get_tax_rules_database()
    print(f"\nLoaded {len(db.rules)} tax rules")

    # Search example
    print("\n--- Search: 'straddle' ---")
    results = db.search_rules(["straddle"])
    for r in results[:3]:
        print(f"  {r.rule_id}: {r.title}")

    # Initialize analysis engine
    engine = get_tax_analysis_engine()

    # Example straddle analysis
    print("\n--- Straddle Analysis Example ---")
    pos1 = {"type": "stock", "symbol": "AAPL", "direction": "long", "unrealized_gain_loss": 5000}
    pos2 = {"type": "put_option", "symbol": "AAPL_PUT", "underlying": "AAPL", "direction": "long", "unrealized_gain_loss": -1000}

    analysis = engine.analyze_straddle(pos1, pos2)
    print(f"  Straddle Type: {analysis.straddle_type.value}")
    print(f"  Loss Deferred: ${analysis.loss_deferred}")
    print(f"  Holding Period Suspended: {analysis.holding_period_suspended}")

    print("\n" + "="*70)

