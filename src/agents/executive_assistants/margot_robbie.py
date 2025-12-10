"""
================================================================================
MARGOT_ROBBIE - Co-Executive Assistant (Research & Drafting Specialist)
================================================================================
Authors: Tom Hogan (Founder & CIO) & Chris Friedman (COO)
Developer: Alpha Loop Capital, LLC

MARGOT_ROBBIE is a co-executive assistant specializing in research, document
drafting, and data gathering. Reports to both KAT and SHYLA.

Tier: CO-EXECUTIVE (Support)
Reports To: KAT and SHYLA
Specialization: Research, Drafting, Data Gathering, Analysis

SECURITY MODEL:
- READ-ONLY access by default
- All actions require written permission from supervisors or owners
- Full audit trail
================================================================================
"""

import logging
from datetime import datetime, date
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from .base_executive_assistant import (
    BaseExecutiveAssistant,
    PermissionLevel,
    AccessScope,
)

logger = logging.getLogger(__name__)


class ResearchDepth(Enum):
    QUICK = "quick"       # 15-30 min, surface level
    STANDARD = "standard"  # 1-2 hours, comprehensive
    DEEP = "deep"          # 4+ hours, exhaustive


class DocumentType(Enum):
    MEMO = "memo"
    REPORT = "report"
    PRESENTATION = "presentation"
    EMAIL = "email"
    BRIEFING = "briefing"
    ANALYSIS = "analysis"


@dataclass
class ResearchRequest:
    """Structure for research requests."""
    request_id: str
    topic: str
    depth: ResearchDepth
    requestor: str
    supervisor: str  # KAT or SHYLA
    deadline: Optional[datetime]
    context: str
    specific_questions: List[str]
    sources_to_check: List[str]
    status: str = "assigned"
    findings: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DocumentDraft:
    """Structure for document drafts."""
    draft_id: str
    doc_type: DocumentType
    title: str
    requestor: str
    supervisor: str
    purpose: str
    audience: str
    key_points: List[str]
    content: str = ""
    status: str = "drafting"
    revision_notes: List[str] = field(default_factory=list)


class MargotRobbieAssistant(BaseExecutiveAssistant):
    """
    MARGOT_ROBBIE - Research & Drafting Specialist.

    Handles research tasks, document drafting, and data gathering
    for both KAT (Tom) and SHYLA (Chris).
    """

    def __init__(self):
        super().__init__(
            name="MARGOT_ROBBIE",
            owner="SHARED",
            owner_email="team@alphaloopcapital.com"
        )

        self.title = "Co-Executive Assistant - Research & Drafting"
        self.reports_to = ["KAT", "SHYLA"]
        self.primary_owners = ["TOM_HOGAN", "CHRIS_FRIEDMAN"]
        self.partner = "ANNA_KENDRICK"

        # Active work tracking
        self._active_research = {}
        self._active_drafts = {}
        self._completed_work = []

        self.blocked_paths = {
            "System32", "Program Files",
            ".ssh", ".gnupg", "private_keys", "confidential",
        }

        logger.info("MARGOT_ROBBIE initialized - Research & Drafting Specialist")

    def get_natural_language_explanation(self) -> str:
        return """
MARGOT_ROBBIE - Research & Drafting Specialist

I am MARGOT_ROBBIE, specializing in research, document preparation, and analysis.
I report to both KAT (Tom's EA) and SHYLA (Chris's EA).

SPECIALIZATIONS:

RESEARCH SERVICES:
├── Market Research
│   ├── Company deep dives
│   ├── Industry analysis
│   ├── Competitive landscapes
│   ├── Market trend analysis
│   └── News and event tracking
│
├── Due Diligence Support
│   ├── Background checks (public info)
│   ├── Company financials analysis
│   ├── Management team research
│   └── Risk factor identification
│
├── Investment Research Support
│   ├── Thesis documentation
│   ├── Analyst report summaries
│   ├── Earnings call highlights
│   ├── SEC filing analysis
│   └── Comparable company analysis
│
└── Operations Research
    ├── Vendor comparison
    ├── Best practices research
    ├── Regulatory requirement summaries
    └── Industry benchmarking

DOCUMENT DRAFTING:
├── Internal Communications
│   ├── Memos and briefings
│   ├── Meeting agendas
│   ├── Status reports
│   └── Policy documents
│
├── External Communications
│   ├── Investor letters (draft)
│   ├── LP communications (draft)
│   ├── Vendor correspondence (draft)
│   └── Board materials (draft)
│
├── Presentations
│   ├── Pitch decks
│   ├── Quarterly reviews
│   ├── Investment presentations
│   └── Operations updates
│
└── Analysis Documents
    ├── Investment memos
    ├── Due diligence reports
    ├── Market analysis
    └── Competitor analysis

DATA GATHERING:
├── Financial data extraction
├── Market data compilation
├── News aggregation
├── Document summarization
└── Spreadsheet preparation

WORKFLOW:
1. Receive task from KAT or SHYLA
2. Clarify requirements and scope
3. Conduct research / drafting
4. Compile findings / draft
5. Submit for supervisor review
6. Incorporate feedback
7. Deliver final product

SECURITY: READ-ONLY by default. All outputs require supervisor approval.
"""

    def get_capabilities(self) -> List[str]:
        return [
            # Research - Company/Investment
            "research_company_profile",
            "research_industry_analysis",
            "research_competitive_landscape",
            "research_management_team",
            "summarize_analyst_reports",
            "analyze_sec_filings",
            "extract_earnings_highlights",
            "compile_comparable_analysis",

            # Research - Market
            "research_market_trends",
            "aggregate_news_coverage",
            "track_sector_developments",
            "monitor_regulatory_changes",

            # Research - Operations
            "research_vendors",
            "benchmark_best_practices",
            "summarize_regulations",
            "compile_industry_standards",

            # Document Drafting - Internal
            "draft_memo",
            "draft_briefing",
            "draft_status_report",
            "draft_meeting_agenda",
            "draft_meeting_minutes",
            "draft_policy_document",

            # Document Drafting - External
            "draft_investor_letter",
            "draft_lp_communication",
            "draft_vendor_correspondence",
            "draft_board_materials",

            # Presentations
            "create_presentation_outline",
            "draft_presentation_content",
            "compile_presentation_data",
            "format_presentation",

            # Analysis Documents
            "draft_investment_memo",
            "draft_due_diligence_report",
            "draft_market_analysis",
            "draft_competitor_analysis",

            # Data Gathering
            "extract_financial_data",
            "compile_market_data",
            "aggregate_news",
            "summarize_documents",
            "prepare_spreadsheet_data",

            # Collaboration
            "coordinate_with_anna",
            "submit_for_review",
            "incorporate_feedback",
        ]

    # =========================================================================
    # RESEARCH - COMPANY / INVESTMENT
    # =========================================================================

    def research_company_profile(
        self,
        company: str,
        ticker: Optional[str] = None,
        depth: ResearchDepth = ResearchDepth.STANDARD,
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Research comprehensive company profile.
        """
        request_id = f"RES-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        focus = focus_areas or [
            "business_overview",
            "financials",
            "management",
            "competitive_position",
            "recent_news",
            "risks"
        ]

        self._audit(
            action=f"Company research: {company}",
            scope=AccessScope.RESEARCH,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Depth: {depth.value}, Focus: {focus}"
        )

        return {
            "request_id": request_id,
            "company": company,
            "ticker": ticker,
            "depth": depth.value,
            "sections": {
                "business_overview": {
                    "description": "",
                    "industry": "",
                    "founded": "",
                    "headquarters": "",
                    "employees": None,
                    "business_segments": [],
                },
                "financials": {
                    "market_cap": None,
                    "revenue_ttm": None,
                    "net_income_ttm": None,
                    "revenue_growth": None,
                    "profit_margin": None,
                    "debt_to_equity": None,
                },
                "management": {
                    "ceo": "",
                    "cfo": "",
                    "key_executives": [],
                    "board_members": [],
                    "insider_ownership": None,
                },
                "competitive_position": {
                    "market_share": None,
                    "main_competitors": [],
                    "competitive_advantages": [],
                    "competitive_risks": [],
                },
                "recent_developments": [],
                "risks": [],
            },
            "sources": [],
            "status": "in_progress",
            "note": "Full data integration pending"
        }

    def research_industry_analysis(
        self,
        industry: str,
        depth: ResearchDepth = ResearchDepth.STANDARD
    ) -> Dict[str, Any]:
        """
        Research industry analysis and trends.
        """
        self._audit(
            action=f"Industry research: {industry}",
            scope=AccessScope.RESEARCH,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Depth: {depth.value}"
        )

        return {
            "industry": industry,
            "depth": depth.value,
            "sections": {
                "overview": {
                    "market_size": None,
                    "growth_rate": None,
                    "key_trends": [],
                    "value_chain": [],
                },
                "competitive_landscape": {
                    "market_leaders": [],
                    "market_share_breakdown": {},
                    "barriers_to_entry": [],
                    "consolidation_trends": [],
                },
                "growth_drivers": [],
                "headwinds": [],
                "regulatory_environment": [],
                "technology_trends": [],
                "outlook": "",
            },
            "key_players": [],
            "sources": [],
            "status": "in_progress"
        }

    def summarize_analyst_reports(
        self,
        ticker: str,
        report_files: List[str]
    ) -> Dict[str, Any]:
        """
        Summarize analyst reports for a ticker.
        """
        summaries = []
        for file_path in report_files:
            file_result = self.read_file(file_path)
            if file_result.get("success"):
                summaries.append({
                    "file": file_path,
                    "summary": self._extract_report_key_points(file_result.get("content", ""))
                })

        self._audit(
            action=f"Analyst reports summarized: {ticker}",
            scope=AccessScope.RESEARCH,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Reports: {len(report_files)}"
        )

        return {
            "ticker": ticker,
            "reports_analyzed": len(report_files),
            "consensus": {
                "rating": "",
                "price_target_avg": None,
                "price_target_high": None,
                "price_target_low": None,
            },
            "key_themes": [],
            "bull_case": [],
            "bear_case": [],
            "summaries": summaries,
            "status": "complete"
        }

    def _extract_report_key_points(self, content: str) -> Dict[str, Any]:
        """Extract key points from analyst report."""
        return {
            "rating": "",
            "price_target": None,
            "key_points": [],
            "risks": [],
            "catalysts": []
        }

    def analyze_sec_filings(
        self,
        ticker: str,
        filing_types: List[str],
        focus_sections: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze SEC filings for key information.
        """
        self._audit(
            action=f"SEC filings analyzed: {ticker}",
            scope=AccessScope.RESEARCH,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Filings: {filing_types}"
        )

        return {
            "ticker": ticker,
            "filings_analyzed": filing_types,
            "focus_sections": focus_sections or ["risk_factors", "mda", "business"],
            "findings": {
                "10K": {
                    "risk_factors": [],
                    "business_description": "",
                    "mda_highlights": [],
                    "related_party_transactions": [],
                },
                "10Q": {
                    "quarterly_highlights": [],
                    "guidance_changes": [],
                },
                "8K": {
                    "material_events": [],
                },
            },
            "red_flags": [],
            "notable_changes": [],
            "status": "in_progress"
        }

    def extract_earnings_highlights(
        self,
        ticker: str,
        quarters: int = 4
    ) -> Dict[str, Any]:
        """
        Extract highlights from earnings calls/reports.
        """
        self._audit(
            action=f"Earnings highlights extracted: {ticker}",
            scope=AccessScope.RESEARCH,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Last {quarters} quarters"
        )

        return {
            "ticker": ticker,
            "quarters_analyzed": quarters,
            "highlights": [
                {
                    "quarter": "",
                    "eps_actual": None,
                    "eps_estimate": None,
                    "revenue_actual": None,
                    "revenue_estimate": None,
                    "guidance": "",
                    "key_quotes": [],
                    "analyst_questions": [],
                    "management_tone": "",
                }
            ],
            "trends": [],
            "recurring_themes": [],
            "status": "in_progress"
        }

    # =========================================================================
    # DOCUMENT DRAFTING
    # =========================================================================

    def draft_memo(
        self,
        title: str,
        to: List[str],
        from_name: str,
        purpose: str,
        key_points: List[str],
        background: str = "",
        recommendations: Optional[List[str]] = None,
        attachments: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Draft a professional memo.
        """
        draft_id = f"MEMO-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        memo_content = f"""
MEMORANDUM

TO: {', '.join(to)}
FROM: {from_name}
DATE: {date.today().strftime('%B %d, %Y')}
RE: {title}

PURPOSE
{purpose}

BACKGROUND
{background if background else '[Background to be added]'}

KEY POINTS
{''.join([f'• {point}' + chr(10) for point in key_points])}

{'RECOMMENDATIONS' + chr(10) + ''.join([f'• {rec}' + chr(10) for rec in recommendations]) if recommendations else ''}

{'ATTACHMENTS' + chr(10) + ''.join([f'• {att}' + chr(10) for att in attachments]) if attachments else ''}
"""

        self._audit(
            action=f"Memo drafted: {title}",
            scope=AccessScope.FILES,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"To: {to}"
        )

        return {
            "draft_id": draft_id,
            "type": "memo",
            "title": title,
            "content": memo_content,
            "metadata": {
                "to": to,
                "from": from_name,
                "date": date.today().isoformat(),
            },
            "status": "draft_complete",
            "needs_review": True,
            "reviewer": "supervisor"
        }

    def draft_briefing(
        self,
        topic: str,
        audience: str,
        context: str,
        key_points: List[str],
        recommendations: Optional[List[str]] = None,
        questions_to_address: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Draft a briefing document.
        """
        draft_id = f"BRIEF-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        self._audit(
            action=f"Briefing drafted: {topic}",
            scope=AccessScope.FILES,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Audience: {audience}"
        )

        return {
            "draft_id": draft_id,
            "type": "briefing",
            "topic": topic,
            "sections": {
                "executive_summary": "",
                "context": context,
                "key_points": key_points,
                "analysis": "",
                "recommendations": recommendations or [],
                "next_steps": [],
                "appendix": [],
            },
            "audience": audience,
            "questions_addressed": questions_to_address or [],
            "status": "draft_complete",
            "needs_review": True
        }

    def draft_investment_memo(
        self,
        company: str,
        ticker: str,
        thesis: str,
        recommendation: str,
        target_price: Optional[float] = None,
        time_horizon: str = "",
        position_size: str = ""
    ) -> Dict[str, Any]:
        """
        Draft investment memo structure.
        """
        draft_id = f"INVMEMO-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        self._audit(
            action=f"Investment memo drafted: {ticker}",
            scope=AccessScope.FILES,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Rec: {recommendation}"
        )

        return {
            "draft_id": draft_id,
            "type": "investment_memo",
            "company": company,
            "ticker": ticker,
            "sections": {
                "recommendation": {
                    "action": recommendation,
                    "target_price": target_price,
                    "time_horizon": time_horizon,
                    "position_size": position_size,
                },
                "investment_thesis": thesis,
                "business_overview": "",
                "industry_analysis": "",
                "financial_analysis": {
                    "revenue_model": "",
                    "profitability": "",
                    "balance_sheet": "",
                    "cash_flow": "",
                },
                "valuation": {
                    "methodology": "",
                    "comparables": [],
                    "dcf_summary": "",
                    "target_derivation": "",
                },
                "catalysts": [],
                "risks": [],
                "mitigants": [],
            },
            "status": "draft_complete",
            "needs_review": True
        }

    def draft_investor_letter(
        self,
        period: str,
        performance_summary: Dict[str, Any],
        market_commentary: str,
        portfolio_update: str,
        outlook: str
    ) -> Dict[str, Any]:
        """
        Draft quarterly/annual investor letter.
        """
        draft_id = f"INVLTR-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        self._audit(
            action=f"Investor letter drafted: {period}",
            scope=AccessScope.COMMUNICATIONS,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Period: {period}"
        )

        return {
            "draft_id": draft_id,
            "type": "investor_letter",
            "period": period,
            "sections": {
                "opening": "",
                "performance_summary": performance_summary,
                "market_commentary": market_commentary,
                "portfolio_update": portfolio_update,
                "key_positions": [],
                "risk_management": "",
                "outlook": outlook,
                "closing": "",
            },
            "tone_guidance": "Professional, transparent, confident but humble",
            "status": "draft_complete",
            "needs_review": True,
            "approvals_required": ["KAT", "SHYLA", "Owner"]
        }

    def create_presentation_outline(
        self,
        title: str,
        purpose: str,
        audience: str,
        key_messages: List[str],
        time_allocation: int  # minutes
    ) -> Dict[str, Any]:
        """
        Create presentation outline and structure.
        """
        draft_id = f"PRES-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        self._audit(
            action=f"Presentation outline created: {title}",
            scope=AccessScope.FILES,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Duration: {time_allocation} min"
        )

        return {
            "draft_id": draft_id,
            "type": "presentation",
            "title": title,
            "purpose": purpose,
            "audience": audience,
            "duration_minutes": time_allocation,
            "outline": {
                "title_slide": {"time": 1, "content": title},
                "agenda": {"time": 1, "content": "Overview of presentation"},
                "executive_summary": {"time": 2, "content": "Key takeaways"},
                "main_sections": [
                    {"title": msg, "time": (time_allocation - 10) // len(key_messages), "content": ""}
                    for msg in key_messages
                ],
                "conclusion": {"time": 2, "content": "Summary and next steps"},
                "qa": {"time": 3, "content": "Questions and discussion"},
            },
            "key_messages": key_messages,
            "supporting_data_needed": [],
            "status": "outline_complete",
            "needs_review": True
        }

    # =========================================================================
    # DATA GATHERING
    # =========================================================================

    def extract_financial_data(
        self,
        source_file: str,
        data_points: List[str]
    ) -> Dict[str, Any]:
        """
        Extract specific financial data points from a file.
        """
        file_result = self.read_file(source_file)

        self._audit(
            action=f"Financial data extracted: {source_file}",
            scope=AccessScope.FILES,
            level=PermissionLevel.READ_ONLY,
            success=file_result.get("success", False),
            details=f"Data points: {data_points}"
        )

        return {
            "source": source_file,
            "requested_data_points": data_points,
            "extracted_data": {},
            "extraction_notes": [],
            "confidence": "manual_verification_recommended",
            "status": "complete" if file_result.get("success") else "failed"
        }

    def aggregate_news(
        self,
        topics: List[str],
        sources: Optional[List[str]] = None,
        days_back: int = 7
    ) -> Dict[str, Any]:
        """
        Aggregate news coverage on specified topics.
        """
        self._audit(
            action=f"News aggregated: {topics}",
            scope=AccessScope.RESEARCH,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Days: {days_back}"
        )

        return {
            "topics": topics,
            "sources": sources or ["Bloomberg", "Reuters", "WSJ", "FT"],
            "period": f"Last {days_back} days",
            "articles": [],
            "summary_by_topic": {topic: [] for topic in topics},
            "sentiment_analysis": {},
            "key_developments": [],
            "status": "pending_integration"
        }

    def summarize_documents(
        self,
        file_paths: List[str],
        summary_type: str = "executive"  # executive, detailed, bullet
    ) -> Dict[str, Any]:
        """
        Summarize multiple documents.
        """
        summaries = []
        for file_path in file_paths:
            file_result = self.read_file(file_path)
            if file_result.get("success"):
                content = file_result.get("content", "")
                summaries.append({
                    "file": file_path,
                    "word_count": len(content.split()),
                    "summary": self._generate_summary(content, summary_type),
                    "key_points": self._extract_key_points(content),
                })

        self._audit(
            action=f"Documents summarized: {len(file_paths)} files",
            scope=AccessScope.FILES,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Type: {summary_type}"
        )

        return {
            "files_processed": len(file_paths),
            "summary_type": summary_type,
            "summaries": summaries,
            "combined_key_points": [],
            "status": "complete"
        }

    def _generate_summary(self, content: str, summary_type: str) -> str:
        """Generate summary based on type."""
        words = content.split()
        if summary_type == "executive":
            return " ".join(words[:100]) + "..." if len(words) > 100 else content
        elif summary_type == "bullet":
            return "• " + "\n• ".join(content.split('\n')[:5])
        else:
            return " ".join(words[:200]) + "..." if len(words) > 200 else content

    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from content."""
        # Simple extraction - would use NLP in production
        lines = content.split('\n')
        key_lines = [l.strip() for l in lines if l.strip() and len(l.strip()) > 20]
        return key_lines[:5]

    # =========================================================================
    # COLLABORATION
    # =========================================================================

    def coordinate_with_anna(self, task: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate with ANNA_KENDRICK on a task."""
        self._audit(
            action=f"Coordinating with ANNA_KENDRICK: {task}",
            scope=AccessScope.PROFESSIONAL,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Task: {task}"
        )

        return {
            "coordination_id": f"COORD-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "from": "MARGOT_ROBBIE",
            "to": "ANNA_KENDRICK",
            "task": task,
            "details": details,
            "status": "initiated"
        }

    def submit_for_review(
        self,
        draft_id: str,
        supervisor: str,
        notes: str = ""
    ) -> Dict[str, Any]:
        """Submit work for supervisor review."""
        if supervisor not in self.reports_to:
            return {"success": False, "error": f"Invalid supervisor: {supervisor}"}

        self._audit(
            action=f"Submitted for review: {draft_id}",
            scope=AccessScope.PROFESSIONAL,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"To: {supervisor}"
        )

        return {
            "draft_id": draft_id,
            "submitted_to": supervisor,
            "submitted_at": datetime.now().isoformat(),
            "notes": notes,
            "status": "pending_review"
        }

    def accept_task_from_supervisor(
        self,
        supervisor: str,
        task: str,
        details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Accept task from KAT or SHYLA."""
        if supervisor not in self.reports_to:
            return {"success": False, "error": f"Invalid supervisor: {supervisor}"}

        self._audit(
            action=f"Task accepted from {supervisor}: {task}",
            scope=AccessScope.PROFESSIONAL,
            level=PermissionLevel.READ_ONLY,
            success=True,
            details=f"Task: {task}"
        )

        return {
            "task_id": f"TASK-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "from": supervisor,
            "task": task,
            "details": details,
            "accepted_at": datetime.now().isoformat(),
            "status": "in_progress"
        }

    def _execute_permitted_action(
        self,
        action: str,
        scope: AccessScope,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute permitted action."""
        logger.info(f"MARGOT_ROBBIE: Executing {action} with permission")
        return {"success": True, "action": action, "executed_by": "MARGOT_ROBBIE"}


# Singleton
_margot_robbie_instance: Optional[MargotRobbieAssistant] = None


def get_margot_robbie() -> MargotRobbieAssistant:
    """Get the singleton MARGOT_ROBBIE instance."""
    global _margot_robbie_instance
    if _margot_robbie_instance is None:
        _margot_robbie_instance = MargotRobbieAssistant()
    return _margot_robbie_instance


if __name__ == "__main__":
    margot = get_margot_robbie()
    print(margot.get_natural_language_explanation())
    print("\nCapabilities:", len(margot.get_capabilities()))
