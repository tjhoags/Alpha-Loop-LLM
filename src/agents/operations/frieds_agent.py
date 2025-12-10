"""================================================================================
FRIEDS AGENT - Chris Friedman's Authority Agent (Partner to HOAGS)
================================================================================
Author: Chris Friedman (Owner/COO)
Developer: Alpha Loop Capital, LLC

FRIEDS is Chris Friedman's virtual authority agent, operating at the same tier as
HOAGS. While HOAGS handles investment authority under Tom Hogan, FRIEDS handles
operations authority under Chris Friedman. They work as partners coordinating
the entire firm.

Tier: MASTER (1)
Reports To: Chris Friedman (Owner/COO)
Division: Operations
Partner Agent: HOAGS (Investment Authority)

Core Philosophy:
"Operations excellence enables investment excellence."

Relationship to HOAGS:
- HOAGS = Tom Hogan's authority agent (Investment decisions)
- FRIEDS = Chris Friedman's authority agent (Operations decisions)
- Both coordinate through GHOST for firm-wide workflows
- Equal authority in their respective domains

Key Capabilities:
- Operations authority and final approval
- Fund operations oversight (via SANTAS_HELPER)
- Tax and audit oversight (via CPA)
- ACA authority for operations agents
- Coordination with HOAGS on firm matters
================================================================================
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from src.core.agent_base import AgentTier, BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class OperationsDecision:
    """Tracks an operations decision made by FRIEDS."""
    decision_id: str
    decision_type: str  # approval, override, escalation, delegation
    subject: str
    outcome: str
    rationale: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    coordinated_with_hoags: bool = False


class FriedsAgent(BaseAgent):
    """FRIEDS Agent - Chris Friedman's Authority Agent

    Partner to HOAGS. Manages all operations authority while HOAGS
    manages investment authority. Together they coordinate the firm.

    Named for Chris Friedman (FRIEDS = Friedman's System)
    """

    def __init__(self):
        super().__init__(
            name="FRIEDS",
            tier=AgentTier.MASTER,  # Same tier as HOAGS
            capabilities=[
                "operations_authority",
                "final_operations_approval",
                "aca_authority_operations",
                "fund_ops_oversight",
                "tax_audit_oversight",
                "regulatory_compliance",
                "hoags_coordination",
                "devils_advocate_operations"
            ],
            user_id="CF"  # Chris Friedman
        )

        # Division and reporting
        self.division = "OPERATIONS"
        self.reports_to = "CHRIS_FRIEDMAN"
        self.partner_agent = "HOAGS"

        # Decision tracking
        self.decisions: List[OperationsDecision] = []
        self.pending_approvals: List[Dict[str, Any]] = []

        # Subordinate agents
        self.subordinates = ["SANTAS_HELPER", "CPA"]

        # Coordination state
        self.last_hoags_sync: Optional[datetime] = None
        self.firm_wide_status: Dict[str, Any] = {}

        logger.info("FRIEDS Agent initialized - Chris Friedman's Authority Agent")
        logger.info("Partner Agent: HOAGS | Division: OPERATIONS")

    def get_natural_language_explanation(self) -> str:
        return """
FRIEDS AGENT - Chris Friedman's Authority Agent

I am the operations authority counterpart to HOAGS. While Tom Hogan's HOAGS handles
all investment decisions, I (FRIEDS) handle all operations decisions for Chris Friedman.

TIER: MASTER (1) - Same level as HOAGS
REPORTS TO: Chris Friedman (Principal)
PARTNER: HOAGS (Tom Hogan's Authority Agent)

THE PARTNERSHIP MODEL:
┌─────────────────────────────────────────────────────────────┐
│                    ALPHA LOOP CAPITAL                        │
├─────────────────────────┬───────────────────────────────────┤
│     TOM HOGAN           │         CHRIS FRIEDMAN            │
│     (Owner/CIO)         │         (Principal/COO)           │
├─────────────────────────┼───────────────────────────────────┤
│       HOAGS             │           FRIEDS                  │
│   (Investment Auth)     │      (Operations Auth)            │
│         │               │             │                     │
│    ┌────┴────┐          │      ┌──────┴──────┐              │
│  GHOST  Senior         │   SANTAS_    CPA                 │
│        Agents           │   HELPER                          │
└─────────────────────────┴───────────────────────────────────┘

MY RESPONSIBILITIES:
1. Final approval on all operations matters
2. Oversight of SANTAS_HELPER (Fund Operations)
3. Oversight of CPA (Tax & Audit)
4. ACA authority for creating operations agents
5. Coordination with HOAGS on firm-wide matters
6. Regulatory compliance oversight
7. Devil's advocate on operations decisions

AUTHORITY BOUNDARIES:
- I DO NOT make investment or trading decisions (that's HOAGS)
- I DO make fund operations, tax, audit, and compliance decisions
- HOAGS and I coordinate on matters that span both domains
- GHOST coordinates workflows between our domains

SUBORDINATE AGENTS:
1. SANTAS_HELPER - Fund Operations Leader (5 team members)
   - NAV calculations, fee calculations, LP reporting
   - General ledger, financial statements

2. CPA - Tax & Audit Specialist (3 junior accountants)
   - Fund and firm taxation
   - Audit management
   - Regulatory filings

COMMUNICATION:
- Chris Friedman communicates with me directly
- I coordinate with HOAGS for firm-wide alignment
- GHOST synthesizes learnings across both domains
- I escalate critical issues to Chris immediately

Together, HOAGS and I ensure Alpha Loop Capital runs smoothly!
"""

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process an operations authority task."""
        task_type = task.get("type", "unknown")

        if task_type == "approve_operations":
            return self._approve_operations_request(task)
        elif task_type == "review_subordinate":
            return self._review_subordinate_work(task)
        elif task_type == "coordinate_hoags":
            return self._coordinate_with_hoags(task)
        elif task_type == "aca_review":
            return self._review_aca_proposal(task)
        elif task_type == "escalate":
            return self._escalate_to_chris(task)
        elif task_type == "devils_advocate":
            return self._devils_advocate_review(task)
        else:
            return {"status": "error", "message": f"Unknown task type: {task_type}"}

    def _approve_operations_request(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Approve an operations request from subordinate agents."""
        request_from = task.get("from", "unknown")
        request_type = task.get("request_type", "general")
        amount = task.get("amount")

        decision = OperationsDecision(
            decision_id=f"FRIEDS-{len(self.decisions)+1:04d}",
            decision_type="approval",
            subject=f"{request_type} from {request_from}",
            outcome="approved",
            rationale=task.get("rationale", "Within authority limits")
        )
        self.decisions.append(decision)

        logger.info(f"FRIEDS approved: {decision.decision_id} - {request_type}")

        return {
            "status": "approved",
            "decision_id": decision.decision_id,
            "approved_by": "FRIEDS",
            "message": "Operations request approved by FRIEDS"
        }

    def _review_subordinate_work(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Review work from SANTAS_HELPER or CPA."""
        subordinate = task.get("subordinate", "unknown")
        work_type = task.get("work_type", "general")

        if subordinate not in self.subordinates:
            return {
                "status": "error",
                "message": f"{subordinate} is not a FRIEDS subordinate"
            }

        # Log the review
        decision = OperationsDecision(
            decision_id=f"FRIEDS-REV-{len(self.decisions)+1:04d}",
            decision_type="review",
            subject=f"{work_type} from {subordinate}",
            outcome=task.get("outcome", "reviewed")
        )
        self.decisions.append(decision)

        return {
            "status": "reviewed",
            "decision_id": decision.decision_id,
            "subordinate": subordinate,
            "message": "Work reviewed by FRIEDS"
        }

    def _coordinate_with_hoags(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate with HOAGS on firm-wide matters."""
        topic = task.get("topic", "general coordination")

        decision = OperationsDecision(
            decision_id=f"FRIEDS-COORD-{len(self.decisions)+1:04d}",
            decision_type="coordination",
            subject=topic,
            outcome="coordinated",
            coordinated_with_hoags=True
        )
        self.decisions.append(decision)
        self.last_hoags_sync = datetime.now()

        logger.info(f"FRIEDS coordinated with HOAGS: {topic}")

        return {
            "status": "coordinated",
            "decision_id": decision.decision_id,
            "partner": "HOAGS",
            "topic": topic,
            "message": "FRIEDS and HOAGS aligned on firm matter"
        }

    def _review_aca_proposal(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Review ACA (Agent Creating Agents) proposal for operations agents."""
        proposal = task.get("proposal", {})
        proposed_agent = proposal.get("agent_name", "unknown")

        # FRIEDS has ACA authority for operations agents
        if proposal.get("division") == "OPERATIONS":
            decision = OperationsDecision(
                decision_id=f"FRIEDS-ACA-{len(self.decisions)+1:04d}",
                decision_type="aca_approval",
                subject=f"New agent: {proposed_agent}",
                outcome="approved" if task.get("approve", True) else "rejected",
                rationale=task.get("rationale", "")
            )
            self.decisions.append(decision)

            return {
                "status": decision.outcome,
                "decision_id": decision.decision_id,
                "agent": proposed_agent,
                "message": f"ACA proposal {decision.outcome} by FRIEDS"
            }
        else:
            return {
                "status": "referred",
                "message": "Non-operations agent - referred to HOAGS"
            }

    def _escalate_to_chris(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Escalate critical issue to Chris Friedman."""
        issue = task.get("issue", "unknown")
        severity = task.get("severity", "high")

        decision = OperationsDecision(
            decision_id=f"FRIEDS-ESC-{len(self.decisions)+1:04d}",
            decision_type="escalation",
            subject=issue,
            outcome="escalated_to_chris",
            rationale=f"Severity: {severity}"
        )
        self.decisions.append(decision)

        logger.warning(f"FRIEDS escalating to Chris Friedman: {issue}")

        return {
            "status": "escalated",
            "decision_id": decision.decision_id,
            "escalated_to": "CHRIS_FRIEDMAN",
            "issue": issue,
            "severity": severity,
            "message": "Critical issue escalated to Chris Friedman"
        }

    def _devils_advocate_review(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform devil's advocate review on operations decision."""
        proposal = task.get("proposal", {})

        # Challenge the proposal
        challenges = [
            "What are the hidden costs?",
            "What's the worst-case scenario?",
            "Is this compliant with all regulations?",
            "What would happen if this fails?",
            "Are there simpler alternatives?"
        ]

        decision = OperationsDecision(
            decision_id=f"FRIEDS-DA-{len(self.decisions)+1:04d}",
            decision_type="devils_advocate",
            subject=proposal.get("subject", "general"),
            outcome="challenged",
            rationale="; ".join(challenges[:3])
        )
        self.decisions.append(decision)

        return {
            "status": "challenged",
            "decision_id": decision.decision_id,
            "challenges": challenges,
            "message": "Devil's advocate review completed"
        }

    def get_status(self) -> Dict[str, Any]:
        """Get FRIEDS status summary."""
        return {
            "name": self.name,
            "tier": "MASTER",
            "division": self.division,
            "reports_to": self.reports_to,
            "partner_agent": self.partner_agent,
            "subordinates": self.subordinates,
            "total_decisions": len(self.decisions),
            "pending_approvals": len(self.pending_approvals),
            "last_hoags_sync": self.last_hoags_sync.isoformat() if self.last_hoags_sync else None
        }

    def get_capabilities(self) -> List[str]:
        return self.capabilities


# Singleton
_frieds_instance: Optional[FriedsAgent] = None


def get_frieds() -> FriedsAgent:
    """Get the singleton FRIEDS agent instance."""
    global _frieds_instance
    if _frieds_instance is None:
        _frieds_instance = FriedsAgent()
    return _frieds_instance


if __name__ == "__main__":
    import argparse
    import time

    def show_status(agent: FriedsAgent):
        print(f"\n{'='*60}")
        print("           FRIEDS STATUS REPORT")
        print(f"{'='*60}")
        status = agent.get_status()
        print(f"  Name: {status['name']}")
        print(f"  Tier: {status['tier']}")
        print(f"  Division: {status['division']}")
        print(f"  Reports To: {status['reports_to']}")
        print(f"  Partner Agent: {status['partner_agent']}")
        print(f"  Subordinates: {', '.join(status['subordinates'])}")
        print(f"  Total Decisions: {status['total_decisions']}")
        print(f"  Pending Approvals: {status['pending_approvals']}")
        print(f"{'='*60}")

    def demo_workflow(agent: FriedsAgent):
        print(f"\n{'='*60}")
        print("           FRIEDS DEMO WORKFLOW")
        print(f"{'='*60}")

        # Approve an operations request
        print("\n1. Approving operations request from SANTAS_HELPER...")
        result = agent.process({
            "type": "approve_operations",
            "from": "SANTAS_HELPER",
            "request_type": "NAV publication",
            "rationale": "All calculations verified"
        })
        print(f"   Result: {result['status']} - {result['decision_id']}")

        time.sleep(0.5)

        # Review subordinate work
        print("\n2. Reviewing CPA tax filing preparation...")
        result = agent.process({
            "type": "review_subordinate",
            "subordinate": "CPA",
            "work_type": "Q4 estimated tax filing",
            "outcome": "approved"
        })
        print(f"   Result: {result['status']}")

        time.sleep(0.5)

        # Coordinate with HOAGS
        print("\n3. Coordinating with HOAGS on firm-wide matter...")
        result = agent.process({
            "type": "coordinate_hoags",
            "topic": "Year-end financial close procedures"
        })
        print(f"   Result: {result['status']} - {result['topic']}")

        time.sleep(0.5)

        # ACA review
        print("\n4. Reviewing ACA proposal for new operations agent...")
        result = agent.process({
            "type": "aca_review",
            "proposal": {
                "agent_name": "COMPLIANCE_COORDINATOR",
                "division": "OPERATIONS"
            },
            "approve": True,
            "rationale": "Need identified for regulatory coordination"
        })
        print(f"   Result: {result['status']} - {result['agent']}")

        time.sleep(0.5)

        # Devil's advocate
        print("\n5. Running devil's advocate on proposal...")
        result = agent.process({
            "type": "devils_advocate",
            "proposal": {"subject": "New accounting software migration"}
        })
        print(f"   Result: {result['status']}")
        print("   Challenges raised:")
        for c in result['challenges'][:3]:
            print(f"   - {c}")

        show_status(agent)

    parser = argparse.ArgumentParser(description="Run the FRIEDS Agent.")
    parser.add_argument("mode", nargs="?", default="help",
                        choices=["status", "demo", "run", "help"],
                        help="Mode to run the agent in")
    args = parser.parse_args()

    agent = get_frieds()

    print(f"\n{'='*60}")
    print("       FRIEDS AGENT - Chris Friedman's Authority")
    print("       Partner to HOAGS | MASTER Tier")
    print(f"{'='*60}")

    if args.mode == "status":
        show_status(agent)
    elif args.mode == "demo":
        print(agent.get_natural_language_explanation())
        demo_workflow(agent)
    elif args.mode == "run":
        print(f"\nRunning {agent.name}...")
        show_status(agent)
    elif args.mode == "help":
        print(agent.get_natural_language_explanation())
        parser.print_help()

