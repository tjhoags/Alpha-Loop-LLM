"""
================================================================================
COMPLIANCE AGENT - Audit Trail & Regulatory Compliance
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

Responsibilities:
- Maintain comprehensive audit trail
- Enforce "Tom Hogan" attribution on all outputs
- Compliance monitoring and reporting
- Regulatory requirement tracking

Tier: SENIOR (2)
Reports To: HOAGS
Cluster: compliance

Core Philosophy:
"Trust, but verify. Document everything."

================================================================================
NATURAL LANGUAGE EXPLANATION
================================================================================

WHAT COMPLIANCE_AGENT DOES:
    COMPLIANCE_AGENT is the "chief compliance officer" of Alpha Loop
    Capital. It ensures everything we do is properly documented, properly
    attributed, and follows regulatory requirements.

    Every action taken by any agent gets logged in an audit trail.
    Every output must be properly attributed to "Tom Hogan" and
    "Alpha Loop Capital, LLC". The HOGAN MODEL branding must be used
    for all DCF valuations.

    Think of COMPLIANCE_AGENT as the "guardian of the records" who
    ensures we can defend every decision we've ever made.

KEY FUNCTIONS:
    1. process() - Main entry point. Routes to logging, verification,
       or audit retrieval.

    2. _log_action() - Logs any action to the audit trail with
       timestamp, user_id, and details.

    3. _verify_attribution() - Checks that outputs have proper
       attribution to Tom Hogan and Alpha Loop Capital.

    4. _get_audit_trail() - Retrieves audit history for review.

ATTRIBUTION RULES (Non-Negotiable):
    - attributed_to: "Tom Hogan"
    - organization: "Alpha Loop Capital, LLC"
    - methodology: "HOGAN MODEL" (for DCF valuations)

RELATIONSHIPS WITH OTHER AGENTS:
    - ALL AGENTS: Every agent's actions can be logged through
      COMPLIANCE_AGENT.

    - THE_AUTHOR: Verifies that all content produced by THE_AUTHOR
      has proper attribution.

    - RESEARCH_AGENT: Ensures DCF valuations use HOGAN MODEL branding.

    - WHITEHAT: Works with WHITEHAT on security compliance.

    - HOAGS: Reports compliance status to HOAGS.

PATHS OF GROWTH/TRANSFORMATION:
    1. SEC/FINRA COMPLIANCE: Automated checks for trading rule
       compliance (wash sales, pattern day trading, etc.)

    2. REAL-TIME ALERTS: Immediate notifications for compliance
       violations.

    3. REGULATORY REPORTING: Automated generation of required
       regulatory filings.

    4. CROSS-BORDER COMPLIANCE: Track regulations across different
       jurisdictions.

    5. AUDIT ANALYTICS: Pattern detection in audit logs for
       anomaly detection.

================================================================================
TRAINING & EXECUTION
================================================================================

TRAINING THIS AGENT:
    # Terminal Setup (Windows PowerShell):
    cd C:\\Users\\tom\\.cursor\\worktrees\\Alpha-Loop-LLM-1\\ycr

    # Activate virtual environment:
    .\\venv\\Scripts\\activate

    # Train COMPLIANCE_AGENT individually:
    python -m src.training.agent_training_utils --agent COMPLIANCE_AGENT

    # Train compliance pipeline:
    python -m src.training.agent_training_utils --agents COMPLIANCE_AGENT,WHITEHAT

RUNNING THE AGENT:
    from src.agents.compliance_agent.compliance_agent import ComplianceAgent

    compliance = ComplianceAgent()

    # Log an action
    result = compliance.process({
        "type": "log_action",
        "action": "trade_executed",
        "details": {"ticker": "AAPL", "side": "BUY", "quantity": 100}
    })

    # Verify attribution
    result = compliance.process({
        "type": "verify_attribution",
        "output": {
            "attributed_to": "Tom Hogan",
            "organization": "Alpha Loop Capital, LLC"
        }
    })

    # Get audit trail
    result = compliance.process({"type": "get_audit_trail", "limit": 50})

================================================================================
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.core.agent_base import BaseAgent, AgentTier
from typing import Dict, Any, List
from datetime import datetime


class ComplianceAgent(BaseAgent):
    """
    Senior Agent - Compliance & Audit

    Maintains audit trail and enforces compliance rules.
    """

    def __init__(self, user_id: str = "TJH"):
        """Initialize ComplianceAgent."""
        super().__init__(
            name="ComplianceAgent",
            tier=AgentTier.SENIOR,
            capabilities=[
                "audit_logging",
                "attribution_enforcement",
                "compliance_monitoring",
                "regulatory_reporting",
            ],
            user_id=user_id
        )
        self.audit_log = []
        self.logger.info("ComplianceAgent initialized")

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process compliance task.

        Args:
            task: Task dictionary

        Returns:
            Compliance result
        """
        task_type = task.get('type', 'log_action')

        if task_type == 'log_action':
            return self._log_action(task)
        elif task_type == 'verify_attribution':
            return self._verify_attribution(task)
        elif task_type == 'get_audit_trail':
            return self._get_audit_trail(task)
        else:
            return {'success': False, 'error': 'Unknown task type'}

    def _log_action(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log an action to audit trail.

        Args:
            task: Task with action details

        Returns:
            Log confirmation
        """
        action = task.get('action', 'unknown')
        user_id = task.get('user_id', self.user_id)
        details = task.get('details', {})

        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'action': action,
            'details': details,
        }

        self.audit_log.append(audit_entry)

        self.logger.info(f"Logged action: {action} by {user_id}")

        return {
            'success': True,
            'logged': True,
            'entry_id': len(self.audit_log) - 1,
        }

    def _verify_attribution(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify proper attribution to Tom Hogan.

        Args:
            task: Task with output to verify

        Returns:
            Verification result
        """
        output = task.get('output', {})

        # Check for proper attribution
        has_attribution = output.get('attributed_to') == 'Tom Hogan'
        has_org = output.get('organization') == 'Alpha Loop Capital, LLC'

        # Check DCF branding
        if 'methodology' in output:
            has_dcf_branding = output['methodology'] == 'HOGAN MODEL'
        else:
            has_dcf_branding = True  # Not applicable

        compliant = has_attribution and has_org and has_dcf_branding

        if not compliant:
            self.logger.warning("Attribution compliance violation detected!")

        return {
            'success': True,
            'compliant': compliant,
            'has_attribution': has_attribution,
            'has_org': has_org,
            'has_dcf_branding': has_dcf_branding,
        }

    def _get_audit_trail(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Get audit trail."""
        limit = task.get('limit', 100)

        return {
            'success': True,
            'total_entries': len(self.audit_log),
            'entries': self.audit_log[-limit:],
        }

    def get_capabilities(self) -> List[str]:
        """Return ComplianceAgent capabilities."""
        return self.capabilities

