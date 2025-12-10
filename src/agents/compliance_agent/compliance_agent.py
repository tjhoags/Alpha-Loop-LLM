"""
ComplianceAgent - Audit trail and compliance enforcement
Author: Tom Hogan | Alpha Loop Capital, LLC

Responsibilities:
- Maintain audit trail
- Enforce "Tom Hogan" attribution
- Compliance monitoring
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.core.agent_base import BaseAgent, AgentTier
from typing import Dict, Any, List
from datetime import datetime
import json


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

