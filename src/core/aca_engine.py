"""================================================================================
ACA ENGINE - Agent Creating Agents
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

The ACA (Agent Creating Agents) Engine manages the dynamic creation and
coordination of agents within the ALC-Algo ecosystem.

Key Features:
- Dynamic agent creation based on detected capability gaps
- Proposal system for new agents
- Approval workflow (HoagsAgent authority)
- Agent registry and capability tracking
- Ecosystem expansion management

Only HoagsAgent has authority to approve new agent creation if senior level or above.
================================================================================
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class ProposalStatus(Enum):
    """Status of an agent proposal."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"


@dataclass
class AgentProposalRecord:
    """Record of a proposed new agent."""

    proposal_id: str
    proposed_by: str
    proposed_at: datetime
    agent_name: str
    agent_tier: str
    capabilities: List[str]
    gap_description: str
    rationale: str
    priority: str
    status: ProposalStatus = ProposalStatus.PENDING
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejected_by: Optional[str] = None
    rejected_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "proposal_id": self.proposal_id,
            "proposed_by": self.proposed_by,
            "proposed_at": self.proposed_at.isoformat(),
            "agent_name": self.agent_name,
            "agent_tier": self.agent_tier,
            "capabilities": self.capabilities,
            "gap_description": self.gap_description,
            "rationale": self.rationale,
            "priority": self.priority,
            "status": self.status.value,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
        }


@dataclass
class CapabilityGapRecord:
    """Record of a detected capability gap."""

    gap_id: str
    detected_by: str
    detected_at: datetime
    task_type: str
    required_capabilities: List[str]
    missing_capabilities: List[str]
    severity: str
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_by: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "gap_id": self.gap_id,
            "detected_by": self.detected_by,
            "detected_at": self.detected_at.isoformat(),
            "task_type": self.task_type,
            "required_capabilities": self.required_capabilities,
            "missing_capabilities": self.missing_capabilities,
            "severity": self.severity,
            "resolved": self.resolved,
        }


class ACAEngine:
    """Agent Creating Agents Engine

    Manages the dynamic expansion of the agent ecosystem through:
    1. Capability gap detection
    2. Agent proposals
    3. Approval workflow
    4. Agent instantiation

    AUTHORITY: Only HoagsAgent can approve new agents.
    """

    def __init__(self):
        """Initialize the ACA Engine."""
        # Proposal storage
        self.proposals: Dict[str, AgentProposalRecord] = {}

        # Gap storage
        self.capability_gaps: Dict[str, CapabilityGapRecord] = {}

        # Agent registry (name -> instance)
        self.created_agents: Dict[str, Any] = {}

        # Capability registry (capability -> [agent_names])
        self.capability_registry: Dict[str, List[str]] = {}

        # Statistics
        self.total_proposals = 0
        self.approved_count = 0
        self.rejected_count = 0
        self.agents_created = 0

        logger.info("ACA Engine initialized")

    # =========================================================================
    # PROPOSAL MANAGEMENT
    # =========================================================================

    def submit_proposal(
        self,
        proposal_id: str,
        proposed_by: str,
        agent_name: str,
        agent_tier: str,
        capabilities: List[str],
        gap_description: str = "",
        rationale: str = "",
        priority: str = "medium",
    ) -> AgentProposalRecord:
        """Submit a proposal for a new agent.

        Args:
        ----
            proposal_id: Unique identifier
            proposed_by: Name of proposing agent
            agent_name: Name for the new agent
            agent_tier: Tier level
            capabilities: Required capabilities
            gap_description: Description of the gap being filled
            rationale: Why this agent is needed
            priority: Priority level (low, medium, high, critical)

        Returns:
        -------
            AgentProposalRecord
        """
        proposal = AgentProposalRecord(
            proposal_id=proposal_id,
            proposed_by=proposed_by,
            proposed_at=datetime.now(),
            agent_name=agent_name,
            agent_tier=agent_tier,
            capabilities=capabilities,
            gap_description=gap_description,
            rationale=rationale,
            priority=priority,
        )

        self.proposals[proposal_id] = proposal
        self.total_proposals += 1

        logger.info(f"ACA: New proposal submitted - {agent_name} by {proposed_by}")

        return proposal

    def get_pending_proposals(self) -> List[Dict]:
        """Get all pending proposals."""
        return [
            p.to_dict() for p in self.proposals.values()
            if p.status == ProposalStatus.PENDING
        ]

    def get_proposal(self, proposal_id: str) -> Optional[AgentProposalRecord]:
        """Get a specific proposal."""
        return self.proposals.get(proposal_id)

    def approve_proposal(self, proposal_id: str, approver: str) -> bool:
        """Approve a proposal. Only HoagsAgent should call this.

        Args:
        ----
            proposal_id: ID of proposal to approve
            approver: Name of approving agent (should be HoagsAgent)

        Returns:
        -------
            True if approved successfully
        """
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            logger.error(f"ACA: Proposal not found - {proposal_id}")
            return False

        if proposal.status != ProposalStatus.PENDING:
            logger.warning(f"ACA: Proposal not pending - {proposal_id}")
            return False

        proposal.status = ProposalStatus.APPROVED
        proposal.approved_by = approver
        proposal.approved_at = datetime.now()

        self.approved_count += 1

        logger.info(f"ACA: Proposal approved - {proposal.agent_name} by {approver}")

        return True

    def reject_proposal(
        self,
        proposal_id: str,
        rejector: str,
        reason: str = "",
    ) -> bool:
        """Reject a proposal.

        Args:
        ----
            proposal_id: ID of proposal to reject
            rejector: Name of rejecting agent
            reason: Reason for rejection

        Returns:
        -------
            True if rejected successfully
        """
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            return False

        if proposal.status != ProposalStatus.PENDING:
            return False

        proposal.status = ProposalStatus.REJECTED
        proposal.rejected_by = rejector
        proposal.rejected_at = datetime.now()
        proposal.rejection_reason = reason

        self.rejected_count += 1

        logger.info(f"ACA: Proposal rejected - {proposal.agent_name}: {reason}")

        return True

    # =========================================================================
    # GAP MANAGEMENT
    # =========================================================================

    def register_gap(
        self,
        gap_id: str,
        detected_by: str,
        task_type: str,
        required_capabilities: List[str],
        missing_capabilities: List[str],
        severity: str = "medium",
        context: Dict[str, Any] = None,
    ) -> CapabilityGapRecord:
        """Register a detected capability gap.

        Args:
        ----
            gap_id: Unique identifier
            detected_by: Agent that detected the gap
            task_type: Type of task that exposed the gap
            required_capabilities: Capabilities needed
            missing_capabilities: Capabilities not available
            severity: Gap severity
            context: Additional context

        Returns:
        -------
            CapabilityGapRecord
        """
        gap = CapabilityGapRecord(
            gap_id=gap_id,
            detected_by=detected_by,
            detected_at=datetime.now(),
            task_type=task_type,
            required_capabilities=required_capabilities,
            missing_capabilities=missing_capabilities,
            severity=severity,
            context=context or {},
        )

        self.capability_gaps[gap_id] = gap

        logger.info(f"ACA: Capability gap registered - {missing_capabilities}")

        return gap

    def get_active_gaps(self) -> List[Dict]:
        """Get all unresolved capability gaps."""
        return [
            g.to_dict() for g in self.capability_gaps.values()
            if not g.resolved
        ]

    def resolve_gap(self, gap_id: str, resolved_by: str) -> bool:
        """Mark a gap as resolved."""
        gap = self.capability_gaps.get(gap_id)
        if gap:
            gap.resolved = True
            gap.resolved_by = resolved_by
            return True
        return False

    # =========================================================================
    # AGENT CREATION
    # =========================================================================

    def create_agent_from_proposal(
        self,
        proposal_id: str,
        agent_class: Optional[Type] = None,
    ) -> Optional[Any]:
        """Create an agent from an approved proposal.

        Args:
        ----
            proposal_id: ID of approved proposal
            agent_class: Optional specific class to use

        Returns:
        -------
            Created agent instance or None
        """
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            logger.error(f"ACA: Proposal not found - {proposal_id}")
            return None

        if proposal.status != ProposalStatus.APPROVED:
            logger.error(f"ACA: Proposal not approved - {proposal_id}")
            return None

        # Import base agent for dynamic creation

        # Create a dynamic agent class if no class provided
        if agent_class is None:
            # For now, return None - actual implementation would create dynamic class
            logger.warning("ACA: Dynamic agent creation not yet implemented")
            return None

        # Instantiate the agent
        try:
            agent = agent_class(user_id="TJH")

            # Register the agent
            self.created_agents[proposal.agent_name] = agent

            # Update capability registry
            for cap in proposal.capabilities:
                if cap not in self.capability_registry:
                    self.capability_registry[cap] = []
                self.capability_registry[cap].append(proposal.agent_name)

            # Mark proposal as implemented
            proposal.status = ProposalStatus.IMPLEMENTED
            self.agents_created += 1

            logger.info(f"ACA: Agent created - {proposal.agent_name}")

            return agent

        except Exception as e:
            logger.error(f"ACA: Failed to create agent - {e}")
            return None

    # =========================================================================
    # CAPABILITY QUERIES
    # =========================================================================

    def get_agents_with_capability(self, capability: str) -> List[str]:
        """Get all agents with a specific capability."""
        return self.capability_registry.get(capability, [])

    def has_capability(self, capability: str) -> bool:
        """Check if any agent has a capability."""
        return capability in self.capability_registry and len(self.capability_registry[capability]) > 0

    def get_all_capabilities(self) -> List[str]:
        """Get all registered capabilities."""
        return list(self.capability_registry.keys())

    def register_agent_capabilities(self, agent_name: str, capabilities: List[str]):
        """Register capabilities for an existing agent."""
        for cap in capabilities:
            if cap not in self.capability_registry:
                self.capability_registry[cap] = []
            if agent_name not in self.capability_registry[cap]:
                self.capability_registry[cap].append(agent_name)

    # =========================================================================
    # CONTINUOUS IMPROVEMENT FEEDBACK LOOP
    # =========================================================================

    def trigger_improvement_loop(
        self,
        trigger_type: str,
        context: Dict[str, Any],
        auto_approve: bool = False,
    ) -> Optional[str]:
        """Trigger the continuous improvement feedback loop.

        TRAINING NEVER STOPS. This system automatically spawns agents
        to improve performance, speed, and capability.

        Args:
            trigger_type: Type of trigger
                - "performance_gap": AUC/Accuracy below target
                - "speed_bottleneck": Processing too slow
                - "success_pattern": Clone successful approach
                - "model_degradation": Replace degrading model
                - "new_pattern": Novel alpha detected
            context: Details about the trigger
            auto_approve: Auto-approve for critical issues

        Returns:
            proposal_id if triggered, None otherwise
        """
        import hashlib

        proposal_id = hashlib.sha256(
            f"{trigger_type}{datetime.now()}".encode()
        ).hexdigest()[:12]

        # Determine agent spec based on trigger
        specs = self._get_improvement_specs(trigger_type, context)

        if not specs:
            logger.warning(f"ACA: No specs for trigger type: {trigger_type}")
            return None

        # Submit proposal
        proposal = self.submit_proposal(
            proposal_id=proposal_id,
            proposed_by="ACA_FEEDBACK_LOOP",
            agent_name=specs["agent_name"],
            agent_tier=specs["tier"],
            capabilities=specs["capabilities"],
            gap_description=f"Continuous improvement trigger: {trigger_type}",
            rationale=specs["rationale"],
            priority=specs["priority"],
        )

        logger.info(
            f"ACA FEEDBACK LOOP: Triggered {trigger_type} -> {specs['agent_name']}"
        )

        # Auto-approve for critical issues
        if auto_approve or specs["priority"] == "critical":
            self.approve_proposal(proposal_id, "ACA_AUTO_APPROVAL")
            logger.info(f"ACA FEEDBACK LOOP: Auto-approved {proposal_id}")

        return proposal_id

    def _get_improvement_specs(
        self, trigger_type: str, context: Dict
    ) -> Optional[Dict]:
        """Get specs for improvement agent based on trigger type."""

        if trigger_type == "performance_gap":
            current_auc = context.get("current_auc", 0.52)
            target_auc = context.get("target_auc", 0.58)
            symbol = context.get("symbol", "UNKNOWN")

            return {
                "agent_name": f"OPTIMIZER_{symbol}_{datetime.now().strftime('%H%M%S')}",
                "tier": "STANDARD",
                "capabilities": [
                    "hyperparameter_optimization",
                    "feature_engineering_improvement",
                    "model_architecture_search",
                ],
                "rationale": f"AUC {current_auc:.3f} below target {target_auc:.3f}. "
                            f"Spawning optimizer to close gap.",
                "priority": "high" if target_auc - current_auc > 0.05 else "medium",
            }

        elif trigger_type == "speed_bottleneck":
            bottleneck = context.get("bottleneck", "processing")
            current_time = context.get("current_time_ms", 0)
            target_time = context.get("target_time_ms", 100)

            return {
                "agent_name": f"ACCELERATOR_{bottleneck.upper()}_{datetime.now().strftime('%H%M%S')}",
                "tier": "SUPPORT",
                "capabilities": [
                    "parallel_processing",
                    "batch_optimization",
                    "cache_management",
                    "async_execution",
                ],
                "rationale": f"Speed bottleneck: {bottleneck}. "
                            f"Current {current_time}ms, target {target_time}ms. "
                            f"Spawning accelerator for {current_time/target_time:.1f}x speedup.",
                "priority": "critical" if current_time > target_time * 5 else "high",
            }

        elif trigger_type == "success_pattern":
            source_agent = context.get("source_agent", "UNKNOWN")
            pattern = context.get("pattern", "unspecified")

            return {
                "agent_name": f"CLONE_{source_agent}_{datetime.now().strftime('%H%M%S')}",
                "tier": context.get("source_tier", "STANDARD"),
                "capabilities": context.get("capabilities", []) + ["pattern_exploration"],
                "rationale": f"Success pattern detected in {source_agent}: {pattern}. "
                            f"Cloning and mutating to explore adjacent strategies.",
                "priority": "medium",
            }

        elif trigger_type == "model_degradation":
            degrading_model = context.get("model", "UNKNOWN")
            degradation_pct = context.get("degradation_pct", 0)

            return {
                "agent_name": f"REPLACEMENT_{degrading_model}_{datetime.now().strftime('%H%M%S')}",
                "tier": context.get("model_tier", "STANDARD"),
                "capabilities": context.get("capabilities", []),
                "rationale": f"Model {degrading_model} degraded {degradation_pct:.1f}%. "
                            f"Spawning replacement for seamless transition.",
                "priority": "critical",
            }

        elif trigger_type == "new_pattern":
            pattern_desc = context.get("pattern", "Novel pattern")
            confidence = context.get("confidence", 0.5)

            return {
                "agent_name": f"SPECIALIST_{datetime.now().strftime('%H%M%S')}",
                "tier": "SPECIALIST",
                "capabilities": [
                    "pattern_exploitation",
                    "alpha_capture",
                    "real_time_adaptation",
                ],
                "rationale": f"New pattern detected (confidence {confidence:.1%}): {pattern_desc}. "
                            f"Spawning specialist to capture emerging alpha.",
                "priority": "high" if confidence > 0.7 else "medium",
            }

        return None

    def check_performance_and_spawn(
        self,
        metrics: Dict[str, float],
        symbol: str = "GENERAL",
    ) -> List[str]:
        """Check performance metrics and spawn improvement agents as needed.

        TRAINING NEVER STOPS. This method ensures continuous improvement.

        Args:
            metrics: Dict with 'auc', 'accuracy', 'sharpe', etc.
            symbol: Symbol being evaluated

        Returns:
            List of spawned proposal IDs
        """
        spawned = []

        # Phase targets (beat Wall Street)
        phase_targets = {
            "phase_1": {"auc": 0.52, "accuracy": 0.52},  # Baseline
            "phase_2": {"auc": 0.55, "accuracy": 0.55},  # Wall Street parity
            "phase_3": {"auc": 0.58, "accuracy": 0.58},  # Beat Wall Street
            "phase_4": {"auc": 0.62, "accuracy": 0.62},  # Alpha Loop Standard
        }

        current_auc = metrics.get("auc", 0.5)
        current_acc = metrics.get("accuracy", 0.5)

        # Determine current phase and target
        if current_auc < phase_targets["phase_1"]["auc"]:
            target = phase_targets["phase_1"]
            phase = "BASELINE"
        elif current_auc < phase_targets["phase_2"]["auc"]:
            target = phase_targets["phase_2"]
            phase = "WALL_STREET_PARITY"
        elif current_auc < phase_targets["phase_3"]["auc"]:
            target = phase_targets["phase_3"]
            phase = "BEAT_WALL_STREET"
        elif current_auc < phase_targets["phase_4"]["auc"]:
            target = phase_targets["phase_4"]
            phase = "ALPHA_LOOP_STANDARD"
        else:
            # Already at Alpha Loop Standard - maintain excellence
            logger.info(f"ACA: {symbol} at Alpha Loop Standard (AUC={current_auc:.3f})")
            return spawned

        # Below target - spawn improvement agent
        gap = target["auc"] - current_auc
        logger.info(
            f"ACA FEEDBACK: {symbol} in phase {phase}, "
            f"AUC gap: {gap:.3f} (current: {current_auc:.3f}, target: {target['auc']:.2f})"
        )

        proposal_id = self.trigger_improvement_loop(
            trigger_type="performance_gap",
            context={
                "symbol": symbol,
                "current_auc": current_auc,
                "target_auc": target["auc"],
                "current_accuracy": current_acc,
                "target_accuracy": target["accuracy"],
                "phase": phase,
            },
            auto_approve=gap > 0.03,  # Auto-approve for large gaps
        )

        if proposal_id:
            spawned.append(proposal_id)

        return spawned

    # =========================================================================
    # STATUS & STATS
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive ACA status."""
        return {
            "total_proposals": self.total_proposals,
            "pending_proposals": len([p for p in self.proposals.values() if p.status == ProposalStatus.PENDING]),
            "approved_proposals": self.approved_count,
            "rejected_proposals": self.rejected_count,
            "agents_created": self.agents_created,
            "active_gaps": len([g for g in self.capability_gaps.values() if not g.resolved]),
            "total_capabilities": len(self.capability_registry),
            "feedback_loop": "ACTIVE - Training never stops",
            "phase_targets": {
                "phase_1_baseline": "AUC > 0.52",
                "phase_2_wall_street": "AUC > 0.55",
                "phase_3_beat_ws": "AUC > 0.58",
                "phase_4_alc_standard": "AUC > 0.62",
            },
            "timestamp": datetime.now().isoformat(),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Alias for get_status."""
        return self.get_status()


# =============================================================================
# SINGLETON
# =============================================================================

_aca_engine_instance: Optional[ACAEngine] = None


def get_aca_engine() -> ACAEngine:
    """Get the singleton ACA Engine instance."""
    global _aca_engine_instance
    if _aca_engine_instance is None:
        _aca_engine_instance = ACAEngine()
    return _aca_engine_instance

