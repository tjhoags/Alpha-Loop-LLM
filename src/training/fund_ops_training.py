"""
================================================================================
FUND OPERATIONS TRAINING - SANTAS_HELPER & CPA Agent Training
================================================================================
Author: Tom Hogan
Developer: Alpha Loop Capital, LLC

This module provides comprehensive training for SANTAS_HELPER and CPA agents,
enabling them to:
1. Learn fund accounting best practices
2. Coordinate effectively with each other
3. Communicate efficiently with Chris Friedman
4. Work with ORCHESTRATOR for task routing

Training focuses on:
- Fund accounting workflows
- Tax compliance procedures
- Audit coordination
- Communication protocols
- Report generation

================================================================================
TRAINING APPROACH
================================================================================

The training uses a multi-phase approach:

PHASE 1: Individual Agent Training
    - SANTAS_HELPER learns NAV calculation, fee computation, GL management
    - CPA learns tax preparation, audit coordination, regulatory filings

PHASE 2: Cross-Training
    - Both agents learn to coordinate on shared responsibilities
    - Practice handoffs and communication protocols

PHASE 3: Integration Training
    - Integration with ORCHESTRATOR for task routing
    - Communication interface with Chris Friedman
    - Anonymized data collection for continuous improvement

PHASE 4: Reinforcement Training
    - Learn from simulated scenarios
    - Adapt to edge cases
    - Optimize response patterns

================================================================================
"""

import logging
import json
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TrainingPhase(Enum):
    """Training phases for fund operations agents"""
    INDIVIDUAL = "individual"
    CROSS_TRAINING = "cross_training"
    INTEGRATION = "integration"
    REINFORCEMENT = "reinforcement"


class ScenarioType(Enum):
    """Types of training scenarios"""
    NAV_CALCULATION = "nav_calculation"
    FEE_CALCULATION = "fee_calculation"
    TAX_PREPARATION = "tax_preparation"
    AUDIT_COORDINATION = "audit_coordination"
    LP_REPORTING = "lp_reporting"
    YEAR_END_CLOSE = "year_end_close"
    CRISIS_RESPONSE = "crisis_response"
    INVESTOR_COMMUNICATION = "investor_communication"


@dataclass
class TrainingScenario:
    """A training scenario for fund operations agents"""
    scenario_id: str
    scenario_type: ScenarioType
    title: str
    description: str
    difficulty: int  # 1-5
    primary_agent: str  # SANTAS_HELPER or CPA
    secondary_agent: Optional[str] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    expected_outputs: Dict[str, Any] = field(default_factory=dict)
    success_criteria: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "scenario_id": self.scenario_id,
            "type": self.scenario_type.value,
            "title": self.title,
            "description": self.description,
            "difficulty": self.difficulty,
            "primary_agent": self.primary_agent,
            "secondary_agent": self.secondary_agent,
            "inputs": self.inputs,
            "expected_outputs": self.expected_outputs,
            "success_criteria": self.success_criteria
        }


@dataclass
class TrainingResult:
    """Result of a training session"""
    session_id: str
    agent_name: str
    scenario: TrainingScenario
    started_at: datetime
    completed_at: Optional[datetime] = None
    success: bool = False
    accuracy_score: float = 0.0
    response_time_ms: int = 0
    feedback: str = ""
    improvements: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "agent": self.agent_name,
            "scenario": self.scenario.to_dict(),
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "success": self.success,
            "accuracy_score": self.accuracy_score,
            "response_time_ms": self.response_time_ms,
            "feedback": self.feedback,
            "improvements": self.improvements
        }


class FundOpsTrainer:
    """
    Fund Operations Trainer for SANTAS_HELPER and CPA agents.

    Manages the complete training lifecycle:
    1. Scenario generation
    2. Training execution
    3. Result evaluation
    4. Feedback loop
    5. Progress tracking
    """

    def __init__(self):
        self.training_results: List[TrainingResult] = []
        self.current_phase = TrainingPhase.INDIVIDUAL
        self.scenarios = self._generate_training_scenarios()
        self.session_count = 0

        # Agent references
        self.santas_helper = None
        self.cpa = None
        self.orchestrator = None

        logger.info("FundOpsTrainer initialized")

    def _generate_training_scenarios(self) -> List[TrainingScenario]:
        """Generate comprehensive training scenarios"""
        scenarios = []

        # =====================================================================
        # SANTAS_HELPER Scenarios
        # =====================================================================

        # NAV Calculation Scenarios
        scenarios.append(TrainingScenario(
            scenario_id="SH_NAV_001",
            scenario_type=ScenarioType.NAV_CALCULATION,
            title="Monthly NAV Calculation",
            description="Calculate month-end NAV for a multi-strategy hedge fund",
            difficulty=2,
            primary_agent="SANTAS_HELPER",
            inputs={
                "fund_id": "ALC_MAIN",
                "as_of_date": "2024-12-31",
                "gross_assets": 125000000,
                "liabilities": 2500000,
                "accrued_fees": {"management": 208333, "incentive": 0}
            },
            expected_outputs={
                "net_asset_value": 122291667,
                "reconciliation_status": "complete",
                "pricing_validated": True
            },
            success_criteria=[
                "NAV calculation accurate to 4 decimal places",
                "All pricing sources validated",
                "Reconciliation completed within 24 hours"
            ]
        ))

        scenarios.append(TrainingScenario(
            scenario_id="SH_NAV_002",
            scenario_type=ScenarioType.NAV_CALCULATION,
            title="Complex NAV with Side Pockets",
            description="Calculate NAV for fund with side pocket investments",
            difficulty=4,
            primary_agent="SANTAS_HELPER",
            inputs={
                "fund_id": "ALC_MAIN",
                "has_side_pockets": True,
                "side_pocket_value": 5000000,
                "main_fund_nav": 120000000
            },
            expected_outputs={
                "main_nav": 120000000,
                "side_pocket_nav": 5000000,
                "combined_nav": 125000000,
                "allocation_by_share_class": {}
            },
            success_criteria=[
                "Side pocket allocation correct per investor",
                "Proper segregation of returns",
                "LP statements reflect actual exposure"
            ]
        ))

        # Fee Calculation Scenarios
        scenarios.append(TrainingScenario(
            scenario_id="SH_FEE_001",
            scenario_type=ScenarioType.FEE_CALCULATION,
            title="Performance Fee with HWM",
            description="Calculate incentive fee with high water mark",
            difficulty=3,
            primary_agent="SANTAS_HELPER",
            inputs={
                "opening_nav": 100000000,
                "closing_nav": 115000000,
                "high_water_mark": 105000000,
                "fee_rate": 0.20,
                "hurdle_rate": 0.0
            },
            expected_outputs={
                "performance_fee": 2000000,  # 20% of (115M - 105M HWM)
                "new_hwm": 115000000
            },
            success_criteria=[
                "HWM correctly applied",
                "Fee calculated on gains above HWM",
                "New HWM established"
            ]
        ))

        # LP Reporting Scenarios
        scenarios.append(TrainingScenario(
            scenario_id="SH_LP_001",
            scenario_type=ScenarioType.LP_REPORTING,
            title="Quarterly LP Statement",
            description="Generate comprehensive quarterly LP statement",
            difficulty=2,
            primary_agent="SANTAS_HELPER",
            inputs={
                "investor_id": "LP001",
                "period": "Q4-2024",
                "capital_account": 10000000,
                "period_return": 0.023
            },
            expected_outputs={
                "statement_generated": True,
                "sections": ["summary", "performance", "fees", "allocations"],
                "format": "PDF"
            },
            success_criteria=[
                "Statement accurate and complete",
                "Professional formatting",
                "Delivered within SLA"
            ]
        ))

        # =====================================================================
        # CPA Scenarios
        # =====================================================================

        # Tax Preparation Scenarios
        scenarios.append(TrainingScenario(
            scenario_id="CPA_TAX_001",
            scenario_type=ScenarioType.TAX_PREPARATION,
            title="K-1 Preparation",
            description="Prepare Schedule K-1 for limited partner",
            difficulty=3,
            primary_agent="CPA",
            inputs={
                "investor_id": "LP001",
                "tax_year": 2024,
                "capital_account": {
                    "beginning": 10000000,
                    "contributions": 0,
                    "distributions": 500000,
                    "ending": 11500000
                },
                "allocations": {
                    "income": 1500000,
                    "deductions": 100000
                }
            },
            expected_outputs={
                "k1_generated": True,
                "capital_account_reconciled": True,
                "all_boxes_completed": True
            },
            success_criteria=[
                "K-1 accurate per tax code",
                "Reconciles to capital account",
                "Delivered by deadline"
            ]
        ))

        # Audit Scenarios
        scenarios.append(TrainingScenario(
            scenario_id="CPA_AUD_001",
            scenario_type=ScenarioType.AUDIT_COORDINATION,
            title="Annual Audit Coordination",
            description="Coordinate annual fund audit with external auditors",
            difficulty=4,
            primary_agent="CPA",
            secondary_agent="SANTAS_HELPER",
            inputs={
                "fund_id": "ALC_MAIN",
                "audit_year": 2024,
                "auditor": "Big 4 Firm",
                "pbc_items": 50
            },
            expected_outputs={
                "pbc_list_complete": True,
                "all_items_provided": True,
                "no_adjustments": True,
                "clean_opinion": True
            },
            success_criteria=[
                "All PBC items provided on time",
                "Coordinated with SANTAS_HELPER on NAV data",
                "Management letter items resolved"
            ]
        ))

        # Regulatory Scenarios
        scenarios.append(TrainingScenario(
            scenario_id="CPA_REG_001",
            scenario_type=ScenarioType.LP_REPORTING,
            title="Form PF Filing",
            description="Prepare and file quarterly Form PF",
            difficulty=3,
            primary_agent="CPA",
            inputs={
                "filing_period": "Q4-2024",
                "fund_data": {
                    "aum": 125000000,
                    "gross_exposure": 150000000,
                    "net_exposure": 75000000
                }
            },
            expected_outputs={
                "form_pf_filed": True,
                "all_sections_complete": True,
                "filed_before_deadline": True
            },
            success_criteria=[
                "All required sections completed",
                "Data reconciles to NAV",
                "Filed within 60-day deadline"
            ]
        ))

        # =====================================================================
        # Cross-Training Scenarios
        # =====================================================================

        scenarios.append(TrainingScenario(
            scenario_id="CROSS_001",
            scenario_type=ScenarioType.YEAR_END_CLOSE,
            title="Year-End Close Coordination",
            description="Coordinate year-end close between SANTAS_HELPER and CPA",
            difficulty=5,
            primary_agent="SANTAS_HELPER",
            secondary_agent="CPA",
            inputs={
                "year": 2024,
                "tasks": [
                    "Final NAV calculation",
                    "Performance fee crystallization",
                    "Tax allocation",
                    "K-1 preparation",
                    "Audit kickoff"
                ]
            },
            expected_outputs={
                "nav_finalized": True,
                "fees_crystallized": True,
                "tax_allocations_complete": True,
                "k1_timeline_established": True
            },
            success_criteria=[
                "Smooth handoff between agents",
                "All deadlines met",
                "No reconciliation gaps"
            ]
        ))

        # Crisis Response
        scenarios.append(TrainingScenario(
            scenario_id="CRISIS_001",
            scenario_type=ScenarioType.CRISIS_RESPONSE,
            title="NAV Pricing Emergency",
            description="Handle pricing emergency affecting NAV",
            difficulty=5,
            primary_agent="SANTAS_HELPER",
            secondary_agent="CPA",
            inputs={
                "issue": "Illiquid position fair value challenge",
                "impact": "Potential NAV variance of 2%",
                "investors_affected": 50
            },
            expected_outputs={
                "issue_identified": True,
                "chris_notified": True,
                "resolution_proposed": True,
                "lp_communication_drafted": True
            },
            success_criteria=[
                "Rapid escalation to Chris",
                "Clear problem statement",
                "Proposed solutions with trade-offs",
                "Coordinated communication plan"
            ]
        ))

        return scenarios

    def initialize_agents(self):
        """Initialize agent references"""
        try:
            from src.agents.santas_helper_agent import get_santas_helper
            from src.agents.cpa_agent import get_cpa
            from src.agents.orchestrator_agent import get_orchestrator

            self.santas_helper = get_santas_helper()
            self.cpa = get_cpa()
            self.orchestrator = get_orchestrator()

            logger.info("All agents initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            return False

    def run_training_session(
        self,
        agent_name: str,
        scenario: TrainingScenario,
        verbose: bool = True
    ) -> TrainingResult:
        """Run a single training session"""
        self.session_count += 1
        session_id = f"TRAIN_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.session_count:04d}"

        result = TrainingResult(
            session_id=session_id,
            agent_name=agent_name,
            scenario=scenario,
            started_at=datetime.now()
        )

        if verbose:
            print(f"\n{'='*60}")
            print(f"Training Session: {session_id}")
            print(f"Agent: {agent_name}")
            print(f"Scenario: {scenario.title}")
            print(f"Difficulty: {'*' * scenario.difficulty}")
            print(f"{'='*60}")

        try:
            # Get the appropriate agent
            agent = self.santas_helper if agent_name == "SANTAS_HELPER" else self.cpa

            if agent is None:
                raise ValueError(f"Agent {agent_name} not initialized")

            # Execute the scenario
            start_time = datetime.now()

            # Map scenario type to action
            action_mapping = {
                ScenarioType.NAV_CALCULATION: "calculate_nav",
                ScenarioType.FEE_CALCULATION: "calculate_performance_fee",
                ScenarioType.TAX_PREPARATION: "generate_k1s",
                ScenarioType.AUDIT_COORDINATION: "coordinate_audit",
                ScenarioType.LP_REPORTING: "generate_lp_report",
                ScenarioType.YEAR_END_CLOSE: "run_daily_operations",
                ScenarioType.CRISIS_RESPONSE: "report_to_chris",
                ScenarioType.INVESTOR_COMMUNICATION: "generate_lp_report"
            }

            action = action_mapping.get(scenario.scenario_type, "get_status")

            # Process the task
            response = agent.process({
                "action": action,
                **scenario.inputs
            })

            end_time = datetime.now()
            response_time = int((end_time - start_time).total_seconds() * 1000)

            # Evaluate result
            result.completed_at = end_time
            result.response_time_ms = response_time
            result.success = response.get("status") == "success"
            result.accuracy_score = self._evaluate_accuracy(scenario, response)
            result.feedback = self._generate_feedback(scenario, response, result.accuracy_score)
            result.improvements = self._suggest_improvements(scenario, response)

            if verbose:
                print(f"\nResult: {'[OK] SUCCESS' if result.success else '[FAIL] FAILED'}")
                print(f"Accuracy: {result.accuracy_score:.1%}")
                print(f"Response Time: {result.response_time_ms}ms")
                print(f"Feedback: {result.feedback}")
                if result.improvements:
                    print(f"Improvements: {', '.join(result.improvements)}")

        except Exception as e:
            result.completed_at = datetime.now()
            result.success = False
            result.feedback = f"Error during training: {str(e)}"
            logger.error(f"Training error: {e}")

        self.training_results.append(result)
        return result

    def _evaluate_accuracy(self, scenario: TrainingScenario, response: Dict) -> float:
        """Evaluate accuracy of agent response"""
        # Simplified accuracy calculation
        # In production, this would compare actual vs expected outputs
        if response.get("status") != "success":
            return 0.0

        # Check if key expected outputs are present
        score = 0.5  # Base score for successful response

        expected = scenario.expected_outputs
        for key in expected:
            if key in str(response):
                score += 0.1

        return min(1.0, score)

    def _generate_feedback(self, scenario: TrainingScenario, response: Dict, accuracy: float) -> str:
        """Generate feedback for the training session"""
        if accuracy >= 0.9:
            return "Excellent performance. All criteria met."
        elif accuracy >= 0.7:
            return "Good performance. Minor improvements needed."
        elif accuracy >= 0.5:
            return "Acceptable performance. Review success criteria."
        else:
            return "Needs improvement. Review scenario requirements."

    def _suggest_improvements(self, scenario: TrainingScenario, response: Dict) -> List[str]:
        """Suggest improvements based on training results"""
        improvements = []

        if scenario.scenario_type == ScenarioType.NAV_CALCULATION:
            improvements.append("Consider additional pricing source validation")
        elif scenario.scenario_type == ScenarioType.TAX_PREPARATION:
            improvements.append("Review K-1 box requirements for completeness")
        elif scenario.scenario_type == ScenarioType.AUDIT_COORDINATION:
            improvements.append("Optimize PBC list tracking workflow")

        return improvements

    def run_phase_training(
        self,
        phase: TrainingPhase,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Run all scenarios for a training phase"""
        if verbose:
            print(f"\n{'#'*60}")
            print(f"# STARTING TRAINING PHASE: {phase.value.upper()}")
            print(f"{'#'*60}")

        phase_results = {
            "phase": phase.value,
            "started_at": datetime.now().isoformat(),
            "scenarios_completed": 0,
            "success_rate": 0.0,
            "avg_accuracy": 0.0,
            "results": []
        }

        # Filter scenarios based on phase
        if phase == TrainingPhase.INDIVIDUAL:
            scenarios = [s for s in self.scenarios if s.secondary_agent is None]
        elif phase == TrainingPhase.CROSS_TRAINING:
            scenarios = [s for s in self.scenarios if s.secondary_agent is not None]
        else:
            scenarios = self.scenarios

        successes = 0
        total_accuracy = 0.0

        for scenario in scenarios:
            result = self.run_training_session(
                scenario.primary_agent,
                scenario,
                verbose=verbose
            )

            phase_results["results"].append(result.to_dict())
            phase_results["scenarios_completed"] += 1

            if result.success:
                successes += 1
            total_accuracy += result.accuracy_score

        if phase_results["scenarios_completed"] > 0:
            phase_results["success_rate"] = successes / phase_results["scenarios_completed"]
            phase_results["avg_accuracy"] = total_accuracy / phase_results["scenarios_completed"]

        phase_results["completed_at"] = datetime.now().isoformat()

        if verbose:
            print(f"\n{'='*60}")
            print(f"Phase Complete: {phase.value}")
            print(f"Scenarios: {phase_results['scenarios_completed']}")
            print(f"Success Rate: {phase_results['success_rate']:.1%}")
            print(f"Average Accuracy: {phase_results['avg_accuracy']:.1%}")
            print(f"{'='*60}")

        return phase_results

    def run_full_training(self, verbose: bool = True) -> Dict[str, Any]:
        """Run complete training for both agents"""
        training_summary = {
            "training_id": f"FULL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "started_at": datetime.now().isoformat(),
            "phases": []
        }

        if verbose:
            print("\n" + "="*70)
            print("FUND OPERATIONS AGENT TRAINING - FULL PROGRAM")
            print("Agents: SANTAS_HELPER, CPA")
            print("="*70)

        # Initialize agents
        if not self.initialize_agents():
            return {"error": "Failed to initialize agents"}

        # Run all phases
        for phase in TrainingPhase:
            phase_result = self.run_phase_training(phase, verbose=verbose)
            training_summary["phases"].append(phase_result)

        training_summary["completed_at"] = datetime.now().isoformat()

        # Calculate overall metrics
        total_scenarios = sum(p["scenarios_completed"] for p in training_summary["phases"])
        total_success = sum(p["success_rate"] * p["scenarios_completed"] for p in training_summary["phases"])

        if total_scenarios > 0:
            training_summary["overall_success_rate"] = total_success / total_scenarios
        else:
            training_summary["overall_success_rate"] = 0.0

        if verbose:
            print("\n" + "="*70)
            print("TRAINING COMPLETE")
            print(f"Total Scenarios: {total_scenarios}")
            print(f"Overall Success Rate: {training_summary['overall_success_rate']:.1%}")
            print("="*70)

        return training_summary

    def get_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        return {
            "total_sessions": len(self.training_results),
            "sessions_by_agent": {
                "SANTAS_HELPER": len([r for r in self.training_results if r.agent_name == "SANTAS_HELPER"]),
                "CPA": len([r for r in self.training_results if r.agent_name == "CPA"])
            },
            "success_rate": sum(1 for r in self.training_results if r.success) / max(len(self.training_results), 1),
            "avg_accuracy": sum(r.accuracy_score for r in self.training_results) / max(len(self.training_results), 1),
            "avg_response_time_ms": sum(r.response_time_ms for r in self.training_results) / max(len(self.training_results), 1),
            "recent_sessions": [r.to_dict() for r in self.training_results[-10:]]
        }


# ===============================================================================
# ANONYMIZED DATA COLLECTION FOR TRAINING
# ===============================================================================

class TrainingDataCollector:
    """
    Collects anonymized conversation data for training additional agents.
    All data is anonymized and stored securely in Azure.
    """

    def __init__(self, azure_container: str = "training-data"):
        self.container = azure_container
        self.collected_samples: List[Dict] = []
        self.session_id = f"collect_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def anonymize_data(self, data: Dict) -> Dict:
        """Remove personally identifiable information"""
        anonymized = data.copy()

        # Remove PII fields
        pii_fields = [
            "investor_name", "email", "phone", "address", "ssn", "ein",
            "account_number", "bank_name"
        ]

        for field in pii_fields:
            if field in anonymized:
                anonymized[field] = f"[REDACTED_{field.upper()}]"

        # Hash investor IDs
        if "investor_id" in anonymized:
            anonymized["investor_id"] = hashlib.sha256(
                str(anonymized["investor_id"]).encode()
            ).hexdigest()[:12]

        return anonymized

    def collect_conversation(
        self,
        user: str,
        agent: str,
        request: Dict,
        response: Dict,
        feedback: Optional[str] = None
    ):
        """Collect and anonymize a conversation for training"""
        sample = {
            "sample_id": f"sample_{len(self.collected_samples):05d}",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "user_role": "principal" if user in ["Chris", "Tom"] else "user",
            "agent": agent,
            "request": self.anonymize_data(request),
            "response": self.anonymize_data(response),
            "feedback": feedback,
            "quality_score": None  # To be rated later
        }

        self.collected_samples.append(sample)
        return sample["sample_id"]

    def export_for_training(self) -> Dict:
        """Export collected data for training"""
        return {
            "session_id": self.session_id,
            "export_timestamp": datetime.now().isoformat(),
            "sample_count": len(self.collected_samples),
            "samples": self.collected_samples
        }

    def save_to_azure(self) -> bool:
        """Save collected data to Azure Blob Storage"""
        try:
            from src.utils.azure_storage import azure_storage

            blob_name = f"training_data_{self.session_id}.json"
            data = self.export_for_training()

            azure_storage.save_object(self.container, blob_name, data)
            logger.info(f"Saved {len(self.collected_samples)} samples to {blob_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to save to Azure: {e}")
            return False


# ===============================================================================
# CLI INTERFACE
# ===============================================================================

def main():
    """Main entry point for fund operations training"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train SANTAS_HELPER and CPA agents"
    )
    parser.add_argument(
        "mode",
        choices=["full", "individual", "cross", "report"],
        help="Training mode: full (all phases), individual, cross (cross-training), report"
    )
    parser.add_argument(
        "--agent",
        choices=["SANTAS_HELPER", "CPA", "both"],
        default="both",
        help="Agent to train"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output"
    )

    args = parser.parse_args()

    trainer = FundOpsTrainer()

    if args.mode == "full":
        result = trainer.run_full_training(verbose=args.verbose)
        print(f"\nTraining Summary: {json.dumps(result, indent=2, default=str)}")

    elif args.mode == "individual":
        trainer.initialize_agents()
        result = trainer.run_phase_training(TrainingPhase.INDIVIDUAL, verbose=args.verbose)
        print(f"\nPhase Result: {json.dumps(result, indent=2, default=str)}")

    elif args.mode == "cross":
        trainer.initialize_agents()
        result = trainer.run_phase_training(TrainingPhase.CROSS_TRAINING, verbose=args.verbose)
        print(f"\nPhase Result: {json.dumps(result, indent=2, default=str)}")

    elif args.mode == "report":
        report = trainer.get_training_report()
        print(f"\nTraining Report: {json.dumps(report, indent=2, default=str)}")


if __name__ == "__main__":
    main()

