"""
================================================================================
Initialize Clone Product Development Teams
================================================================================
Context Box 1: Phase 1 - Product Selection
Mission: Initialize product development teams via ACA and get HOAGS approval

This script:
1. Detects capability gaps for product development teams
2. Submits ACA proposals for team lead agents
3. Requests HOAGS approval
4. Creates initial team structure

Agents to create:
- BUSINESS_DEV_LEAD: Market research, competitive analysis, pricing
- MARKETING_TEAM_LEAD: Brand positioning, go-to-market, content
- PRODUCT_TEAM_LEAD: Product requirements, user flows, roadmap
- BACKEND_TEAM_LEAD: API design, database, infrastructure
- FRONTEND_TEAM_LEAD: UI/UX, landing page, app interface
================================================================================
"""

import hashlib
import logging
from datetime import datetime
from typing import Dict, List

from src.core.aca_engine import get_aca_engine
from src.agents.hoags_agent.hoags_agent import get_hoags
from src.agents.orchestrator_agent.orchestrator_agent import get_orchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Team Lead Agent Specifications
TEAM_LEAD_SPECS = [
    {
        "agent_name": "BUSINESS_DEV_LEAD",
        "agent_tier": "SENIOR",
        "capabilities": [
            "market_research",
            "competitive_analysis",
            "pricing_strategy",
            "revenue_modeling",
            "b2b_product_analysis",
            "market_opportunity_assessment",
        ],
        "gap_description": "Need specialized business development lead for B2B product clone analysis and market research",
        "rationale": "To analyze successful B2B products, identify clone candidates, evaluate market opportunities, and develop pricing strategies for fast ROI",
        "priority": "high",
    },
    {
        "agent_name": "MARKETING_TEAM_LEAD",
        "agent_tier": "SENIOR",
        "capabilities": [
            "brand_positioning",
            "go_to_market_strategy",
            "content_marketing",
            "product_hunt_launch",
            "viral_growth_design",
            "seo_strategy",
            "conversion_optimization",
        ],
        "gap_description": "Need marketing team lead for world-class product launch and growth strategy",
        "rationale": "To create compelling brand positioning, execute Product Hunt launch, design viral growth loops, and drive user acquisition",
        "priority": "high",
    },
    {
        "agent_name": "PRODUCT_TEAM_LEAD",
        "agent_tier": "SENIOR",
        "capabilities": [
            "product_requirements",
            "user_flow_design",
            "feature_prioritization",
            "product_roadmap",
            "user_research",
            "product_market_fit_analysis",
        ],
        "gap_description": "Need product team lead for product requirements, user flows, and feature roadmap",
        "rationale": "To define MVP features, design user journeys, prioritize roadmap, and ensure product-market fit",
        "priority": "high",
    },
    {
        "agent_name": "BACKEND_TEAM_LEAD",
        "agent_tier": "SENIOR",
        "capabilities": [
            "api_design",
            "database_architecture",
            "backend_development",
            "infrastructure_setup",
            "scalability_planning",
            "third_party_integrations",
        ],
        "gap_description": "Need backend team lead for API, database, and infrastructure design (zero Alpha Loop dependencies)",
        "rationale": "To design and build standalone backend with modern tech stack (FastAPI/Express, PostgreSQL, Redis) completely separate from Alpha Loop codebase",
        "priority": "critical",
    },
    {
        "agent_name": "FRONTEND_TEAM_LEAD",
        "agent_tier": "SENIOR",
        "capabilities": [
            "frontend_development",
            "ui_ux_design",
            "landing_page_creation",
            "nextjs_development",
            "component_library",
            "design_system",
        ],
        "gap_description": "Need frontend team lead for UI, landing page, and app interface (zero Alpha Loop dependencies)",
        "rationale": "To build beautiful, conversion-optimized frontend with Next.js 14, Tailwind CSS, and shadcn/ui completely separate from Alpha Loop codebase",
        "priority": "critical",
    },
]


def generate_proposal_id(agent_name: str) -> str:
    """Generate unique proposal ID."""
    timestamp = datetime.now().isoformat()
    combined = f"{agent_name}_{timestamp}"
    return hashlib.sha256(combined.encode()).hexdigest()[:12]


def initialize_teams() -> Dict[str, any]:
    """
    Initialize product development teams via ACA.
    
    Returns:
        Dict with proposal IDs and status
    """
    logger.info("=" * 80)
    logger.info("INITIALIZING CLONE PRODUCT DEVELOPMENT TEAMS")
    logger.info("=" * 80)
    
    aca_engine = get_aca_engine()
    orchestrator = get_orchestrator()
    hoags = get_hoags()
    
    results = {
        "proposals_submitted": [],
        "proposals_approved": [],
        "proposals_rejected": [],
        "teams_initialized": False,
    }
    
    # Step 1: Detect capability gaps via ORCHESTRATOR
    logger.info("\n[Step 1] Detecting capability gaps...")
    gap_context = {
        "task_type": "b2b_product_clone_launch",
        "required_capabilities": [
            "market_research",
            "competitive_analysis",
            "brand_positioning",
            "product_requirements",
            "backend_development",
            "frontend_development",
        ],
        "missing_capabilities": [
            "specialized_b2b_product_analysis",
            "product_launch_strategy",
            "standalone_backend_development",
            "standalone_frontend_development",
        ],
    }
    
    gap_id = f"clone_teams_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    gap_record = aca_engine.register_gap(
        gap_id=gap_id,
        detected_by="ORCHESTRATOR",
        task_type=gap_context["task_type"],
        required_capabilities=gap_context["required_capabilities"],
        missing_capabilities=gap_context["missing_capabilities"],
        severity="high",
        context=gap_context,
    )
    logger.info(f"✓ Capability gap registered: {gap_id}")
    
    # Step 2: Submit ACA proposals for each team lead
    logger.info("\n[Step 2] Submitting ACA proposals for team leads...")
    
    for spec in TEAM_LEAD_SPECS:
        proposal_id = generate_proposal_id(spec["agent_name"])
        
        proposal = aca_engine.submit_proposal(
            proposal_id=proposal_id,
            proposed_by="ORCHESTRATOR",
            agent_name=spec["agent_name"],
            agent_tier=spec["agent_tier"],
            capabilities=spec["capabilities"],
            gap_description=spec["gap_description"],
            rationale=spec["rationale"],
            priority=spec["priority"],
        )
        
        results["proposals_submitted"].append({
            "proposal_id": proposal_id,
            "agent_name": spec["agent_name"],
            "status": "pending",
        })
        
        logger.info(f"✓ Proposal submitted: {spec['agent_name']} (ID: {proposal_id})")
        logger.info(f"  Capabilities: {', '.join(spec['capabilities'][:3])}...")
    
    # Step 3: Request HOAGS approval
    logger.info("\n[Step 3] Requesting HOAGS approval...")
    
    pending_proposals = aca_engine.get_pending_proposals()
    logger.info(f"Found {len(pending_proposals)} pending proposals")
    
    # Present proposals to HOAGS for review
    approval_results = []
    for proposal_data in results["proposals_submitted"]:
        proposal_id = proposal_data["proposal_id"]
        agent_name = proposal_data["agent_name"]
        
        # Get full proposal details
        proposal = aca_engine.get_proposal(proposal_id)
        if not proposal:
            logger.warning(f"⚠ Proposal not found: {proposal_id}")
            continue
        
        # Request HOAGS approval
        logger.info(f"\nRequesting approval for {agent_name}...")
        logger.info(f"  Rationale: {proposal.rationale[:100]}...")
        
        # HOAGS reviews and approves
        approval_task = {
            "action": "approve_aca_proposal",
            "proposal_id": proposal_id,
            "auto_create": False,  # Don't create yet, just approve
        }
        
        approval_result = hoags.process(approval_task)
        
        if approval_result.get("success"):
            results["proposals_approved"].append({
                "proposal_id": proposal_id,
                "agent_name": agent_name,
                "approved_by": approval_result.get("approved_by", "HOAGS"),
            })
            logger.info(f"✓ APPROVED: {agent_name}")
        else:
            results["proposals_rejected"].append({
                "proposal_id": proposal_id,
                "agent_name": agent_name,
                "reason": approval_result.get("error", "Unknown"),
            })
            logger.warning(f"✗ REJECTED: {agent_name} - {approval_result.get('error')}")
    
    # Step 4: Summary
    logger.info("\n" + "=" * 80)
    logger.info("INITIALIZATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Proposals Submitted: {len(results['proposals_submitted'])}")
    logger.info(f"Proposals Approved: {len(results['proposals_approved'])}")
    logger.info(f"Proposals Rejected: {len(results['proposals_rejected'])}")
    
    if len(results["proposals_approved"]) == len(TEAM_LEAD_SPECS):
        results["teams_initialized"] = True
        logger.info("\n✓ ALL TEAMS INITIALIZED - Ready for Phase 2: Product Analysis")
    else:
        logger.warning(f"\n⚠ PARTIAL INITIALIZATION - {len(results['proposals_approved'])}/{len(TEAM_LEAD_SPECS)} teams approved")
    
    # Save results to file
    output_file = "docs/clone_analysis/TEAM_INITIALIZATION.md"
    try:
        import os
        os.makedirs("docs/clone_analysis", exist_ok=True)
        
        with open(output_file, "w") as f:
            f.write("# Product Development Teams - Initialization Report\n\n")
            f.write(f"**Date:** {datetime.now().isoformat()}\n\n")
            f.write("## Summary\n\n")
            f.write(f"- Proposals Submitted: {len(results['proposals_submitted'])}\n")
            f.write(f"- Proposals Approved: {len(results['proposals_approved'])}\n")
            f.write(f"- Proposals Rejected: {len(results['proposals_rejected'])}\n\n")
            
            f.write("## Approved Teams\n\n")
            for approved in results["proposals_approved"]:
                f.write(f"### {approved['agent_name']}\n")
                f.write(f"- Proposal ID: {approved['proposal_id']}\n")
                f.write(f"- Approved By: {approved['approved_by']}\n\n")
            
            if results["proposals_rejected"]:
                f.write("## Rejected Proposals\n\n")
                for rejected in results["proposals_rejected"]:
                    f.write(f"### {rejected['agent_name']}\n")
                    f.write(f"- Proposal ID: {rejected['proposal_id']}\n")
                    f.write(f"- Reason: {rejected['reason']}\n\n")
        
        logger.info(f"\n✓ Results saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
    
    return results


if __name__ == "__main__":
    results = initialize_teams()
    
    print("\n" + "=" * 80)
    print("CONTEXT BOX 1 COMPLETE")
    print("=" * 80)
    print("\nNext Steps:")
    print("1. Review approved teams in docs/clone_analysis/TEAM_INITIALIZATION.md")
    print("2. Proceed to Phase 2: Product Analysis")
    print("3. Run: scripts/analyze_clone_candidates.py")
    print("=" * 80)

