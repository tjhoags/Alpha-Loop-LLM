"""
Alpha Loop Capital - Agent Registry
====================================

OWNERSHIP STRUCTURE:
- Tom Hogan (Founder & CIO) - Investment Division via HOAGS
- Chris Friedman (COO) - Operations Division via FRIEDS

MASTER AGENTS (Partners):
- HOAGS: Tom Hogan's authority agent (Investment)
- GHOST: Shared autonomous coordinator (Both divisions)
- FRIEDS: Chris Friedman's authority agent (Operations)

AGENT BREAKDOWN:
- Investment Domain: 72 agents
  - Master (HOAGS, GHOST): 2
  - Senior: 10
  - Operational: 8
  - Strategy: 34
  - Sector: 11
  - Security: 2
  - Swarm: 5

- Operations Domain: 12 agents
  - Master (FRIEDS): 1
  - Senior (SANTAS_HELPER, CPA, MARKETING, SOFTWARE, BUSINESS_DEV): 5
  - Sub-agents: 10

TOTAL: 94 agents

AUTHORITATIVE AGENT LOCATIONS:
- Master agents: hoags_agent/, ghost_agent/, operations/frieds_agent.py
- Senior (Investment): senior/
- Senior (Operations): santas_helper_agent/, cpa_agent/
- Operational: data_agent/, execution_agent/, etc.
- Specialized: specialized/
- Sectors: sectors/
- Security: hackers/
- Swarm: swarm/
"""

# =============================================================================
# IMPORTS - Master Agents
# =============================================================================
from src.agents.hoags_agent.hoags_agent import HoagsAgent, get_hoags
from src.agents.ghost_agent.ghost_agent import GhostAgent, get_ghost
from src.agents.operations.frieds_agent import FriedsAgent, get_frieds

# =============================================================================
# IMPORTS - Executive Assistants
# =============================================================================
from src.agents.kat_agent.kat_agent import KatAgent, get_kat
from src.agents.shyla_agent.shyla_agent import ShylaAgent, get_shyla
from src.agents.co_assistants.margot_robbie import MargotRobbieAgent, get_margot
from src.agents.co_assistants.anna_kendrick import AnnaKendrickAgent, get_anna

# =============================================================================
# IMPORTS - Senior Operations Agents (Chris Friedman's division)
# =============================================================================
from src.agents.santas_helper_agent.santas_helper_agent import (
    SantasHelperAgent,
    get_santas_helper,
)
from src.agents.cpa_agent.cpa_agent import CPAAgent, get_cpa
from src.agents.cpa_agent.tax_rules_engine import (
    TaxRulesDatabase,
    TaxAnalysisEngine,
    get_tax_rules_database,
    get_tax_analysis_engine,
)
from src.agents.marketing_agent.marketing_agent import MarketingAgent, get_marketing
from src.agents.software_agent.software_agent import SoftwareAgent, get_software
from src.agents.business_dev_agent.business_dev_agent import BusinessDevAgent, get_business_dev

# =============================================================================
# CONSTANTS
# =============================================================================

# Total agent count
# Includes: KAT, SHYLA, MARGOT_ROBBIE, ANNA_KENDRICK, MARKETING, SOFTWARE,
# BUSINESS_DEV and their sub-agents (COFFEE_BREAK, BEAN_COUNTER for SHYLA)
TOTAL_AGENTS = 94

# Division constants
INVESTMENT_DIVISION = "INVESTMENT"
OPERATIONS_DIVISION = "OPERATIONS"

# Master agents
MASTER_AGENTS = {
    "HOAGS": {
        "owner": "TOM_HOGAN",
        "division": INVESTMENT_DIVISION,
        "role": "Investment Authority"
    },
    "GHOST": {
        "owner": "SHARED",
        "division": "BOTH",
        "role": "Autonomous Coordinator"
    },
    "FRIEDS": {
        "owner": "CHRIS_FRIEDMAN",
        "division": OPERATIONS_DIVISION,
        "role": "Operations Authority"
    }
}

# Owner contacts
OWNERS = {
    "TOM_HOGAN": {
        "role": "Founder & CIO",
        "ownership": "MAJORITY",
        "email": "tom@alphaloopcapital.com",
        "division": INVESTMENT_DIVISION,
        "authority_agent": "HOAGS",
        "executive_assistant": "KAT"
    },
    "CHRIS_FRIEDMAN": {
        "role": "COO",
        "ownership": "MINORITY",
        "email": "chris@alphaloopcapital.com",
        "division": OPERATIONS_DIVISION,
        "authority_agent": "FRIEDS",
        "executive_assistant": "SHYLA"
    }
}

# Agent hierarchy
HIERARCHY = {
    INVESTMENT_DIVISION: {
        "master": ["HOAGS", "GHOST"],
        "senior": ["SCOUT", "HUNTER", "ORCHESTRATOR", "KILLJOY", "BOOKMAKER",
                   "STRINGS", "AUTHOR", "SKILLS", "CAPITAL", "NOBUS"],
        "operational": ["DATA_AGENT", "EXECUTION_AGENT", "COMPLIANCE_AGENT",
                       "PORTFOLIO_AGENT", "RISK_AGENT", "SENTIMENT_AGENT",
                       "RESEARCH_AGENT", "STRATEGY_AGENT"],
        "strategy": 34,
        "sector": 11,
        "security": ["WHITE_HAT", "BLACK_HAT"],
        "swarm": 5
    },
    OPERATIONS_DIVISION: {
        "master": ["FRIEDS"],
        "executive_assistants": {
            "tom": ["KAT"],           # Tom's EA (reports to HOAGS)
            "chris": ["SHYLA"],       # Chris's EA (reports to FRIEDS)
            "shared": ["MARGOT_ROBBIE", "ANNA_KENDRICK"],  # Co-EAs
        },
        "senior": ["SANTAS_HELPER", "CPA", "MARKETING", "SOFTWARE", "BUSINESS_DEV"],
        "shared_with_investment": ["GHOST", "ORCHESTRATOR", "NOBUS"],
        "sub_agents": {
            "KAT": [],  # Security: READ-ONLY by default
            "SHYLA": ["COFFEE_BREAK", "BEAN_COUNTER"],
            "SANTAS_HELPER": ["NAV_SPECIALIST", "GL_ACCOUNTANT", "PERFORMANCE_ANALYST",
                             "INVESTOR_RELATIONS", "ADMIN_COORDINATOR"],
            "CPA": ["TAX_JUNIOR", "AUDIT_JUNIOR", "REPORTING_JUNIOR"]
        }
    }
}

# =============================================================================
# EXPORTS
# =============================================================================
__all__ = [
    # Constants
    "TOTAL_AGENTS",
    "INVESTMENT_DIVISION",
    "OPERATIONS_DIVISION",
    "MASTER_AGENTS",
    "OWNERS",
    "HIERARCHY",
    # Master Agents
    "HoagsAgent",
    "get_hoags",
    "GhostAgent",
    "get_ghost",
    "FriedsAgent",
    "get_frieds",
    # Executive Assistants
    "KatAgent",
    "get_kat",
    "ShylaAgent",
    "get_shyla",
    "MargotRobbieAgent",
    "get_margot",
    "AnnaKendrickAgent",
    "get_anna",
    # Operations Agents
    "SantasHelperAgent",
    "get_santas_helper",
    "CPAAgent",
    "get_cpa",
    "MarketingAgent",
    "get_marketing",
    "SoftwareAgent",
    "get_software",
    "BusinessDevAgent",
    "get_business_dev",
    # Tax Rules Engine
    "TaxRulesDatabase",
    "TaxAnalysisEngine",
    "get_tax_rules_database",
    "get_tax_analysis_engine",
]
