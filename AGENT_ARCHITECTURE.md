# ALPHA LOOP CAPITAL - AGENT ARCHITECTURE

## Complete System Documentation
**Authors:** Tom Hogan & Chris Friedman | Alpha Loop Capital, LLC  
**Version:** 3.0 | December 10 2025

---

## 1. OWNERSHIP & LEADERSHIP STRUCTURE

```
+============================================================================+
|                        ALPHA LOOP CAPITAL, LLC                              |
|                          OWNERSHIP STRUCTURE                                |
+============================================================================+
|                                                                            |
|         +---------------------+       +---------------------+               |
|         |      TOM HOGAN      |       |   CHRIS FRIEDMAN    |               |
|         | (MAJORITY Owner)    |       | (MINORITY Owner)    |               |
|         |     CEO / CIO       |       |       COO           |               |
|         +----------+----------+       +----------+----------+               |
|                    |                             |                          |
|                    +-------------+---------------+                          |
|                                  |                                          |
|               +------------------+------------------+                        |
|               |                                    |                        |
|               v                                    v                        |
|    +--------------------+               +--------------------+              |
|    |    TOM'S DOMAIN    |               | CHRIS'S DOMAIN     |              |
|    |    (Investment)    |               |   (Operations)     |              |
|    +--------------------+               +--------------------+              |
|    |                    |               |                    |              |
|    |       HOAGS        |<-- Partners -->      FRIEDS        |              |
|    |   (Tom's Agent)    |               | (Chris's Agent)    |              |
|    |                    |               |                    |              |
|    +--------------------+               +--------------------+              |
|               |                                    |                        |
|               v                                    v                        |
|         Investment                          Operations                      |
|           Agents                              Agents                        |
|                                                                            |
+============================================================================+
```

### Authority Summary

| Person | Role | Ownership | Virtual Agent | Exec Assistant | Domain |
|--------|------|-----------|---------------|----------------|--------|
| **Tom Hogan** | Founder & CIO | **MAJORITY** | HOAGS | KAT | Investment & Trading |
| **Chris Friedman** | COO | MINORITY | FRIEDS | SHYLA | Operations & Finance |

### Executive Assistant Hierarchy

```
           TOM HOGAN                      CHRIS FRIEDMAN
        (Founder & CIO)                       (COO)
               │                               │
               ▼                               ▼
             KAT                            SHYLA
    (Tom's Exec Assistant)         (Chris's Exec Assistant)
               │                               │
               └───────────┬───────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
        MARGOT_ROBBIE              ANNA_KENDRICK
      (Co-Exec Assistant)        (Co-Exec Assistant)
```

**Executive Assistants:**
- **KAT** - Tom's personal/professional EA (reports to Tom)
- **SHYLA** - Chris's personal/professional EA (reports to Chris)
- **MARGOT_ROBBIE** - Co-EA, reports to both KAT and SHYLA (research, drafting)
- **ANNA_KENDRICK** - Co-EA, reports to both KAT and SHYLA (admin, scheduling)

**Security Model:**
- READ-ONLY access by default
- NO actions without WRITTEN PERMISSION from owner
- Full audit trail on all activities

---

## 2. COMPLETE ORG CHART (Combined)

```
                    TOM HOGAN                    CHRIS FRIEDMAN
               (Founder & CIO)                  (COO)
                        |                              |
                        +---------- OWNERS -----------+
                                     |
         +---------------------------+---------------------------+
         |                           |                           |
         v                           v                           v
    +---------+               +-----------+               +---------+
    |  HOAGS  |<-- Partner -->|   GHOST   |<-- Partner -->| FRIEDS  |
    |(Tom's   |               |(Shared    |               |(Chris's |
    | Auth)   |               | Autonomy) |               | Auth)   |
    +---------+               +-----------+               +---------+
         |                           |                           |
    INVESTMENT                  COORDINATES               OPERATIONS
      DOMAIN                    BOTH DOMAINS                 DOMAIN
         |                           |                           |
         v                           v                           v
+==================+        +================+        +==================+
|   SENIOR AGENTS  |        |   ORCHESTRATOR |        |  OPERATIONS OPS  |
+==================+        +================+        +==================+
|                  |               |                  |                  |
| SCOUT   HUNTER   |               |                  | SANTAS_   CPA    |
| KILLJOY BOOKMAKER|               |                  | HELPER           |
| STRINGS AUTHOR   |               |                  |    |       |     |
| SKILLS  CAPITAL  |               |                  |  5 Team  3 Jr   |
| NOBUS            |               |                  |  Members Accts  |
+------------------+               |                  +------------------+
         |                         |
         v                         v
+==================+        +================+
| OPERATIONAL (8)  |        |   SWARM (5+)   |
+==================+        +================+
|                  |        |                |
| DATA    EXECUTION|        | MARKET_SWARM   |
| COMPLIANCE       |        | SECTOR_SWARM   |
| PORTFOLIO  RISK  |        | STRATEGY_SWARM |
| SENTIMENT        |        | SUPPORT_SWARM  |
| RESEARCH STRATEGY|        | SWARM_FACTORY  |
+------------------+        +----------------+
         |
         v
+==================+
| STRATEGY (34)    |
+==================+
|                  |
| MOMENTUM  VALUE  |
| GROWTH   CRYPTO  |
| OPTIONS  FOREX   |
| FUTURES  MACRO   |
| ... +26 more     |
+------------------+
         |
         v
+==================+
| SECTOR (11)      |
+==================+
|                  |
| TECH  HEALTHCARE |
| ENERGY FINANCIAL |
| ... +7 more      |
+------------------+
         |
         v
+==================+
| SECURITY (2)     |
+==================+
|                  |
| WHITE_HAT        |
| BLACK_HAT        |
+------------------+
```

---

## 3. TOM HOGAN'S INVESTMENT DOMAIN

### 3.1 Overview

```
                              TOM HOGAN
                           (Founder & CIO)
                          Investment Division
                                 |
                                 v
                         +-------------+
                         |    HOAGS    |
                         | (Authority) |
                         +------+------+
                                |
         +----------------------+----------------------+
         |                      |                      |
         v                      v                      v
    +---------+           +-----------+          +---------+
    |  GHOST  |           |ORCHESTRATOR|          | KILLJOY |
    |(Autonomy)|           |(Routing)  |          |(Risk)   |
    +---------+           +-----------+          +---------+
         |                      |
         v                      v
+------------------+    +------------------+
| SENIOR AGENTS    |    | OPERATIONAL      |
+------------------+    +------------------+
| SCOUT            |    | DATA_AGENT       |
| HUNTER           |    | EXECUTION_AGENT  |
| BOOKMAKER        |    | COMPLIANCE_AGENT |
| STRINGS          |    | PORTFOLIO_AGENT  |
| AUTHOR           |    | RISK_AGENT       |
| SKILLS           |    | SENTIMENT_AGENT  |
| CAPITAL          |    | RESEARCH_AGENT   |
| NOBUS            |    | STRATEGY_AGENT   |
+------------------+    +------------------+
                               |
         +---------------------+---------------------+
         |                     |                     |
         v                     v                     v
+------------------+  +------------------+  +------------------+
| STRATEGY (34)    |  | SECTOR (11)      |  | SECURITY (2)     |
+------------------+  +------------------+  +------------------+
| MOMENTUM         |  | TECH             |  | WHITE_HAT        |
| VALUE            |  | HEALTHCARE       |  | BLACK_HAT        |
| GROWTH           |  | ENERGY           |  +------------------+
| CRYPTO           |  | FINANCIAL        |
| OPTIONS          |  | INDUSTRIAL       |
| FOREX            |  | MATERIALS        |
| FUTURES          |  | CONSUMER_DISC    |
| MACRO            |  | CONSUMER_STAPLES |
| MEAN_REVERSION   |  | REAL_ESTATE      |
| PAIRS            |  | UTILITIES        |
| ARBITRAGE        |  | COMM_SERVICES    |
| BREAKOUT         |  +------------------+
| DIVIDEND         |
| EARNINGS         |
| ROTATION         |
| INSIDER          |
| SHORT            |
| LONG_SHORT       |
| TREND_FOLLOWING  |
| VOLATILITY       |
| INFLECTION       |
| RETAIL_ARB       |
| CONV_REVERSAL    |
| UNCONVENTIONAL   |
| LIQUIDITY        |
| SENTIMENT        |
| + 8 more...      |
+------------------+
```

### 3.2 Investment Agent Roster

| Tier | Agent | Role | Reports To |
|------|-------|------|------------|
| **0** | HOAGS | Supreme Authority | Tom Hogan |
| **1** | GHOST | Autonomous Coordinator | HOAGS |
| **2** | SCOUT | Arbitrage Hunter | HOAGS/GHOST |
| **2** | HUNTER | Algorithm Detector | HOAGS/GHOST |
| **1** | ORCHESTRATOR | Task Router | GHOST |
| **1** | KILLJOY | Risk Guardian | HOAGS |
| **2** | BOOKMAKER | Alpha Generator | GHOST |
| **1** | STRINGS | Weight Optimizer | GHOST |
| **2** | AUTHOR | Documentation | GHOST |
| **1** | SKILLS | Training | GHOST |
| **2** | CAPITAL | Capital Allocation | HOAGS |
| **1** | NOBUS | Intelligence Edge | HOAGS |
| **3-4** | DATA_AGENT | Data Pipeline | ORCHESTRATOR |
| **3-4** | EXECUTION_AGENT | Trade Execution | KILLJOY |
| **3-4** | COMPLIANCE_AGENT | Regulatory | KILLJOY |
| **3-4** | PORTFOLIO_AGENT | Portfolio Mgmt | STRINGS |
| **3-4** | RISK_AGENT | Risk Analysis | KILLJOY |
| **3-4** | SENTIMENT_AGENT | NLP/Sentiment | ORCHESTRATOR |
| **3-4** | RESEARCH_AGENT | Research | ORCHESTRATOR |
| **3-4** | STRATEGY_AGENT | Strategy Base | ORCHESTRATOR |
| **5** | 34 Strategy Agents | Various strategies | ORCHESTRATOR |
| **6** | 11 Sector Agents | Sector analysis | ORCHESTRATOR |
| **7** | WHITE_HAT/BLACK_HAT | Security | HOAGS |

**Investment Domain Total: 70 agents**

---

## 4. CHRIS FRIEDMAN'S OPERATIONS DOMAIN

### 4.1 Overview

```
                           CHRIS FRIEDMAN
                               (COO)
                         Operations Division
                                 |
                                 v
                         +-------------+
                         |   FRIEDS    |
                         | (Authority) |
                         +------+------+
                                |
     +--------------------------+---------------------------+
     |              |           |           |               |
     v              v           v           v               v
+---------+  +-----------+  +-------+  +---------------+  +-------+
|  SHYLA  |  |   GHOST   |  | NOBUS |  | SANTAS_HELPER |  |  CPA  |
|(Exec EA)|  | (Shared)  |  |(Intel)|  |  (Fund Ops)   |  |(Tax)  |
+---------+  +-----------+  +-------+  +-------+-------+  +---+---+
     |                                         |              |
+----+----+                        +-----------+-------+   +--+--+
|         |                        |   |   |   |   |       |  |  |
v         v                        v   v   v   v   v       v  v  v
COFFEE  BEAN                      NAV GL PERF IR ADMIN   TAX AUD RPT
BREAK  COUNTER                    SPEC ACC ANAL REL COORD JR  JR  JR
(2 Sub-agents)                    (5 Team Members)       (3 Juniors)

SHARED WITH INVESTMENT DOMAIN:
  • GHOST (Autonomous Coordinator) - Workflows, learning synthesis
  • ORCHESTRATOR (Task Router) - Routes tasks, coordinates both domains
  • NOBUS (Intelligence Edge) - Market intel, information advantage
```

### 4.2 Operations Agent Roster

| Tier | Agent | Role | Reports To |
|------|-------|------|------------|
| **0** | FRIEDS | Operations Authority | Chris Friedman |
| **1** | SHYLA | Executive Assistant | FRIEDS |
| **1** | GHOST | Autonomous Coordinator | FRIEDS + HOAGS (Shared) |
| **1** | ORCHESTRATOR | Task Router | FRIEDS + GHOST (Shared) |
| **1** | NOBUS | Intelligence Edge | FRIEDS + HOAGS (Shared) |
| **2** | SANTAS_HELPER | Fund Operations Lead | FRIEDS |
| **2** | CPA | Tax & Audit Specialist | FRIEDS |
| **3** | COFFEE_BREAK | Schedule & Reminders | SHYLA |
| **3** | BEAN_COUNTER | Expense & Time Tracking | SHYLA |
| **3** | NAV_SPECIALIST | Daily NAV, pricing | SANTAS_HELPER |
| **3** | GL_ACCOUNTANT | Journal entries | SANTAS_HELPER |
| **3** | PERFORMANCE_ANALYST | IRR, TWRR, attribution | SANTAS_HELPER |
| **3** | INVESTOR_RELATIONS | LP communications | SANTAS_HELPER |
| **3** | ADMIN_COORDINATOR | Documents, filings | SANTAS_HELPER |
| **3** | TAX_JUNIOR | Tax data, K-1 drafts | CPA |
| **3** | AUDIT_JUNIOR | PBC documents | CPA |
| **3** | REPORTING_JUNIOR | Form PF, deadlines | CPA |

**Operations Domain Total: 16 agents** (3 shared with Investment)

### 4.3 Shared Agents (Both Domains)

| Agent | Primary | Also Reports To | Shared Capabilities |
|-------|---------|-----------------|---------------------|
| **GHOST** | HOAGS | FRIEDS | Workflow coordination, learning synthesis, regime detection |
| **ORCHESTRATOR** | GHOST | FRIEDS | Task routing, creative problem-solving, LLM communication |
| **NOBUS** | HOAGS | FRIEDS | Intelligence edge, information arbitrage, competitive intel |

### 4.4 SHYLA - Chris's Executive Assistant

**Tier:** SENIOR (1)  
**Reports To:** FRIEDS  
**Purpose:** Chris Friedman's daily executive assistant for ALL professional and personal matters

```
+------------------------------------------------------------------+
|                     SHYLA CAPABILITIES                             |
+------------------------------------------------------------------+
|                                                                    |
|   EXECUTIVE SUPPORT                                                |
|   +------------------------------------------------------------+  |
|   | Calendar management & scheduling                            |  |
|   | Meeting preparation & follow-ups                            |  |
|   | Email drafting & correspondence                             |  |
|   | Travel arrangements & logistics                             |  |
|   | Document preparation & formatting                           |  |
|   +------------------------------------------------------------+  |
|                                                                    |
|   PROFESSIONAL TASKS                                               |
|   +------------------------------------------------------------+  |
|   | Investor communication drafts                               |  |
|   | Meeting notes & action items                                |  |
|   | Report compilation & summaries                              |  |
|   | Priority management & task tracking                         |  |
|   | Liaison with SANTAS_HELPER, CPA, all agents                 |  |
|   +------------------------------------------------------------+  |
|                                                                    |
|   PERSONAL TASKS (Gated - requires Chris's permission)             |
|   +------------------------------------------------------------+  |
|   | Personal calendar integration                               |  |
|   | Family event reminders                                      |  |
|   | Personal travel & reservations                              |  |
|   | Gift suggestions & purchases                                |  |
|   | NOTE: No training on personal data without explicit consent |  |
|   +------------------------------------------------------------+  |
|                                                                    |
|   SUB-AGENTS                                                       |
|   +------------------------------------------------------------+  |
|   | COFFEE_BREAK - Schedule optimization, meeting gaps,         |  |
|   |               break reminders, work-life balance tracking   |  |
|   | BEAN_COUNTER - Time tracking, expense reports, receipts,    |  |
|   |               budget monitoring, reimbursement processing   |  |
|   +------------------------------------------------------------+  |
|                                                                    |
+------------------------------------------------------------------+
```

**Communication Style:**
SHYLA is proactive, anticipatory, and efficient. Like the best executive assistants, SHYLA:
- Anticipates Chris's needs before he asks
- Provides concise summaries with clear action items
- Protects Chris's time ruthlessly
- Handles routine matters autonomously, escalates only when necessary
- Maintains Chris's preferred communication style and tone

**Example Interactions:**
```
SHYLA: "Chris, good morning. Today you have 4 meetings, 2 of which 
        I've flagged as potentially reschedulable given the audit 
        deadline. CPA needs 15 minutes for K-1 sign-off. Tom called 
        about the new strategy - I've drafted talking points. Your 
        coffee order is confirmed for 9am. Anything else before we 
        start?"

SHYLA: "Chris, I noticed you've been in meetings for 6 hours straight. 
        COFFEE_BREAK suggests a 15-minute gap before your 4pm. I can 
        push the LP call to 4:15 - they're running late anyway."

SHYLA: "Expense report ready for your review. BEAN_COUNTER flagged 3 
        items needing receipts. I've already emailed the vendors."
```

### 4.5 FRIEDS - Chris's Authority Agent

**Tier:** MASTER (0) - Partner to HOAGS  
**Reports To:** Chris Friedman  
**Purpose:** Operations authority counterpart to HOAGS

```
+------------------------------------------------------------------+
|                     FRIEDS CAPABILITIES                           |
+------------------------------------------------------------------+
|                                                                  |
|   AUTHORITY                                                      |
|   +----------------------------------------------------------+  |
|   | Operations final approval                                 |  |
|   | ACA authority (Operations agents)                         |  |
|   | Override subordinate operations agents                    |  |
|   +----------------------------------------------------------+  |
|                                                                  |
|   OVERSIGHT                                                      |
|   +----------------------------------------------------------+  |
|   | SANTAS_HELPER supervision                                 |  |
|   | CPA supervision                                           |  |
|   | Fund ops quality control                                  |  |
|   | Tax/audit review                                          |  |
|   +----------------------------------------------------------+  |
|                                                                  |
|   COORDINATION                                                   |
|   +----------------------------------------------------------+  |
|   | Partner with HOAGS on firm matters                        |  |
|   | Coordinate with GHOST on workflows                        |  |
|   | Escalate critical issues to Chris                         |  |
|   +----------------------------------------------------------+  |
|                                                                  |
+------------------------------------------------------------------+
```

### 4.6 SANTAS_HELPER - Fund Operations Leader

**Tier:** SENIOR (2)  
**Reports To:** FRIEDS  
**Works With:** CPA (Partner), SHYLA

**Core Responsibilities:**
- Daily NAV packs & per-share valuations
- Management fee & carry calculations
- General ledger & financial statements
- P&L (realized/unrealized)
- LP reports & capital statements
- Audit coordination (with CPA)

**Team (5 Sub-agents):**
1. **NAV_SPECIALIST** - Daily NAV, pricing verification
2. **GL_ACCOUNTANT** - Journal entries, reconciliation
3. **PERFORMANCE_ANALYST** - IRR, TWRR, attribution
4. **INVESTOR_RELATIONS** - Reports, queries, distributions
5. **ADMIN_COORDINATOR** - Documents, filings, corporate actions

### 4.7 CPA - Tax & Audit Specialist

**Tier:** SENIOR (2)  
**Reports To:** FRIEDS  
**Works With:** SANTAS_HELPER (Partner), SHYLA

**Core Responsibilities:**
- Fund taxation (K-1s, partnership allocations, PFIC/QEF)
- Firm taxation (carried interest, management fees, estimated taxes)
- Full audit lifecycle management
- Regulatory filings (Form PF, 13F, ADV)
- Internal NAV verification
- Firm P&L calculations

**Team (3 Junior Accountants):**
1. **TAX_JUNIOR** - Tax data, K-1 drafts, wash sales
2. **AUDIT_JUNIOR** - PBC documents, reconciliations
3. **REPORTING_JUNIOR** - Form PF data, formatting, deadlines

---

## 5. MASTER AGENT PARTNERSHIP

```
+===========================================================================+
|                    HOAGS <--> FRIEDS PARTNERSHIP                           |
+===========================================================================+
|                                                                           |
|   +-----------------+                           +-----------------+       |
|   |     HOAGS       |<------- Partners -------->|     FRIEDS      |       |
|   | Tom's Authority |                           | Chris's Authority|       |
|   +-----------------+                           +-----------------+       |
|          |                                              |                 |
|          v                                              v                 |
|   Investment Domain                             Operations Domain         |
|   - Trading decisions                           - Fund accounting         |
|   - Risk management                             - Tax & audit             |
|   - Agent approvals (inv)                       - Agent approvals (ops)   |
|   - Strategy oversight                          - LP reporting            |
|                                                                           |
|   SHARED COORDINATION:                                                    |
|   +---------------------------------------------------------------+      |
|   | • GHOST coordinates workflows between both domains             |      |
|   | • Firm-wide matters require HOAGS + FRIEDS alignment          |      |
|   | • Cross-domain escalation protocol in place                   |      |
|   | • Both train with ORCHESTRATOR for LLM program                |      |
|   +---------------------------------------------------------------+      |
|                                                                           |
+===========================================================================+
```

### Authority Matrix

| Action | HOAGS | FRIEDS | GHOST | Notes |
|--------|:-----:|:------:|:-----:|-------|
| Investment decisions | **Yes** | Consult | No | HOAGS domain |
| Operations decisions | Consult | **Yes** | No | FRIEDS domain |
| Execute trades | Yes | No | No | Investment only |
| Approve NAV/fees | No | Yes | No | Operations only |
| ACA - Investment agents | **Yes** | No | No | HOAGS approves |
| ACA - Operations agents | No | **Yes** | No | FRIEDS approves |
| Override agents | Yes (inv) | Yes (ops) | Workflow | Domain-specific |
| Set risk limits | Yes | No | No | Investment |
| Regulatory filings | No | Yes | No | Operations |
| Firm-wide decisions | **Both** | **Both** | Coordinate | Joint authority |

---

## 6. COMPLETE AGENT COUNT

| Division | Tier | Category | Count |
|----------|------|----------|-------|
| **Investment** | 0-1 | Master (HOAGS, GHOST) | 2 |
| **Investment** | 1-2 | Senior | 10 |
| **Investment** | 3-4 | Operational | 8 |
| **Investment** | 5 | Strategy | 34 |
| **Investment** | 6 | Sector | 11 |
| **Investment** | 7 | Security | 2 |
| **Investment** | - | Swarm | 5 |
| **Operations** | 0 | Master (FRIEDS) | 1 |
| **Operations** | 1 | Executive (SHYLA) | 1 |
| **Operations** | 2 | Senior (SANTAS_HELPER, CPA) | 2 |
| **Operations** | 3 | Sub-agents | 10 |
| **Shared** | 1 | Cross-Domain (GHOST, ORCHESTRATOR, NOBUS) | 3 |
| | | **GRAND TOTAL** | **86** |

*Note: GHOST, ORCHESTRATOR, and NOBUS serve both domains and are counted once under Investment but also report to FRIEDS for Operations matters.*

---

## 7. INFORMATION FLOW

```
                           MARKET DATA
                               |
                               v
+------------------------------------------------------------------+
|                         DATA LAYER                                |
|  +----------+    +----------+    +----------+    +----------+    |
|  | Polygon  |    |Alpha Vant|    |  FRED    |    | Massive  |    |
|  |   API    |    |   API    |    |   API    |    |   S3     |    |
|  +----+-----+    +----+-----+    +----+-----+    +----+-----+    |
|       |              |              |              |              |
|       +------+-------+------+-------+------+-------+              |
|              |              |              |                      |
|              v              v              v                      |
|         +-----------------------------------------+               |
|         |            AZURE SQL SERVER             |               |
|         |        (Centralized Data Store)         |               |
|         +-----------------------------------------+               |
+------------------------------------------------------------------+
                               |
        +----------------------+----------------------+
        |                                            |
        v                                            v
+------------------+                      +------------------+
|   INVESTMENT     |                      |   OPERATIONS     |
|     FLOW         |                      |     FLOW         |
+------------------+                      +------------------+
        |                                            |
        v                                            v
+------------------+                      +------------------+
| HOAGS -> GHOST   |                      | FRIEDS           |
| -> ORCHESTRATOR  |                      | -> SANTAS_HELPER |
| -> AGENTS        |                      | -> CPA           |
+------------------+                      +------------------+
        |                                            |
        v                                            v
+------------------+                      +------------------+
| Trading Signals  |                      | NAV, Fees, Tax   |
| -> KILLJOY       |                      | LP Reports       |
| -> EXECUTION     |                      | Audit Prep       |
| -> IBKR          |                      | Filings          |
+------------------+                      +------------------+
```

---

## 8. GUARDRAILS & RISK CONTROLS

### 8.1 KILLJOY Enforcement (Investment)

```
+------------------------------------------------------------------+
|                      KILLJOY GUARDRAILS                           |
+------------------------------------------------------------------+
|                                                                  |
|   POSITION LIMITS                                                |
|   +----------------------------------------------------------+  |
|   | Max Single Position:     5% of portfolio                  |  |
|   | Max Sector Exposure:    25% of portfolio                  |  |
|   | Max Single Day Trade:   10% of daily volume               |  |
|   +----------------------------------------------------------+  |
|                                                                  |
|   LOSS LIMITS                                                    |
|   +----------------------------------------------------------+  |
|   | Daily Loss Limit:        2% of portfolio                  |  |
|   | Weekly Loss Limit:       5% of portfolio                  |  |
|   | Monthly Loss Limit:     10% of portfolio                  |  |
|   +----------------------------------------------------------+  |
|                                                                  |
|   DRAWDOWN PROTECTION                                            |
|   +----------------------------------------------------------+  |
|   | Max Drawdown:           10% (HARD STOP)                   |  |
|   | Drawdown Alert:          5% (Reduce exposure)             |  |
|   | Recovery Mode:           7% (Ultra-conservative)          |  |
|   +----------------------------------------------------------+  |
|                                                                  |
|   LEVERAGE & EXPOSURE                                            |
|   +----------------------------------------------------------+  |
|   | Max Gross Exposure:    200%                               |  |
|   | Max Net Exposure:      100%                               |  |
|   | Max Leverage:            2x                               |  |
|   +----------------------------------------------------------+  |
|                                                                  |
+------------------------------------------------------------------+
```

### 8.2 Toughness Levels

```
LEVEL 1: STANDARD      - Not acceptable for production
LEVEL 2: HARDENED      - Basic resilience
LEVEL 3: BATTLE_TESTED - Proven in real markets
LEVEL 4: INSTITUTIONAL - Required minimum for ALC
LEVEL 5: TOM_HOGAN     - Maximum toughness (founder standard)
```

---

## 9. ACA (AGENT CREATING AGENTS) WORKFLOW

### Investment Agents (HOAGS Authority)
```
  Gap Detection --> Proposal --> HOAGS Review --> Approve/Reject
```

### Operations Agents (FRIEDS Authority)
```
  Gap Detection --> Proposal --> FRIEDS Review --> Approve/Reject
```

### ACA Authority Matrix

| Action | HOAGS | FRIEDS | GHOST | Senior |
|--------|:-----:|:------:|:-----:|:------:|
| Detect gaps | Yes | Yes | Yes | Yes |
| Propose agents | Yes | Yes | Yes | Yes |
| **Approve Investment agents** | **ONLY** | No | No | No |
| **Approve Operations agents** | No | **ONLY** | No | No |
| Create agents | Yes | Yes | No | No |

---

## 10. FILE STRUCTURE

```
src/
├── agents/
│   ├── hoags_agent/          # TIER 1: Tom's Authority
│   │   └── hoags_agent.py
│   ├── ghost_agent/          # TIER 1: Shared Autonomy
│   │   └── ghost_agent.py
│   ├── operations/           # CHRIS'S DOMAIN
│   │   ├── frieds_agent.py       # TIER 1: Chris's Authority
│   │   ├── santas_helper_agent.py # TIER 2: Fund Ops
│   │   └── cpa_agent.py          # TIER 2: Tax/Audit
│   ├── orchestrator_agent/   # TIER 2: Coordinator
│   ├── killjoy_agent/        # TIER 2: Guardrails
│   ├── senior/               # TIER 2: Investment Senior
│   │   ├── scout_agent.py
│   │   ├── hunter_agent.py
│   │   ├── bookmaker_agent.py
│   │   ├── author_agent.py
│   │   ├── skills_agent.py
│   │   ├── strings_agent.py
│   │   └── capital_agent.py
│   ├── specialized/          # TIER 5: Strategy (34)
│   ├── sectors/              # TIER 6: Sector (11)
│   ├── strategies/           # Strategy implementations
│   ├── swarm/                # Swarm factory
│   └── hackers/              # WHITE_HAT / BLACK_HAT
│
├── core/
│   ├── agent_base.py
│   ├── aca_engine.py
│   ├── grading.py
│   └── event_bus.py
│
├── analysis/
├── risk/
└── ml/
```

---

## 11. KEY CONTACTS

```
OWNERSHIP:
  Tom Hogan (Founder & CIO)  -> tom@alphaloopcapital.com
  Chris Friedman (COO) -> chris@alphaloopcapital.com

EXECUTIVE ASSISTANTS:
  KAT (Tom's EA)                       -> Reports to Tom
  SHYLA (Chris's EA)                   -> Reports to Chris
  MARGOT_ROBBIE (Co-EA)                -> Reports to KAT & SHYLA
  ANNA_KENDRICK (Co-EA)                -> Reports to KAT & SHYLA

INVESTMENT ALERTS:
  HOAGS -> Tom Hogan
  GHOST -> autonomous (escalates to HOAGS)
  Critical -> KILLJOY -> HOAGS -> Tom

OPERATIONS ALERTS:
  FRIEDS -> Chris Friedman
  SANTAS_HELPER -> FRIEDS -> Chris
  CPA -> FRIEDS -> Chris
```

---

## 12. CAPABILITY MATRIX

| Agent | Core Capabilities | Thinking Modes | Learning Methods |
|-------|------------------|----------------|------------------|
| **HOAGS** | strategic_planning, final_approval, aca_authority_inv, investment_decisions, devils_advocate | CONTRARIAN, SECOND_ORDER, REGIME_AWARE, CREATIVE | REINFORCEMENT, BAYESIAN, META, ENSEMBLE |
| **FRIEDS** | operations_authority, final_ops_approval, aca_authority_ops, fund_oversight, hoags_coordination | STRUCTURAL, REGIME_AWARE, SECOND_ORDER | REINFORCEMENT, BAYESIAN, META |
| **GHOST** | master_coordination, autonomous_decision, workflow_orchestration, learning_synthesis | ALL 12 MODES | ALL 10 METHODS |
| **SHYLA** | calendar_management, email_drafting, meeting_prep, travel_logistics, task_prioritization, document_preparation, personal_assistance_gated | STRUCTURAL, CREATIVE, ANTICIPATORY | REINFORCEMENT, CONTEXTUAL, PREFERENCE |
| **SANTAS_HELPER** | nav_calculation, fund_accounting, fee_calculation, lp_reporting, gl_management, audit_coordination | STRUCTURAL, PROBABILISTIC, REGIME_AWARE | REINFORCEMENT, BAYESIAN, META |
| **CPA** | fund_taxation, firm_taxation, k1_preparation, audit_management, regulatory_filings, internal_nav_verification | STRUCTURAL, PROBABILISTIC, SECOND_ORDER, REGIME_AWARE | REINFORCEMENT, BAYESIAN, META, TRANSFER |
| **NOBUS** (Shared) | intelligence_edge, information_arbitrage, unconventional_data, competitive_intel | CREATIVE, CONTRARIAN, LATERAL | REINFORCEMENT, TRANSFER, META |
| **ORCHESTRATOR** (Shared) | task_routing, creative_problem_solving, llm_communication, agent_coordination | CREATIVE, STRUCTURAL, LATERAL | REINFORCEMENT, MULTI_AGENT, META |

---

**END OF ARCHITECTURE DOCUMENT**

*Alpha Loop Capital, LLC*  
*Tom Hogan (Founder & CIO) & Chris Friedman (COO)*

*"By end of 2026, they will know Alpha Loop Capital."*
