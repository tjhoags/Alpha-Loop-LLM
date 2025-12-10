# ALPHA LOOP CAPITAL - AGENT ARCHITECTURE

## Complete System Documentation
**Author:** Tom Hogan | Alpha Loop Capital, LLC  
**Version:** 2.0 | December 2024

---

## 1. HIERARCHY DIAGRAM

```
                              +------------------+
                              |     TOM HOGAN    |
                              |   (Human Owner)  |
                              +--------+---------+
                                       |
                                       v
         +------------------------------------------------------------+
         |                        TIER 1: MASTERS                      |
         +------------------------------------------------------------+
         |                                                            |
         |    +-------------+                    +-------------+      |
         |    |    HOAGS    |<------------------>|    GHOST    |      |
         |    | (Authority) |    Coordination    | (Autonomy)  |      |
         |    +------+------+                    +------+------+      |
         |           |                                  |             |
         +-----------|----------------------------------|-------------+
                     |                                  |
                     v                                  v
         +------------------------------------------------------------+
         |                      TIER 2: SENIOR AGENTS                  |
         +------------------------------------------------------------+
         |                                                            |
         |  +--------+  +--------+  +------------+  +---------+       |
         |  | SCOUT  |  | HUNTER |  |ORCHESTRATOR|  | KILLJOY |       |
         |  |Arbitrage|  |AlgoHunt|  |Coordinator |  |Guardrail|       |
         |  +--------+  +----+---+  +------------+  +---------+       |
         |                   |                                        |
         |  +--------+  +----v---+  +--------+  +--------+            |
         |  |BOOKMAKER|  | GHOST  |  | SKILLS |  | AUTHOR |           |
         |  |  Alpha  |  |Pathways|  |Training|  |  Docs  |           |
         |  +--------+  +--------+  +--------+  +--------+            |
         |                                                            |
         |  +--------+                                                |
         |  | STRINGS|                                                |
         |  |Weights |                                                |
         |  +--------+                                                |
         +------------------------------------------------------------+
                     |
                     v
         +------------------------------------------------------------+
         |                   TIER 3-4: OPERATIONAL AGENTS              |
         +------------------------------------------------------------+
         |                                                            |
         |  +----------+  +-----------+  +------------+  +----------+ |
         |  |   DATA   |  | EXECUTION |  | COMPLIANCE |  | RESEARCH | |
         |  |  Agent   |  |   Agent   |  |   Agent    |  |  Agent   | |
         |  +----------+  +-----------+  +------------+  +----------+ |
         |                                                            |
         |  +----------+  +-----------+  +------------+               |
         |  |PORTFOLIO |  |   RISK    |  | SENTIMENT  |               |
         |  |  Agent   |  |   Agent   |  |   Agent    |               |
         |  +----------+  +-----------+  +------------+               |
         +------------------------------------------------------------+
                     |
                     v
         +------------------------------------------------------------+
         |                 TIER 5: STRATEGY AGENTS (34)                |
         +------------------------------------------------------------+
         |                                                            |
         |  +----------+ +----------+ +----------+ +----------+       |
         |  | MOMENTUM | |  VALUE   |  | GROWTH  | |  CRYPTO  |       |
         |  +----------+ +----------+ +----------+ +----------+       |
         |                                                            |
         |  +----------+ +----------+ +----------+ +----------+       |
         |  | OPTIONS  | |  FOREX   | | FUTURES  | |   MACRO  |       |
         |  +----------+ +----------+ +----------+ +----------+       |
         |                                                            |
         |  +----------+ +----------+ +----------+ +----------+       |
         |  |MEAN_REV  | | BREAKOUT | |  PAIRS   | | DIVIDEND |       |
         |  +----------+ +----------+ +----------+ +----------+       |
         |                                                            |
         |  ... and 22 more specialized strategy agents               |
         +------------------------------------------------------------+
                     |
                     v
         +------------------------------------------------------------+
         |                   TIER 6: SECTOR AGENTS (11)                |
         +------------------------------------------------------------+
         |                                                            |
         |  +------+ +------+ +------+ +------+ +------+ +------+     |
         |  | TECH | |HEALTH| |ENERGY| |FINANC| |INDUST| |MATER |     |
         |  +------+ +------+ +------+ +------+ +------+ +------+     |
         |                                                            |
         |  +------+ +------+ +------+ +------+ +------+              |
         |  |CONSUM| |CONSUM| |REAL  | | UTIL | |COMMUN|              |
         |  | DISC | |STAPLE| |ESTATE| |      | |SERVIC|              |
         |  +------+ +------+ +------+ +------+ +------+              |
         +------------------------------------------------------------+
```

---

## 2. INFORMATION FLOW DIAGRAM

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
                               v
+------------------------------------------------------------------+
|                       PROCESSING LAYER                            |
|                                                                  |
|    +------------------+          +------------------+            |
|    | Feature Engineer |          |  ML Training     |            |
|    | (100+ features)  |          | (XGB/LGBM/CAT)   |            |
|    +--------+---------+          +--------+---------+            |
|             |                             |                      |
|             v                             v                      |
|    +------------------+          +------------------+            |
|    | Valuation Suite  |          |  Quant Metrics   |            |
|    | (DCF, Hogan, LBO)|          | (VaR, Greeks)    |            |
|    +--------+---------+          +--------+---------+            |
|             |                             |                      |
|             +-------------+---------------+                      |
|                           |                                      |
+------------------------------------------------------------------+
                            |
                            v
+------------------------------------------------------------------+
|                        AGENT LAYER                                |
|                                                                  |
|   +--------+     +--------+     +--------+     +--------+        |
|   | HOAGS  |---->| GHOST  |---->|ORCHESTR|---->| AGENTS |        |
|   |Approve |     |Coordinat|     | Route  |     |Execute |        |
|   +--------+     +--------+     +--------+     +--------+        |
|                       |                            |             |
|                       v                            v             |
|              +------------------+        +------------------+    |
|              |  ACA Engine      |        | Signal Generator |    |
|              | (Create Agents)  |        | (Trading Signals)|    |
|              +------------------+        +------------------+    |
+------------------------------------------------------------------+
                            |
                            v
+------------------------------------------------------------------+
|                       EXECUTION LAYER                             |
|                                                                  |
|   +------------+     +------------+     +------------+           |
|   |  KILLJOY   |---->| Execution  |---->|    IBKR    |           |
|   | (Guardrails)|     |   Engine   |     |   (Broker) |           |
|   +------------+     +------------+     +------------+           |
|                                                                  |
|   Risk Checks:                                                   |
|   - Max Position Size: 5%                                        |
|   - Daily Loss Limit: 2%                                         |
|   - Max Drawdown: 10%                                            |
|   - Gross Exposure: 200%                                         |
+------------------------------------------------------------------+
```

---

## 3. AGENT RELATIONSHIPS MATRIX

```
                HOAGS  GHOST  SCOUT  HUNTER  ORCH  KILLJOY  BOOK  STRINGS
    HOAGS         -    <->    <-      <-     <-      <-      <-     <-
    GHOST       <->     -     ->      ->     ->      ->      ->     ->
    SCOUT        ->    <-      -      <>      -      <-       -      -
    HUNTER       ->    <->    <>       -     <>       -      <>      -
    ORCH         ->    <-      -      <>      -       -      <>     <>
    KILLJOY      ->    <-     ->       -      -       -      ->     ->
    BOOK         ->    <-      -      <>     <>      <-       -     <>
    STRINGS      ->    <-      -       -     <>      <-      <>      -

    LEGEND:
    ->  : Reports to / Sends signals
    <-  : Receives from / Gets signals  
    <-> : Bidirectional coordination
    <>  : Collaboration / Sharing
    -   : No direct relationship
```

---

## 4. CAPABILITY MATRIX

| Agent | Core Capabilities | Thinking Modes | Learning Methods |
|-------|------------------|----------------|------------------|
| **HOAGS** | strategic_planning, final_approval, aca_authority, investment_decisions, devils_advocate | CONTRARIAN, SECOND_ORDER, REGIME_AWARE, CREATIVE | REINFORCEMENT, BAYESIAN, META, ENSEMBLE |
| **GHOST** | master_coordination, autonomous_decision, workflow_orchestration, learning_synthesis | ALL 12 MODES | ALL 10 METHODS |
| **SCOUT** | bid_ask_analysis, options_mispricing, scalp_optimization, retail_flow_detection | BEHAVIORAL, STRUCTURAL, INFORMATION_EDGE | REINFORCEMENT, ACTIVE |
| **HUNTER** | algorithm_detection, counter_strategy, ghost_pathway_creation, flow_analysis | ADVERSARIAL, GAME_THEORETIC, ABSENCE | ADVERSARIAL, META |
| **ORCHESTRATOR** | task_routing, creative_thinking, agent_improvement, psychological_analysis | CREATIVE, BEHAVIORAL, STRUCTURAL | META, TRANSFER |
| **KILLJOY** | capital_allocation, heat_control, drawdown_watch | PROBABILISTIC, REGIME_AWARE | BAYESIAN |
| **BOOKMAKER** | alpha_generation, odds_calculation, edge_identification | PROBABILISTIC, INFORMATION_EDGE | ENSEMBLE, BAYESIAN |
| **STRINGS** | weight_optimization, portfolio_construction, rebalancing | STRUCTURAL, PROBABILISTIC | ENSEMBLE, EVOLUTIONARY |

---

## 5. GUARDRAILS & RISK CONTROLS

### 5.1 KILLJOY Enforcement

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

### 5.2 Agent Self-Guardrails

```python
# Every agent has built-in battle stats:
_crashes_survived: int = 0              # System resilience
_drawdowns_navigated: int = 0           # Risk experience
_regime_changes_adapted: int = 0        # Adaptation count
_black_swans_handled: int = 0           # Extreme events
_consecutive_failures_without_tilt: int = 0  # Emotional stability
_max_stress_handled: float = 0.0        # Stress capacity
_recovery_speed: float = 1.0            # Recovery ability
```

### 5.3 Toughness Levels

```
LEVEL 1: STANDARD      - Not acceptable for production
LEVEL 2: HARDENED      - Basic resilience
LEVEL 3: BATTLE_TESTED - Proven in real markets
LEVEL 4: INSTITUTIONAL - Required minimum for ALC
LEVEL 5: TOM_HOGAN     - Maximum toughness (founder standard)
```

---

## 6. ACA (AGENT CREATING AGENTS) WORKFLOW

```
+------------------------------------------------------------------+
|                      ACA WORKFLOW DIAGRAM                         |
+------------------------------------------------------------------+

  Step 1: Gap Detection
  +-----------------+
  | Any Agent       |
  | Detects Gap     |----> detect_capability_gap()
  +-----------------+
           |
           v
  Step 2: Proposal Creation
  +-----------------+
  | Agent Creates   |
  | Proposal        |----> propose_agent()
  +-----------------+
           |
           v
  Step 3: ACA Engine Registration
  +-----------------+
  | ACA Engine      |
  | Registers Gap   |----> submit_proposal()
  +-----------------+
           |
           v
  Step 4: HOAGS Review
  +-----------------+
  | HOAGS Reviews   |
  | All Proposals   |----> _review_aca_proposals()
  +-----------------+
           |
     +-----+-----+
     |           |
     v           v
  APPROVE     REJECT
     |           |
     v           v
  Step 5a:    Step 5b:
  +---------+ +---------+
  | Create  | | Log     |
  | Agent   | | Reason  |
  +---------+ +---------+
```

### ACA Authority Matrix

| Action | HOAGS | GHOST | SENIOR | STANDARD | STRATEGY |
|--------|-------|-------|--------|----------|----------|
| Detect gaps | Yes | Yes | Yes | Yes | Yes |
| Propose agents | Yes | Yes | Yes | Limited | No |
| Review proposals | Yes | No | No | No | No |
| **Approve agents** | **ONLY** | No | No | No | No |
| Create agents | Yes | No | No | No | No |
| Reject proposals | Yes | No | No | No | No |

---

## 7. LEARNING & ADAPTATION

### 7.1 Flywheel Effect

```
                    +----------------+
                    |  Agent Learns  |
                    |  from Trade    |
                    +-------+--------+
                            |
                            v
                    +----------------+
                    | Learning Saved |
                    | to Azure Blob  |
                    +-------+--------+
                            |
            +---------------+---------------+
            |               |               |
            v               v               v
     +-----------+   +-----------+   +-----------+
     |  LENOVO   |   |   MAC     |   |  OTHER    |
     |  Machine  |   |  Machine  |   |  Machine  |
     +-----------+   +-----------+   +-----------+
            |               |               |
            +---------------+---------------+
                            |
                            v
                    +----------------+
                    |  GHOST Merges  |
                    | All Learnings  |
                    +-------+--------+
                            |
                            v
                    +----------------+
                    | Synthesized    |
                    | Intelligence   |
                    +----------------+
                            |
            +---------------+---------------+
            |               |               |
            v               v               v
     +-----------+   +-----------+   +-----------+
     | Improves  |   | Improves  |   | Improves  |
     | Agent A   |   | Agent B   |   | Agent C   |
     +-----------+   +-----------+   +-----------+

    >>> FLYWHEEL: Each learning improves ALL agents <<<
```

### 7.2 Confidence Calibration

```
Raw Prediction Confidence
         |
         v
    +----+----+
    | History |  <-- Track (confidence, was_correct) pairs
    +---------+
         |
         v
    +----+----+
    | Bucket  |  <-- Group by confidence range (0-10%, 10-20%, etc.)
    +---------+
         |
         v
    +----+----+
    | Compare |  <-- Expected vs Actual accuracy per bucket
    +---------+
         |
         v
    +----+----+
    | Adjust  |  <-- _confidence_adjustment (0.5 to 1.5)
    +---------+
         |
         v
    Calibrated Confidence = Raw * _confidence_adjustment
```

---

## 8. GRADING SYSTEM

### 8.1 Thresholds

| Metric | Minimum Required | Grade Impact |
|--------|------------------|--------------|
| Success Rate | 80% | Major |
| Executions | 10 | Major |
| Capabilities | 3 | Medium |
| Learning Events | 10 | Medium |
| Battle Stats | >0 | Minor |

### 8.2 Grade Mapping

```
Grade A: All thresholds met        -> PROMOTED TO PRODUCTION
Grade B: 1-2 issues                -> PROMOTED WITH MONITORING
Grade C: 3-4 issues                -> CONTINUE TRAINING
Grade D: 5+ issues                 -> NOT PROMOTED (needs work)
```

### 8.3 Promotion Flow

```
    TRAINING
        |
        v
    +--------+
    | Grade  |
    | Check  |
    +---+----+
        |
   +----+----+----+
   |    |    |    |
   v    v    v    v
   A    B    C    D
   |    |    |    |
   v    v    |    |
PROD  PROD  |    |
      +MON  v    v
         TRAIN  TRAIN
         MORE   MORE
```

---

## 9. FILE STRUCTURE

```
src/
+-- agents/
|   +-- hoags_agent/          # TIER 1: Supreme Authority
|   |   +-- hoags_agent.py
|   +-- ghost_agent/          # TIER 1: Autonomous Controller
|   |   +-- ghost_agent.py
|   +-- orchestrator_agent/   # TIER 2: Coordinator
|   +-- killjoy_agent/        # TIER 2: Guardrails
|   +-- senior/               # TIER 2: Senior Agents
|   |   +-- scout_agent.py
|   |   +-- hunter_agent.py
|   |   +-- bookmaker_agent.py
|   |   +-- author_agent.py
|   |   +-- skills_agent.py
|   |   +-- strings_agent.py
|   +-- specialized/          # TIER 5: Strategy Agents (34)
|   |   +-- momentum_agent.py
|   |   +-- value_agent.py
|   |   +-- growth_agent.py
|   |   +-- crypto_agent.py
|   |   +-- options_agent.py
|   |   +-- ... (29 more)
|   +-- sectors/              # TIER 6: Sector Agents (11)
|   +-- strategies/           # Strategy implementations
|   +-- swarm/                # Swarm factory
|   +-- hackers/              # White hat / Black hat
|
+-- core/
|   +-- agent_base.py         # BaseAgent class (1,278 lines)
|   +-- aca_engine.py         # ACA system (445 lines)
|   +-- grading.py            # Agent grading
|   +-- event_bus.py          # Inter-agent communication
|   +-- learning_optimizer.py # Learning coordination
|
+-- analysis/
|   +-- valuation_suite.py    # DCF, Hogan Model, LBO, M&A
|   +-- behavioral_finance.py # Psychology, Game Theory
|
+-- risk/
|   +-- quant_metrics.py      # VaR, Greeks, Duration
|   +-- risk_manager.py       # Risk controls
|
+-- ml/
|   +-- advanced_training.py  # Parallel ML training
|   +-- feature_engineering.py # 100+ features
```

---

## 10. QUICK REFERENCE

### Commands

```bash
# Start training (Windows)
.\venv\Scripts\activate
python -c "from src.ml.advanced_training import run_overnight_training; run_overnight_training()"

# Check grades
python scripts/model_dashboard.py

# Start paper trading
python src/trading/production_algo.py --mode paper
```

### Key Contacts

```
Agent Communication:
  HOAGS -> tom@alphaloopcapital.com
  GHOST -> autonomous (no human contact)
  All alerts -> KILLJOY -> HOAGS
```

---

**END OF ARCHITECTURE DOCUMENT**

*"By end of 2026, they will know Alpha Loop Capital."*

