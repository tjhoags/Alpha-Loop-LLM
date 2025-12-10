# AGENT RELATIONSHIPS - QUICK REFERENCE

## WHO REPORTS TO WHOM

```
                            TOM HOGAN
                                |
                    +-----------+-----------+
                    |                       |
                 HOAGS                   GHOST
              (Authority)             (Autonomy)
                    |                       |
        +-----------+-----------+-----------+-----------+
        |           |           |           |           |
     SCOUT      HUNTER    ORCHESTR     KILLJOY    BOOKMAKER
        |           |           |           |           |
        +-----------+-----------+-----------+-----------+
                            |
            +---------------+---------------+
            |               |               |
        STRATEGY        SECTOR          SWARM
        AGENTS          AGENTS         AGENTS
         (34)            (11)           (5)
```

## KEY RELATIONSHIPS

### HOAGS (The Boss)
```
HOAGS
  |
  +---> Approves all ACA proposals
  +---> Reviews KILLJOY alerts
  +---> Overrides any agent decision
  +---> Final authority on all trades
  |
  +<--- Receives from GHOST: Daily summaries
  +<--- Receives from KILLJOY: Risk alerts
  +<--- Receives from SCOUT: Arbitrage opportunities
```

### GHOST (The Coordinator)
```
GHOST
  |
  +---> Coordinates all agent workflows
  +---> Synthesizes learnings (flywheel)
  +---> Detects regime changes
  +---> Routes tasks via ORCHESTRATOR
  |
  +<--- Receives from HUNTER: Pathways
  +<--- Receives from ALL: Learning data
  +<--- Reports to: HOAGS only
```

### HUNTER + GHOST (Special Relationship)
```
HUNTER                              GHOST
   |                                  |
   +--- Detects algorithm ----+       |
   |    signatures            |       |
   |                          v       |
   +--- Creates pathways ---> GHOST <-+
   |    for GHOST             |
   |                          |
   +<-- Gets enhanced --------+
        insights back
```

### KILLJOY (The Guardrail)
```
              ALL TRADE SIGNALS
                     |
                     v
              +------------+
              |  KILLJOY   |
              | (Guardian) |
              +-----+------+
                    |
        +-----------+-----------+
        |           |           |
        v           v           v
     APPROVE     REDUCE      BLOCK
        |           |           |
        v           v           v
    Execute    Scale Down    Stop
```

### SCOUT (The Hunter)
```
SCOUT
  |
  +---> Scans <$30bn market cap stocks
  +---> Finds retail bid/ask inefficiencies
  +---> Detects options mispricing
  |
  +---> IMMEDIATE alerts to HOAGS
  +---> Shares with BOOKMAKER for alpha calc
```

### ORCHESTRATOR (The Router)
```
          INCOMING TASK
                |
                v
        +---------------+
        | ORCHESTRATOR  |
        | (Task Router) |
        +-------+-------+
                |
    +-----------+-----------+-----------+
    |           |           |           |
    v           v           v           v
 SCOUT      HUNTER     STRATEGY    SECTOR
(arb)      (algo)      (trade)    (sector)
```

## COMMUNICATION PROTOCOLS

### Signal Flow
```
Market Data --> DATA_AGENT --> FEATURE_ENG --> ML_MODELS
                                                  |
                                                  v
                                          SIGNAL_GENERATOR
                                                  |
                                                  v
                                            RISK_AGENT
                                                  |
                                                  v
                                             KILLJOY
                                                  |
                              +-------------------+-------------------+
                              |                   |                   |
                              v                   v                   v
                          APPROVE             MODIFY              BLOCK
                              |                   |                   |
                              v                   v                   |
                       EXEC_AGENT          EXEC_AGENT                 X
                              |                   |
                              v                   v
                            IBKR               IBKR
```

### Learning Flow
```
Trade Outcome
      |
      v
+-----+-----+
| Agent A   |---> Learning Record ---> Azure Blob
| Learns    |                              |
+-----------+                              |
                                          |
+-----+-----+                              |
| Agent B   |---> Learning Record ------>--+
| Learns    |                              |
+-----------+                              |
                                          |
                                          v
                                    +-----+-----+
                                    |   GHOST   |
                                    |  Merges   |
                                    +-----+-----+
                                          |
                                          v
                              +------------------------+
                              | SYNTHESIZED KNOWLEDGE  |
                              +------------------------+
                                          |
                    +---------------------+---------------------+
                    |                     |                     |
                    v                     v                     v
              All Agents            All Agents            All Agents
               Improved              Improved              Improved
```

## AUTHORITY LEVELS

| Action | HOAGS | GHOST | Senior | Standard |
|--------|:-----:|:-----:|:------:|:--------:|
| Execute trades | Yes | No | No | No |
| Approve agents | **Only** | No | No | No |
| Override agents | Yes | Yes* | No | No |
| Set risk limits | Yes | No | No | No |
| Create pathways | No | Yes | Yes | No |
| Detect gaps | Yes | Yes | Yes | Yes |

*GHOST can override for workflow only, not trade decisions

## ESCALATION PATH

```
Low Priority Issue:
  Agent --> ORCHESTRATOR --> Logged

Medium Priority Issue:
  Agent --> ORCHESTRATOR --> GHOST --> Review Queue

High Priority Issue:
  Agent --> KILLJOY --> GHOST --> HOAGS Alert

Critical Issue:
  Agent --> KILLJOY --> IMMEDIATE HALT --> HOAGS --> Tom Hogan
```

## DAILY WORKFLOW

```
6:00 AM  DATA_AGENT: Pull overnight data
6:30 AM  GHOST: Regime detection
7:00 AM  STRATEGY_AGENTS: Generate signals
7:30 AM  RISK_AGENT: Risk assessment
8:00 AM  KILLJOY: Pre-market guardrail check
8:30 AM  HOAGS: Review major positions
9:00 AM  ORCHESTRATOR: Coordinate opening trades
9:30 AM  MARKET OPEN: Execute approved signals

Throughout Day:
  - SCOUT: Continuous inefficiency scan
  - HUNTER: Algorithm tracking
  - KILLJOY: Real-time risk monitoring

4:00 PM  MARKET CLOSE
4:30 PM  GHOST: Synthesize learnings
5:00 PM  ALL_AGENTS: Update models
6:00 PM  HOAGS: Daily summary review
```

---

## QUICK COMMAND REFERENCE

```bash
# Check agent status
python -c "from src.agents.hoags_agent.hoags_agent import HoagsAgent; h = HoagsAgent(); print(h.get_stats())"

# Run daily workflow
python -c "from src.agents.hoags_agent.hoags_agent import HoagsAgent; h = HoagsAgent(); h.run_daily_workflow()"

# Check GHOST status
python -c "from src.agents.ghost_agent.ghost_agent import GhostAgent; g = GhostAgent(); print(g._get_status())"

# Review ACA proposals
python -c "from src.core.aca_engine import get_aca_engine; e = get_aca_engine(); print(e.get_pending_proposals())"
```

---

*Architecture designed by Tom Hogan | Alpha Loop Capital, LLC*

