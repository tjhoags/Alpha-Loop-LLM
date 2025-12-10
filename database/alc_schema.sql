-- ============================================================================
-- ALPHA LOOP CAPITAL - MASTER DATABASE SCHEMA
-- Version: 2.0
-- Created: December 2025
-- Developer: Tom Hogan / Claude AI Assistant
-- ============================================================================
-- This schema stores ALL data for the 44-agent trading system:
-- 1. Market data (prices, fundamentals, alternative data)
-- 2. Agent signals and recommendations  
-- 3. Training data for ML models
-- 4. Performance tracking
-- 5. Flywheel knowledge sharing
-- ============================================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- SECTION 1: MARKET DATA
-- ============================================================================

-- Daily OHLCV price data
CREATE TABLE IF NOT EXISTS market_prices (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(15,4),
    high DECIMAL(15,4),
    low DECIMAL(15,4),
    close DECIMAL(15,4),
    adj_close DECIMAL(15,4),
    volume BIGINT,
    source VARCHAR(50) DEFAULT 'yahoo',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date, source)
);

-- Intraday price data (1-minute bars)
CREATE TABLE IF NOT EXISTS market_prices_intraday (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open DECIMAL(15,4),
    high DECIMAL(15,4),
    low DECIMAL(15,4),
    close DECIMAL(15,4),
    volume BIGINT,
    source VARCHAR(50) DEFAULT 'ibkr',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, source)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_prices_symbol ON market_prices(symbol);
CREATE INDEX IF NOT EXISTS idx_prices_date ON market_prices(date);
CREATE INDEX IF NOT EXISTS idx_prices_symbol_date ON market_prices(symbol, date);
CREATE INDEX IF NOT EXISTS idx_intraday_symbol ON market_prices_intraday(symbol);
CREATE INDEX IF NOT EXISTS idx_intraday_timestamp ON market_prices_intraday(timestamp);


-- ============================================================================
-- SECTION 2: FUNDAMENTALS
-- ============================================================================

-- Company fundamentals (quarterly data)
CREATE TABLE IF NOT EXISTS company_fundamentals (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    period_end DATE NOT NULL,
    period_type VARCHAR(10) DEFAULT 'quarterly',  -- quarterly or annual
    
    -- Income Statement
    revenue DECIMAL(20,2),
    gross_profit DECIMAL(20,2),
    operating_income DECIMAL(20,2),
    net_income DECIMAL(20,2),
    ebitda DECIMAL(20,2),
    eps_basic DECIMAL(10,4),
    eps_diluted DECIMAL(10,4),
    
    -- Balance Sheet
    total_assets DECIMAL(20,2),
    total_liabilities DECIMAL(20,2),
    total_equity DECIMAL(20,2),
    cash_and_equivalents DECIMAL(20,2),
    total_debt DECIMAL(20,2),
    
    -- Cash Flow
    operating_cash_flow DECIMAL(20,2),
    free_cash_flow DECIMAL(20,2),
    capex DECIMAL(20,2),
    
    -- Ratios (calculated)
    pe_ratio DECIMAL(10,2),
    pb_ratio DECIMAL(10,2),
    debt_to_equity DECIMAL(10,4),
    current_ratio DECIMAL(10,4),
    roe DECIMAL(10,4),
    roa DECIMAL(10,4),
    gross_margin DECIMAL(10,4),
    operating_margin DECIMAL(10,4),
    net_margin DECIMAL(10,4),
    
    -- Metadata
    source VARCHAR(50) DEFAULT 'yahoo',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, period_end, period_type)
);

-- Earnings events (for earnings agents)
CREATE TABLE IF NOT EXISTS earnings_events (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    earnings_date DATE NOT NULL,
    earnings_time VARCHAR(20),  -- pre-market, post-market, during
    
    -- Estimates
    eps_estimate DECIMAL(10,4),
    revenue_estimate DECIMAL(20,2),
    
    -- Actuals
    eps_actual DECIMAL(10,4),
    revenue_actual DECIMAL(20,2),
    
    -- Surprise
    eps_surprise DECIMAL(10,4),
    eps_surprise_pct DECIMAL(10,4),
    revenue_surprise DECIMAL(20,2),
    revenue_surprise_pct DECIMAL(10,4),
    
    -- Guidance
    guidance_low DECIMAL(10,4),
    guidance_high DECIMAL(10,4),
    guidance_vs_consensus DECIMAL(10,4),
    
    -- Price reaction
    price_before DECIMAL(15,4),
    price_after DECIMAL(15,4),
    price_change_pct DECIMAL(10,4),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, earnings_date)
);

CREATE INDEX IF NOT EXISTS idx_fundamentals_symbol ON company_fundamentals(symbol);
CREATE INDEX IF NOT EXISTS idx_earnings_symbol ON earnings_events(symbol);
CREATE INDEX IF NOT EXISTS idx_earnings_date ON earnings_events(earnings_date);


-- ============================================================================
-- SECTION 3: AGENT SIGNALS
-- ============================================================================

-- Every signal/recommendation from every agent
CREATE TABLE IF NOT EXISTS agent_signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_name VARCHAR(100) NOT NULL,
    agent_tier VARCHAR(20) NOT NULL,  -- master, senior, standard
    
    -- Signal details
    signal_time TIMESTAMP NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(50) NOT NULL,  -- BUY, SELL, HOLD, AVOID, etc.
    direction VARCHAR(10),  -- LONG, SHORT, NEUTRAL
    
    -- Conviction
    confidence DECIMAL(5,4) NOT NULL,  -- 0.0 to 1.0
    conviction_level VARCHAR(20),  -- HIGH, MEDIUM, LOW
    
    -- Price targets
    entry_price DECIMAL(15,4),
    target_price DECIMAL(15,4),
    stop_loss DECIMAL(15,4),
    expected_return DECIMAL(10,4),
    risk_reward_ratio DECIMAL(10,4),
    
    -- Thesis
    thesis_summary TEXT,
    thesis_full JSONB,
    key_catalysts TEXT[],
    key_risks TEXT[],
    
    -- Time horizon
    time_horizon VARCHAR(50),  -- intraday, swing, position, long_term
    expected_holding_days INTEGER,
    
    -- Source analysis
    analysis_type VARCHAR(50),  -- fundamental, technical, alternative, macro
    data_sources TEXT[],
    
    -- Outcome tracking (filled later)
    outcome_realized BOOLEAN DEFAULT FALSE,
    outcome_return DECIMAL(10,4),
    outcome_days INTEGER,
    outcome_notes TEXT,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for signal analysis
CREATE INDEX IF NOT EXISTS idx_signals_agent ON agent_signals(agent_name);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON agent_signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_time ON agent_signals(signal_time);
CREATE INDEX IF NOT EXISTS idx_signals_confidence ON agent_signals(confidence);
CREATE INDEX IF NOT EXISTS idx_signals_outcome ON agent_signals(outcome_realized);


-- ============================================================================
-- SECTION 4: TRAINING DATA
-- ============================================================================

-- Labeled training examples
CREATE TABLE IF NOT EXISTS training_examples (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_name VARCHAR(100) NOT NULL,
    
    -- Feature data
    symbol VARCHAR(20) NOT NULL,
    example_date DATE NOT NULL,
    features JSONB NOT NULL,  -- All input features as JSON
    
    -- Label
    label VARCHAR(50) NOT NULL,  -- What was the correct action
    label_numeric DECIMAL(10,4),  -- Numeric label for regression
    
    -- Forward returns (what actually happened)
    return_1d DECIMAL(10,4),
    return_5d DECIMAL(10,4),
    return_20d DECIMAL(10,4),
    return_60d DECIMAL(10,4),
    
    -- Quality
    confidence_weight DECIMAL(5,4) DEFAULT 1.0,  -- Weight for training
    is_validated BOOLEAN DEFAULT FALSE,
    validated_by VARCHAR(100),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Agent models (versions and performance)
CREATE TABLE IF NOT EXISTS agent_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    
    -- Training info
    trained_at TIMESTAMP NOT NULL,
    training_examples INTEGER,
    epochs INTEGER,
    
    -- Performance metrics
    accuracy DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    sharpe_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(8,4),
    win_rate DECIMAL(5,4),
    
    -- Backtest results
    backtest_return DECIMAL(10,4),
    backtest_volatility DECIMAL(10,4),
    backtest_trades INTEGER,
    
    -- Model storage
    model_path VARCHAR(500),  -- Path to saved model file
    model_config JSONB,  -- Model hyperparameters
    
    -- Status
    is_active BOOLEAN DEFAULT FALSE,
    deployed_at TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(agent_name, model_version)
);

CREATE INDEX IF NOT EXISTS idx_training_agent ON training_examples(agent_name);
CREATE INDEX IF NOT EXISTS idx_training_symbol ON training_examples(symbol);
CREATE INDEX IF NOT EXISTS idx_models_agent ON agent_models(agent_name);
CREATE INDEX IF NOT EXISTS idx_models_active ON agent_models(is_active);


-- ============================================================================
-- SECTION 5: PERFORMANCE TRACKING
-- ============================================================================

-- Actual trades executed
CREATE TABLE IF NOT EXISTS trade_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Trade details
    symbol VARCHAR(20) NOT NULL,
    trade_time TIMESTAMP NOT NULL,
    trade_type VARCHAR(10) NOT NULL,  -- BUY, SELL, SHORT, COVER
    quantity INTEGER NOT NULL,
    price DECIMAL(15,4) NOT NULL,
    commission DECIMAL(10,4) DEFAULT 0,
    
    -- Source
    agent_name VARCHAR(100),  -- Which agent recommended
    signal_id UUID REFERENCES agent_signals(id),
    execution_method VARCHAR(50),  -- manual, algo, market, limit
    
    -- P&L (calculated at close or exit)
    realized_pnl DECIMAL(15,4),
    unrealized_pnl DECIMAL(15,4),
    
    -- Metadata
    broker VARCHAR(50),  -- IBKR, Coinbase, etc.
    account_id VARCHAR(100),
    notes TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Daily portfolio snapshots
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL,
    snapshot_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Portfolio value
    total_value DECIMAL(20,4),
    cash_balance DECIMAL(20,4),
    positions_value DECIMAL(20,4),
    
    -- Performance
    daily_return DECIMAL(10,6),
    ytd_return DECIMAL(10,6),
    total_return DECIMAL(10,6),
    
    -- Risk metrics
    sharpe_ratio DECIMAL(8,4),
    sortino_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(10,6),
    current_drawdown DECIMAL(10,6),
    var_95 DECIMAL(15,4),  -- Value at Risk 95%
    var_99 DECIMAL(15,4),  -- Value at Risk 99%
    
    -- Positions
    position_count INTEGER,
    long_exposure DECIMAL(10,4),
    short_exposure DECIMAL(10,4),
    net_exposure DECIMAL(10,4),
    gross_exposure DECIMAL(10,4),
    
    -- Full positions as JSON
    positions JSONB,
    
    UNIQUE(snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trade_history(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_agent ON trade_history(agent_name);
CREATE INDEX IF NOT EXISTS idx_snapshots_date ON portfolio_snapshots(snapshot_date);


-- ============================================================================
-- SECTION 6: FLYWHEEL KNOWLEDGE SHARING
-- ============================================================================

-- Insights shared between agents
CREATE TABLE IF NOT EXISTS agent_insights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Source
    source_agent VARCHAR(100) NOT NULL,
    
    -- Insight content
    insight_type VARCHAR(50) NOT NULL,  -- pattern, risk, opportunity, warning
    category VARCHAR(50),  -- market, sector, macro, technical, etc.
    title VARCHAR(200),
    content TEXT NOT NULL,
    confidence DECIMAL(5,4),
    
    -- Relevance
    symbols TEXT[],
    sectors TEXT[],
    time_horizon VARCHAR(50),
    
    -- Aggregation (for HoagsAgent)
    aggregated_by VARCHAR(100),
    aggregated_at TIMESTAMP,
    aggregation_weight DECIMAL(5,4),
    
    -- Redistribution
    redistributed BOOLEAN DEFAULT FALSE,
    redistributed_to TEXT[],  -- List of agents
    redistributed_at TIMESTAMP,
    
    -- Validation
    is_validated BOOLEAN DEFAULT FALSE,
    validation_outcome VARCHAR(50),  -- correct, incorrect, pending
    validation_notes TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

-- Training sessions
CREATE TABLE IF NOT EXISTS training_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_name VARCHAR(200),
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    
    -- Status
    status VARCHAR(50) DEFAULT 'running',  -- running, completed, failed, paused
    
    -- Progress
    total_agents INTEGER,
    agents_completed INTEGER DEFAULT 0,
    current_agent VARCHAR(100),
    
    -- Results
    overall_improvement DECIMAL(10,4),
    flywheel_cycles INTEGER DEFAULT 0,
    
    -- Details
    agent_results JSONB,
    errors TEXT[],
    config JSONB,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_insights_source ON agent_insights(source_agent);
CREATE INDEX IF NOT EXISTS idx_insights_type ON agent_insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_insights_created ON agent_insights(created_at);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON training_sessions(status);


-- ============================================================================
-- SECTION 7: ALTERNATIVE DATA
-- ============================================================================

-- News sentiment
CREATE TABLE IF NOT EXISTS news_sentiment (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20),
    published_at TIMESTAMP NOT NULL,
    
    -- Article info
    headline TEXT NOT NULL,
    source VARCHAR(100),
    url TEXT,
    
    -- Sentiment
    sentiment_score DECIMAL(5,4),  -- -1.0 to 1.0
    sentiment_label VARCHAR(20),  -- positive, negative, neutral
    relevance_score DECIMAL(5,4),  -- 0.0 to 1.0
    
    -- Entities
    entities JSONB,  -- Companies, people, topics mentioned
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insider transactions
CREATE TABLE IF NOT EXISTS insider_transactions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    filed_date DATE NOT NULL,
    transaction_date DATE NOT NULL,
    
    -- Insider info
    insider_name VARCHAR(200),
    insider_title VARCHAR(200),
    insider_type VARCHAR(50),  -- officer, director, 10% owner
    
    -- Transaction
    transaction_type VARCHAR(50),  -- purchase, sale, exercise
    shares INTEGER,
    price DECIMAL(15,4),
    value DECIMAL(20,4),
    
    -- Post-transaction
    shares_owned INTEGER,
    ownership_type VARCHAR(50),  -- direct, indirect
    
    source VARCHAR(50) DEFAULT 'sec',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Economic indicators
CREATE TABLE IF NOT EXISTS economic_indicators (
    id SERIAL PRIMARY KEY,
    indicator_name VARCHAR(100) NOT NULL,
    release_date DATE NOT NULL,
    
    -- Values
    actual DECIMAL(20,4),
    forecast DECIMAL(20,4),
    previous DECIMAL(20,4),
    
    -- Surprise
    surprise DECIMAL(20,4),
    surprise_pct DECIMAL(10,4),
    
    -- Market impact
    spx_reaction DECIMAL(10,4),
    
    source VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(indicator_name, release_date)
);

CREATE INDEX IF NOT EXISTS idx_news_symbol ON news_sentiment(symbol);
CREATE INDEX IF NOT EXISTS idx_news_published ON news_sentiment(published_at);
CREATE INDEX IF NOT EXISTS idx_insider_symbol ON insider_transactions(symbol);
CREATE INDEX IF NOT EXISTS idx_econ_name ON economic_indicators(indicator_name);


-- ============================================================================
-- SECTION 8: VIEWS FOR ANALYSIS
-- ============================================================================

-- Agent performance summary
CREATE OR REPLACE VIEW v_agent_performance AS
SELECT 
    agent_name,
    COUNT(*) as total_signals,
    COUNT(CASE WHEN outcome_realized THEN 1 END) as realized_signals,
    AVG(CASE WHEN outcome_realized THEN outcome_return END) as avg_return,
    AVG(confidence) as avg_confidence,
    COUNT(CASE WHEN outcome_realized AND outcome_return > 0 THEN 1 END)::DECIMAL / 
        NULLIF(COUNT(CASE WHEN outcome_realized THEN 1 END), 0) as win_rate,
    MIN(signal_time) as first_signal,
    MAX(signal_time) as last_signal
FROM agent_signals
GROUP BY agent_name;

-- High confidence recent signals
CREATE OR REPLACE VIEW v_high_confidence_signals AS
SELECT 
    id,
    agent_name,
    symbol,
    signal_type,
    confidence,
    thesis_summary,
    target_price,
    stop_loss,
    signal_time
FROM agent_signals
WHERE confidence >= 0.75
  AND signal_time > NOW() - INTERVAL '7 days'
  AND NOT outcome_realized
ORDER BY confidence DESC, signal_time DESC;

-- Daily performance summary
CREATE OR REPLACE VIEW v_daily_performance AS
SELECT 
    snapshot_date,
    total_value,
    daily_return,
    ytd_return,
    sharpe_ratio,
    max_drawdown,
    position_count,
    net_exposure
FROM portfolio_snapshots
ORDER BY snapshot_date DESC;

-- Flywheel metrics
CREATE OR REPLACE VIEW v_flywheel_metrics AS
SELECT 
    DATE(created_at) as insight_date,
    COUNT(*) as insights_created,
    COUNT(CASE WHEN redistributed THEN 1 END) as insights_redistributed,
    COUNT(CASE WHEN is_validated THEN 1 END) as insights_validated,
    AVG(confidence) as avg_confidence,
    COUNT(DISTINCT source_agent) as contributing_agents
FROM agent_insights
GROUP BY DATE(created_at)
ORDER BY insight_date DESC;

-- ============================================================================
-- GRANT PERMISSIONS (if using separate user)
-- ============================================================================
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO alc_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO alc_user;

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

-- Insert agent list
INSERT INTO agent_models (agent_name, model_version, trained_at, training_examples, is_active)
VALUES 
    ('HoagsAgent', 'v0.1', NOW(), 0, TRUE),
    ('MoneybagsAgent', 'v0.1', NOW(), 0, TRUE),
    ('EquitiesTraderAgent', 'v0.1', NOW(), 0, TRUE),
    ('OptionsTraderAgent', 'v0.1', NOW(), 0, TRUE),
    ('RiskManagementAgent', 'v0.1', NOW(), 0, TRUE),
    ('NewsAnalystAgent', 'v0.1', NOW(), 0, TRUE),
    ('ValuationAgent', 'v0.1', NOW(), 0, TRUE)
ON CONFLICT (agent_name, model_version) DO NOTHING;

-- ============================================================================
-- SCHEMA COMPLETE
-- ============================================================================

SELECT 'ALC Database Schema Created Successfully!' as status;
SELECT COUNT(*) as table_count FROM information_schema.tables WHERE table_schema = 'public';
