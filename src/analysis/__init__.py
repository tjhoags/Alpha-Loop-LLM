"""
Analysis module - Comprehensive valuation, factor analysis, and behavioral finance.
"""

from src.analysis.valuation_suite import (
    # Data structures
    FinancialProjections,
    LBOAssumptions,
    MergerAssumptions,
    FactorStyle,
    
    # DCF Models
    DCFModel,
    HoganModel,
    
    # Transaction Models
    LBOModel,
    MergerModel,
    
    # Comparable Analysis
    ComparableAnalysis,
    
    # Factor Models
    FactorModels,
    
    # Utilities
    football_field_data,
)

from src.analysis.behavioral_finance import (
    # Sentiment
    SentimentSource,
    SentimentData,
    SocialSentimentAnalyzer,
    
    # Psychology
    MarketRegime,
    MarketPsychology,
    
    # Game Theory
    GameTheorySignals,
    
    # Crowd Behavior
    CrowdBehavior,
    CrowdBehaviorAnalysis,
    
    # Cognitive Biases
    CognitiveBiasSignals,
    
    # Aggregate
    calculate_behavioral_alpha_score,
)

__all__ = [
    # Valuation
    "FinancialProjections",
    "LBOAssumptions", 
    "MergerAssumptions",
    "FactorStyle",
    "DCFModel",
    "HoganModel",
    "LBOModel",
    "MergerModel",
    "ComparableAnalysis",
    "FactorModels",
    "football_field_data",
    
    # Behavioral Finance
    "SentimentSource",
    "SentimentData",
    "SocialSentimentAnalyzer",
    "MarketRegime",
    "MarketPsychology",
    "GameTheorySignals",
    "CrowdBehavior",
    "CrowdBehaviorAnalysis",
    "CognitiveBiasSignals",
    "calculate_behavioral_alpha_score",
]

