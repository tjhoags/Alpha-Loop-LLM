"""Analysis module - Comprehensive valuation, factor analysis, and behavioral finance.
"""

from src.analysis.behavioral_finance import (
    # Cognitive Biases
    CognitiveBiasSignals,
    # Crowd Behavior
    CrowdBehavior,
    CrowdBehaviorAnalysis,
    # Game Theory
    GameTheorySignals,
    MarketPsychology,
    # Psychology
    MarketRegime,
    SentimentData,
    # Sentiment
    SentimentSource,
    SocialSentimentAnalyzer,
    # Aggregate
    calculate_behavioral_alpha_score,
)
from src.analysis.valuation_suite import (
    # Comparable Analysis
    ComparableAnalysis,
    # DCF Models
    DCFModel,
    # Factor Models
    FactorModels,
    FactorStyle,
    # Data structures
    FinancialProjections,
    HoganModel,
    LBOAssumptions,
    # Transaction Models
    LBOModel,
    MergerAssumptions,
    MergerModel,
    # Utilities
    football_field_data,
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

