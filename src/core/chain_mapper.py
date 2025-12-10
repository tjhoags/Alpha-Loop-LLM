"""
MODULE 1: BENEFICIARY CHAIN MAPPER
==================================
Alpha Loop Capital - Consequence Engine

Purpose: Automate multi-order beneficiary analysis
         Maps Order 0→4 consequence chains from any investment thesis
         
Core Edge: Position where algos aren't competing (Order 2-4)

Author: Tom Hogan
Version: 1.0
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
from datetime import datetime
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderLevel(Enum):
    """
    Order levels represent distance from the obvious play.
    Lower competition exists at higher order levels.
    
    ORDER 0: Direct/Obvious (Algos compete here, HFT active)
    ORDER 1: Sell-side consensus (Covered by analysts, crowded)
    ORDER 2: Hidden beneficiaries (Your sweet spot - fewer eyes)
    ORDER 3: Deep edge (Market not looking, requires thesis)
    ORDER 4: Ignored/Unknown (Maximum edge if connection is real)
    """
    ZERO = 0    # Direct/obvious - algos here
    ONE = 1     # Sell-side consensus
    TWO = 2     # Hidden beneficiaries - Tom's zone
    THREE = 3   # Deep edge
    FOUR = 4    # Ignored/unknown


class ConnectionType(Enum):
    """Types of beneficiary relationships"""
    DIRECT = "direct"                    # The thesis target itself
    PRODUCER = "producer"                # Produces the commodity/product
    SUPPLIER = "supplier"                # Supplies critical inputs
    CUSTOMER = "customer"                # Major customer benefits
    ASSET_HOLDER = "asset_holder"        # Holds physical/financial assets
    ASSET_MANAGER = "asset_manager"      # Manages exposure (AUM play)
    JV_PARTNER = "jv_partner"            # Joint venture partner
    ROYALTY = "royalty"                  # Royalty/streaming exposure
    INFRASTRUCTURE = "infrastructure"    # Picks and shovels
    LOGISTICS = "logistics"              # Transport/storage
    FINANCING = "financing"              # Provides capital/financing
    SUBSTITUTE = "substitute"            # Alternative if thesis plays out
    DERIVATIVE = "derivative"            # Financial derivative exposure


class AlgoCompetition(Enum):
    """Level of algorithmic trading competition"""
    HIGH = "high"       # >50% algo volume, tight spreads, HFT active
    MEDIUM = "medium"   # 25-50% algo volume, moderate spreads
    LOW = "low"         # <25% algo volume, wider spreads, less efficient


class CatalystTimeline(Enum):
    """Expected timeline for thesis to play out"""
    IMMEDIATE = "immediate"   # 0-30 days
    SHORT = "3mo"            # 1-3 months
    MEDIUM = "6mo"           # 3-6 months
    LONG = "12mo"            # 6-12 months
    EXTENDED = "12mo+"       # >12 months


@dataclass
class ChainNode:
    """
    A single node in the beneficiary chain.
    Represents one company/asset that benefits from the thesis.
    """
    ticker: str
    company_name: str
    order_level: OrderLevel
    connection_type: ConnectionType
    connection_strength: float  # 0.0-1.0, how certain is this link?
    
    # Competition metrics
    analyst_coverage: int = 0
    algo_competition: AlgoCompetition = AlgoCompetition.MEDIUM
    daily_volume_mm: float = 0.0
    market_cap_mm: float = 0.0
    
    # Timing
    catalyst_timeline: CatalystTimeline = CatalystTimeline.MEDIUM
    next_catalyst: str = ""
    next_catalyst_date: Optional[str] = None
    
    # Risk factors
    downside_scenario: str = ""
    conviction_level: float = 0.5
    
    # Chain relationship
    parent_ticker: Optional[str] = None
    connection_description: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "order_level": self.order_level.value,
            "connection_type": self.connection_type.value,
            "connection_strength": self.connection_strength,
            "analyst_coverage": self.analyst_coverage,
            "algo_competition": self.algo_competition.value,
            "daily_volume_mm": self.daily_volume_mm,
            "market_cap_mm": self.market_cap_mm,
            "catalyst_timeline": self.catalyst_timeline.value,
            "next_catalyst": self.next_catalyst,
            "conviction_level": self.conviction_level,
            "parent_ticker": self.parent_ticker,
            "connection_description": self.connection_description
        }
    
    @property
    def tom_score(self) -> float:
        """
        Calculate Tom Score - composite ranking for position attractiveness.
        Higher = better opportunity.
        
        Components:
        - Order level bonus (Order 2-4 get premium)
        - Low competition bonus
        - Connection strength
        - Conviction level
        - Low analyst coverage bonus
        """
        score = 50.0  # Base score
        
        # Order level bonus (Order 2-3 are sweet spot)
        order_bonus = {
            OrderLevel.ZERO: 0,
            OrderLevel.ONE: 5,
            OrderLevel.TWO: 20,
            OrderLevel.THREE: 25,
            OrderLevel.FOUR: 15  # Slightly less because harder to validate
        }
        score += order_bonus.get(self.order_level, 0)
        
        # Competition bonus
        if self.algo_competition == AlgoCompetition.LOW:
            score += 15
        elif self.algo_competition == AlgoCompetition.MEDIUM:
            score += 5
        
        # Connection strength (0-10 points)
        score += self.connection_strength * 10
        
        # Conviction (0-10 points)
        score += self.conviction_level * 10
        
        # Low analyst coverage bonus (less covered = more opportunity)
        if self.analyst_coverage <= 2:
            score += 10
        elif self.analyst_coverage <= 5:
            score += 5
        
        return min(100, max(0, score))


@dataclass
class BeneficiaryChain:
    """
    Complete beneficiary chain for a thesis.
    Maps all order levels from the obvious play to hidden gems.
    """
    thesis_name: str
    thesis_description: str
    thesis_category: str
    created_date: str
    
    # Chain nodes by order level
    nodes: List[ChainNode] = field(default_factory=list)
    
    # Thesis metadata
    confidence: float = 0.0
    time_horizon: CatalystTimeline = CatalystTimeline.MEDIUM
    thesis_status: str = "active"
    
    def add_node(self, node: ChainNode) -> None:
        """Add a beneficiary to the chain"""
        self.nodes.append(node)
        logger.info(f"Added {node.ticker} to {self.thesis_name} chain at Order {node.order_level.value}")
    
    def get_by_order(self, order: OrderLevel) -> List[ChainNode]:
        """Get all nodes at a specific order level"""
        return [n for n in self.nodes if n.order_level == order]
    
    def get_toms_picks(self) -> List[ChainNode]:
        """Get Order 2-4 beneficiaries sorted by Tom Score"""
        toms_zone = [n for n in self.nodes if n.order_level.value >= 2]
        return sorted(toms_zone, key=lambda x: x.tom_score, reverse=True)
    
    def get_all_tickers(self) -> Set[str]:
        """Get set of all tickers in chain"""
        return {n.ticker for n in self.nodes}
    
    def to_dict(self) -> Dict:
        return {
            "thesis_name": self.thesis_name,
            "thesis_description": self.thesis_description,
            "thesis_category": self.thesis_category,
            "created_date": self.created_date,
            "confidence": self.confidence,
            "time_horizon": self.time_horizon.value,
            "nodes": [n.to_dict() for n in self.nodes],
            "node_count": len(self.nodes),
            "toms_picks": [n.to_dict() for n in self.get_toms_picks()[:5]]
        }


class ChainMapper:
    """
    BENEFICIARY CHAIN MAPPER
    
    The workhorse that:
    1. Takes an investment thesis as input
    2. Maps all beneficiaries across Order 0-4
    3. Scores opportunities using Tom Score
    4. Identifies the best Order 2-4 plays
    
    Usage:
        mapper = ChainMapper()
        chain = mapper.create_chain("Uranium Bull", "Nuclear renaissance thesis", "commodity_squeeze")
        mapper.add_beneficiary(chain, "CCJ", "Cameco", OrderLevel.ZERO, ConnectionType.PRODUCER, ...)
        mapper.add_beneficiary(chain, "SII", "Sprott Inc", OrderLevel.TWO, ConnectionType.ASSET_MANAGER, ...)
        recommendations = mapper.get_recommendations(chain)
    """
    
    def __init__(self):
        self.chains: Dict[str, BeneficiaryChain] = {}
        self.thesis_categories = [
            "commodity_squeeze",
            "energy_transition", 
            "geopolitical",
            "narrative_momentum",
            "structural_flow",
            "sector_rotation",
            "macro_theme"
        ]
    
    def create_chain(
        self,
        thesis_name: str,
        thesis_description: str,
        thesis_category: str,
        confidence: float = 0.5,
        time_horizon: CatalystTimeline = CatalystTimeline.MEDIUM
    ) -> BeneficiaryChain:
        """Create a new beneficiary chain for a thesis"""
        chain = BeneficiaryChain(
            thesis_name=thesis_name,
            thesis_description=thesis_description,
            thesis_category=thesis_category,
            created_date=datetime.now().strftime("%Y-%m-%d"),
            confidence=confidence,
            time_horizon=time_horizon
        )
        self.chains[thesis_name] = chain
        logger.info(f"Created chain: {thesis_name}")
        return chain
    
    def add_beneficiary(
        self,
        chain: BeneficiaryChain,
        ticker: str,
        company_name: str,
        order_level: OrderLevel,
        connection_type: ConnectionType,
        connection_strength: float,
        connection_description: str = "",
        parent_ticker: Optional[str] = None,
        **kwargs
    ) -> ChainNode:
        """Add a beneficiary to an existing chain"""
        node = ChainNode(
            ticker=ticker,
            company_name=company_name,
            order_level=order_level,
            connection_type=connection_type,
            connection_strength=connection_strength,
            connection_description=connection_description,
            parent_ticker=parent_ticker,
            **kwargs
        )
        chain.add_node(node)
        return node
    
    def get_recommendations(
        self,
        chain: BeneficiaryChain,
        min_tom_score: float = 65.0,
        max_results: int = 10
    ) -> List[Dict]:
        """
        Get actionable recommendations from a chain.
        
        Returns sorted list of opportunities with:
        - Tom Score
        - Entry thesis
        - Risk factors
        - Suggested position size
        """
        picks = chain.get_toms_picks()
        recommendations = []
        
        for node in picks:
            if node.tom_score < min_tom_score:
                continue
                
            rec = {
                "ticker": node.ticker,
                "company_name": node.company_name,
                "tom_score": round(node.tom_score, 1),
                "order_level": node.order_level.value,
                "connection_type": node.connection_type.value,
                "thesis": chain.thesis_name,
                "entry_rationale": node.connection_description,
                "catalyst": node.next_catalyst,
                "timeline": node.catalyst_timeline.value,
                "risk": node.downside_scenario,
                "algo_competition": node.algo_competition.value,
                "suggested_action": self._get_action(node.tom_score),
                "position_size_suggestion": self._get_position_size(node.tom_score, node.conviction_level)
            }
            recommendations.append(rec)
            
            if len(recommendations) >= max_results:
                break
        
        return recommendations
    
    def _get_action(self, tom_score: float) -> str:
        """Get suggested action based on Tom Score"""
        if tom_score >= 85:
            return "STRONG BUY - Full position"
        elif tom_score >= 75:
            return "BUY - 75% position"
        elif tom_score >= 65:
            return "ACCUMULATE - 50% position, add on dips"
        else:
            return "WATCHLIST - Wait for better entry"
    
    def _get_position_size(self, tom_score: float, conviction: float) -> str:
        """Get suggested position size"""
        base_size = min(5.0, max(1.0, (tom_score - 50) / 10))
        adjusted_size = base_size * conviction
        return f"{adjusted_size:.1f}% of portfolio"
    
    def export_chain(self, chain_name: str, filepath: str) -> None:
        """Export chain to JSON file"""
        if chain_name not in self.chains:
            raise ValueError(f"Chain {chain_name} not found")
        
        chain = self.chains[chain_name]
        with open(filepath, 'w') as f:
            json.dump(chain.to_dict(), f, indent=2)
        logger.info(f"Exported {chain_name} to {filepath}")
    
    def import_chain(self, filepath: str) -> BeneficiaryChain:
        """Import chain from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        chain = self.create_chain(
            thesis_name=data["thesis_name"],
            thesis_description=data["thesis_description"],
            thesis_category=data["thesis_category"],
            confidence=data.get("confidence", 0.5),
            time_horizon=CatalystTimeline(data.get("time_horizon", "6mo"))
        )
        
        for node_data in data.get("nodes", []):
            self.add_beneficiary(
                chain=chain,
                ticker=node_data["ticker"],
                company_name=node_data["company_name"],
                order_level=OrderLevel(node_data["order_level"]),
                connection_type=ConnectionType(node_data["connection_type"]),
                connection_strength=node_data["connection_strength"],
                connection_description=node_data.get("connection_description", ""),
                conviction_level=node_data.get("conviction_level", 0.5),
                analyst_coverage=node_data.get("analyst_coverage", 0),
                algo_competition=AlgoCompetition(node_data.get("algo_competition", "medium"))
            )
        
        logger.info(f"Imported chain: {chain.thesis_name} with {len(chain.nodes)} nodes")
        return chain


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def create_example_uranium_chain() -> BeneficiaryChain:
    """Create example uranium thesis chain"""
    mapper = ChainMapper()
    
    chain = mapper.create_chain(
        thesis_name="Uranium Renaissance",
        thesis_description="Nuclear energy renaissance driven by AI data center demand and clean energy goals",
        thesis_category="commodity_squeeze",
        confidence=0.8,
        time_horizon=CatalystTimeline.LONG
    )
    
    # Order 0 - Direct/Obvious
    mapper.add_beneficiary(
        chain, "CCJ", "Cameco Corporation", OrderLevel.ZERO, ConnectionType.PRODUCER,
        connection_strength=0.95,
        connection_description="Largest pure-play uranium producer",
        algo_competition=AlgoCompetition.HIGH,
        analyst_coverage=18,
        conviction_level=0.9
    )
    
    # Order 1 - Sell-side consensus
    mapper.add_beneficiary(
        chain, "UEC", "Uranium Energy Corp", OrderLevel.ONE, ConnectionType.PRODUCER,
        connection_strength=0.85,
        connection_description="US-focused uranium producer",
        algo_competition=AlgoCompetition.MEDIUM,
        analyst_coverage=8,
        conviction_level=0.75
    )
    
    # Order 2 - Hidden beneficiaries (Tom's zone)
    mapper.add_beneficiary(
        chain, "SII", "Sprott Inc", OrderLevel.TWO, ConnectionType.ASSET_MANAGER,
        connection_strength=0.8,
        connection_description="Manages SRUUF physical uranium trust - AUM grows with price",
        algo_competition=AlgoCompetition.LOW,
        analyst_coverage=3,
        conviction_level=0.85,
        next_catalyst="Q1 2025 flows report"
    )
    
    # Order 3 - Deep edge
    mapper.add_beneficiary(
        chain, "BWXT", "BWX Technologies", OrderLevel.THREE, ConnectionType.INFRASTRUCTURE,
        connection_strength=0.7,
        connection_description="Nuclear fuel and reactor components - Navy + commercial",
        algo_competition=AlgoCompetition.MEDIUM,
        analyst_coverage=5,
        conviction_level=0.7
    )
    
    return chain


if __name__ == "__main__":
    # Demo
    chain = create_example_uranium_chain()
    mapper = ChainMapper()
    mapper.chains[chain.thesis_name] = chain
    
    print("\n" + "="*60)
    print("URANIUM RENAISSANCE - BENEFICIARY CHAIN")
    print("="*60)
    
    for order in range(5):
        nodes = chain.get_by_order(OrderLevel(order))
        if nodes:
            print(f"\nOrder {order}:")
            for n in nodes:
                print(f"  {n.ticker}: {n.company_name} (Tom Score: {n.tom_score:.1f})")
    
    print("\n" + "="*60)
    print("TOM'S PICKS (Order 2-4)")
    print("="*60)
    
    recs = mapper.get_recommendations(chain, min_tom_score=60)
    for rec in recs:
        print(f"\n{rec['ticker']} - Tom Score: {rec['tom_score']}")
        print(f"  Action: {rec['suggested_action']}")
        print(f"  Size: {rec['position_size_suggestion']}")

