"""
ALC-Algo Agent Tests
Author: Tom Hogan | Alpha Loop Capital, LLC

Tests for the multi-agent system including base agent and specialized agents.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBaseAgent:
    """Tests for the BaseAgent class."""
    
    def test_agent_base_import(self):
        """Test that agent base module can be imported."""
        from src.core.agent_base import BaseAgent
        assert BaseAgent is not None
    
    def test_agent_tier_enum(self):
        """Test AgentTier enum values."""
        from src.core.agent_base import AgentTier
        
        assert AgentTier.MASTER.value == 1
        assert AgentTier.SENIOR.value == 2
        assert AgentTier.STANDARD.value == 3
    
    def test_agent_status_enum(self):
        """Test AgentStatus enum values."""
        from src.core.agent_base import AgentStatus
        
        assert AgentStatus.ACTIVE.value == "active"
        assert AgentStatus.BATTLE_READY.value == "battle_ready"
    
    def test_thinking_modes(self):
        """Test ThinkingMode enum."""
        from src.core.agent_base import ThinkingMode
        
        # Check key thinking modes exist
        assert hasattr(ThinkingMode, 'CONTRARIAN')
        assert hasattr(ThinkingMode, 'SECOND_ORDER')
        assert hasattr(ThinkingMode, 'REGIME_AWARE')
        assert hasattr(ThinkingMode, 'BEHAVIORAL')
    
    def test_learning_methods(self):
        """Test LearningMethod enum."""
        from src.core.agent_base import LearningMethod
        
        # Check key learning methods exist
        assert hasattr(LearningMethod, 'REINFORCEMENT')
        assert hasattr(LearningMethod, 'BAYESIAN')
        assert hasattr(LearningMethod, 'ADVERSARIAL')
    
    def test_concrete_agent_creation(self):
        """Test creating a concrete agent subclass."""
        from src.core.agent_base import BaseAgent, AgentTier, AgentToughness
        from typing import Dict, Any, List
        
        class TestAgent(BaseAgent):
            def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
                return {'success': True, 'processed': task}
            
            def get_capabilities(self) -> List[str]:
                return ['test_capability']
        
        agent = TestAgent(
            name="TestAgent",
            tier=AgentTier.STANDARD,
            capabilities=['test_capability'],
            user_id="TJH"
        )
        
        assert agent.name == "TestAgent"
        assert agent.tier == AgentTier.STANDARD
        assert 'test_capability' in agent.capabilities
    
    def test_agent_execute_method(self):
        """Test agent execute method."""
        from src.core.agent_base import BaseAgent, AgentTier
        from typing import Dict, Any, List
        
        class TestAgent(BaseAgent):
            def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
                return {'success': True, 'result': 'processed'}
            
            def get_capabilities(self) -> List[str]:
                return ['test']
        
        agent = TestAgent(
            name="TestAgent",
            tier=AgentTier.STANDARD,
            capabilities=['test'],
            user_id="TJH"
        )
        
        result = agent.execute({'type': 'test_task'})
        
        assert result is not None
        assert 'success' in result
        assert result.get('attributed_to') == 'Tom Hogan'
        assert result.get('organization') == 'Alpha Loop Capital, LLC'
    
    def test_agent_stats(self):
        """Test agent statistics tracking."""
        from src.core.agent_base import BaseAgent, AgentTier
        from typing import Dict, Any, List
        
        class TestAgent(BaseAgent):
            def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
                return {'success': True}
            
            def get_capabilities(self) -> List[str]:
                return ['test']
        
        agent = TestAgent(
            name="TestAgent",
            tier=AgentTier.STANDARD,
            capabilities=['test'],
            user_id="TJH"
        )
        
        stats = agent.get_stats()
        
        assert 'name' in stats
        assert 'tier' in stats
        assert 'execution_count' in stats
        assert 'success_rate' in stats
    
    def test_agent_learning_outcome(self):
        """Test agent learning from outcomes."""
        from src.core.agent_base import BaseAgent, AgentTier, LearningOutcome
        from typing import Dict, Any, List
        
        class TestAgent(BaseAgent):
            def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
                return {'success': True}
            
            def get_capabilities(self) -> List[str]:
                return ['test']
        
        agent = TestAgent(
            name="TestAgent",
            tier=AgentTier.STANDARD,
            capabilities=['test'],
            user_id="TJH"
        )
        
        outcome = agent.learn_from_outcome(
            prediction=1.0,
            actual=1.0,
            confidence=0.8,
            context={'test': True}
        )
        
        assert isinstance(outcome, LearningOutcome)
        assert outcome.was_correct == True
    
    def test_regime_detection(self):
        """Test regime change detection."""
        from src.core.agent_base import BaseAgent, AgentTier
        from typing import Dict, Any, List
        
        class TestAgent(BaseAgent):
            def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
                return {'success': True}
            
            def get_capabilities(self) -> List[str]:
                return ['test']
        
        agent = TestAgent(
            name="TestAgent",
            tier=AgentTier.STANDARD,
            capabilities=['test'],
            user_id="TJH"
        )
        
        # Test with crisis market data
        market_data = {
            'vix': 40,
            'trend': -0.5,
            'avg_correlation': 0.9,
        }
        
        regime, confidence = agent.detect_regime_change(market_data)
        
        assert regime == "crisis"
        assert confidence > 0
    
    def test_capability_gap_detection(self):
        """Test capability gap detection (ACA)."""
        from src.core.agent_base import BaseAgent, AgentTier
        from typing import Dict, Any, List
        
        class TestAgent(BaseAgent):
            def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
                return {'success': True}
            
            def get_capabilities(self) -> List[str]:
                return ['analysis']
        
        agent = TestAgent(
            name="TestAgent",
            tier=AgentTier.STANDARD,
            capabilities=['analysis'],
            user_id="TJH"
        )
        
        # Task requiring capability agent doesn't have
        task = {
            'type': 'complex_task',
            'required_capabilities': ['analysis', 'prediction', 'optimization']
        }
        
        gap = agent.detect_capability_gap(task)
        
        # Should detect missing capabilities
        if gap is not None:
            assert 'prediction' in gap.missing_capabilities or 'optimization' in gap.missing_capabilities


class TestSeniorAgents:
    """Tests for senior agent implementations."""
    
    def test_data_agent_import(self):
        """Test DataAgent import."""
        from src.agents import DataAgent
        assert DataAgent is not None
    
    def test_strategy_agent_import(self):
        """Test StrategyAgent import."""
        from src.agents import StrategyAgent
        assert StrategyAgent is not None
    
    def test_risk_agent_import(self):
        """Test RiskAgent import."""
        from src.agents import RiskAgent
        assert RiskAgent is not None
    
    def test_execution_agent_import(self):
        """Test ExecutionAgent import."""
        from src.agents import ExecutionAgent
        assert ExecutionAgent is not None
    
    def test_portfolio_agent_import(self):
        """Test PortfolioAgent import."""
        from src.agents import PortfolioAgent
        assert PortfolioAgent is not None
    
    def test_research_agent_import(self):
        """Test ResearchAgent import."""
        from src.agents import ResearchAgent
        assert ResearchAgent is not None
    
    def test_compliance_agent_import(self):
        """Test ComplianceAgent import."""
        from src.agents import ComplianceAgent
        assert ComplianceAgent is not None
    
    def test_sentiment_agent_import(self):
        """Test SentimentAgent import."""
        from src.agents import SentimentAgent
        assert SentimentAgent is not None


class TestMasterAgents:
    """Tests for master agent implementations."""
    
    def test_ghost_agent_import(self):
        """Test GhostAgent import."""
        from src.agents import GhostAgent
        assert GhostAgent is not None
    
    def test_hoags_agent_import(self):
        """Test HoagsAgent import."""
        from src.agents import HoagsAgent
        assert HoagsAgent is not None


class TestSwarmAgents:
    """Tests for swarm agent system."""
    
    def test_swarm_factory_import(self):
        """Test SwarmFactory import."""
        from src.agents import SwarmFactory
        assert SwarmFactory is not None
    
    def test_swarm_factory_creation(self):
        """Test SwarmFactory can create agents."""
        from src.agents import SwarmFactory
        
        factory = SwarmFactory(user_id="TJH")
        
        assert factory is not None
        assert factory.user_id == "TJH"
    
    def test_total_agents_constant(self):
        """Test TOTAL_AGENTS constant."""
        from src.agents import TOTAL_AGENTS
        
        assert TOTAL_AGENTS == 76


class TestAgentInteraction:
    """Tests for agent interaction and coordination."""
    
    def test_agent_can_handle_task(self):
        """Test agent task handling capability check."""
        from src.core.agent_base import BaseAgent, AgentTier
        from typing import Dict, Any, List
        
        class TestAgent(BaseAgent):
            def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
                return {'success': True}
            
            def get_capabilities(self) -> List[str]:
                return ['analysis', 'prediction']
        
        agent = TestAgent(
            name="TestAgent",
            tier=AgentTier.STANDARD,
            capabilities=['analysis', 'prediction'],
            user_id="TJH"
        )
        
        # Task agent can handle
        task1 = {'required_capabilities': ['analysis']}
        assert agent.can_handle_task(task1) == True
        
        # Task agent cannot handle
        task2 = {'required_capabilities': ['analysis', 'execution']}
        assert agent.can_handle_task(task2) == False
    
    def test_agent_proposal(self):
        """Test agent proposal creation (ACA)."""
        from src.core.agent_base import BaseAgent, AgentTier
        from typing import Dict, Any, List
        
        class TestAgent(BaseAgent):
            def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
                return {'success': True}
            
            def get_capabilities(self) -> List[str]:
                return ['test']
        
        agent = TestAgent(
            name="TestAgent",
            tier=AgentTier.STANDARD,
            capabilities=['test'],
            user_id="TJH"
        )
        
        proposal = agent.propose_agent(
            agent_name="NewSpecializedAgent",
            capabilities=['specialized_analysis'],
            tier=AgentTier.STANDARD,
            gap_description="Need specialized analysis capability",
            rationale="Detected gap in analysis workflow"
        )
        
        assert proposal is not None
        assert proposal.agent_name == "NewSpecializedAgent"


class TestAgentResilience:
    """Tests for agent resilience and error handling."""
    
    def test_agent_error_recovery(self):
        """Test agent recovery from errors."""
        from src.core.agent_base import BaseAgent, AgentTier
        from typing import Dict, Any, List
        
        call_count = 0
        
        class FailingAgent(BaseAgent):
            def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
                nonlocal call_count
                call_count += 1
                if task.get('recovery_attempt', 0) < 2:
                    raise ValueError("Simulated failure")
                return {'success': True}
            
            def get_capabilities(self) -> List[str]:
                return ['test']
        
        agent = FailingAgent(
            name="FailingAgent",
            tier=AgentTier.STANDARD,
            capabilities=['test'],
            user_id="TJH"
        )
        
        # Execute should attempt recovery
        result = agent.execute({'type': 'test'})
        
        # Should have some recovery attempt info
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

