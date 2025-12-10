
"""Sector agents package."""
from .base_sector import BaseSectorAgent
from .consumer_sector import ConsumerSectorAgent
from .crypto_sector import CryptoSectorAgent
from .emerging_markets import EmergingMarketsAgent
from .energy_sector import EnergySectorAgent
from .financial_sector import FinancialSectorAgent
from .healthcare_sector import HealthcareSectorAgent
from .industrial_sector import IndustrialSectorAgent
from .materials_sector import MaterialsSectorAgent
from .realestate_sector import RealestateSectorAgent
from .tech_sector import TechSectorAgent
__all__ = [name for name in globals() if name.endswith('Agent')]
