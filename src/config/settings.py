"""================================================================================
SETTINGS - Centralized Configuration Management
================================================================================
All application settings loaded from environment variables with sensible defaults.
Uses Pydantic for validation and type safety.

IMPORTANT: Never commit credentials to source control. Use .env files.
================================================================================
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from multiple possible locations
_env_paths = [
    Path(os.getenv("ALC_ENV_PATH", "")),  # Custom env path from environment
    Path(__file__).resolve().parents[2] / ".env",  # Project root .env
    Path.home() / ".alc" / ".env",  # User home config
]

_env_loaded = False
for env_path in _env_paths:
    if env_path.exists():
        load_dotenv(env_path, override=True)
        _env_loaded = True
        break

if not _env_loaded:
    load_dotenv()  # Fallback to default .env discovery


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All credentials and paths should be set via environment variables.
    See .env.example for required variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ==========================================================================
    # DATABASE - Azure SQL Server
    # ==========================================================================
    database_url: Optional[str] = Field(default=None, description="Full SQLAlchemy URL (overrides components)")
    sql_server: str = Field(default_factory=lambda: os.getenv("SQL_SERVER", ""))
    sql_db: str = Field(default_factory=lambda: os.getenv("SQL_DB", "alc_market_data"))
    db_username: str = Field(default_factory=lambda: os.getenv("DB_USERNAME", ""))
    db_password: str = Field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))
    db_odbc_driver: str = Field(default="ODBC Driver 17 for SQL Server")

    # ==========================================================================
    # DATA APIS - Using os.getenv since dotenv is loaded above
    # ==========================================================================
    alpha_vantage_api_key: str = Field(default_factory=lambda: os.getenv("ALPHAVANTAGE_API_KEY", ""))
    polygon_api_key: str = Field(default_factory=lambda: os.getenv("PolygonIO_API_KEY", ""))
    massive_access_key: str = Field(default_factory=lambda: os.getenv("MASSIVE_ACCESS_KEY", ""))
    massive_secret_key: str = Field(default_factory=lambda: os.getenv("MASSIVE_SECRET_KEY", ""))
    massive_endpoint_url: str = Field(default_factory=lambda: os.getenv("MASSIVE_ENDPOINT_URL", "https://files.polygon.io"))
    coinbase_api_key: str = Field(default_factory=lambda: os.getenv("COINBASE_API_KEY", ""))
    coinbase_api_secret: str = Field(default_factory=lambda: os.getenv("COINBASE_API_SECRET", ""))
    fred_api_key: str = Field(default_factory=lambda: os.getenv("FRED_API_KEY", ""))

    # ==========================================================================
    # AI SERVICES
    # ==========================================================================
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_SECRET", ""))
    anthropic_api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    perplexity_api_key: str = Field(default_factory=lambda: os.getenv("PERPLEXITY_API_KEY", ""))
    google_api_key: str = Field(default_factory=lambda: os.getenv("API_KEY", ""))

    # ==========================================================================
    # TRADING / IBKR
    # ==========================================================================
    ibkr_host: str = Field(default="127.0.0.1")
    ibkr_port: int = Field(default=7497)  # 7497=Paper, 7496=Live
    ibkr_client_id: int = Field(default=1)

    # Paths
    base_dir: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = base_dir / "data"
    models_dir: Path = base_dir / "models"
    logs_dir: Path = base_dir / "logs"
    vectorstore_dir: Path = base_dir / "vectorstore"

    # ==========================================================================
    # DATA / ML CONFIG
    # ==========================================================================
    # FULL UNIVERSE MODE - No hardcoded tickers, pulls ALL from Polygon/Massive
    # NO FILTERS - Get everything
    target_symbols: List[str] = Field(default_factory=list)  # Empty = use full universe
    use_full_universe: bool = Field(default=True)  # Pull ALL tickers
    train_test_split: float = Field(default=0.8)
    lookback_window: int = Field(default=60)
    time_granularity_minutes: int = Field(default=5)
    polygon_lookback_hours: int = Field(default=240)
    coinbase_lookback_hours: int = Field(default=240)
    alpha_vantage_outputsize: str = Field(default="full")

    # Research ingestion / NLP
    research_paths: List[str] = Field(
        default_factory=lambda: [
            p for p in os.getenv("RESEARCH_PATHS", "").split(";") if p
        ] or [],
    )
    embedding_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    max_chunk_size: int = Field(default=1200)
    chunk_overlap: int = Field(default=200)

    # ==========================================================================
    # RISK MANAGEMENT (HARD LIMITS)
    # ==========================================================================
    max_daily_loss_pct: float = Field(default=0.02)
    max_drawdown_pct: float = Field(default=0.05)
    max_position_size_pct: float = Field(default=0.10)
    max_positions: int = Field(default=10)
    kelly_fraction_cap: float = Field(default=0.25)

    # ==========================================================================
    # MODEL GRADING THRESHOLDS
    # ==========================================================================
    min_auc: float = Field(default=0.52)
    min_accuracy: float = Field(default=0.52)
    min_sharpe_ratio: float = Field(default=1.5)
    max_validation_drawdown: float = Field(default=0.05)

    # ==========================================================================
    # MISC
    # ==========================================================================
    log_level: str = Field(default="INFO")

    @field_validator("train_test_split")
    @classmethod
    def validate_split(cls, v: float) -> float:
        if not 0.5 < v < 0.95:
            raise ValueError("train_test_split should be between 0.5 and 0.95")
        return v

    # Use SQLite for local development (fast, no setup)
    use_sqlite: bool = Field(default=False)

    @property
    def sqlalchemy_url(self) -> str:
        """Build SQLAlchemy connection URL with fallback to SQLite."""
        if self.database_url:
            return self.database_url

        # Fallback to SQLite for local development
        if self.use_sqlite or not self.sql_server:
            sqlite_path = self.data_dir / "market_data.db"
            return f"sqlite:///{sqlite_path}"

        # Azure SQL Server
        db_name = self.sql_db.split("/")[-1] if "/" in self.sql_db else self.sql_db
        server = self.sql_server

        if server and not server.endswith(".database.windows.net"):
            server = f"{server}.database.windows.net"

        driver = self.db_odbc_driver.replace(" ", "+")

        if self.db_username and self.db_password:
            # URL-encode password to handle special characters
            from urllib.parse import quote_plus
            encoded_pwd = quote_plus(self.db_password)
            return f"mssql+pyodbc://{self.db_username}:{encoded_pwd}@{server}/{db_name}?driver={driver}"
        else:
            return f"mssql+pyodbc://@{server}/{db_name}?driver={driver}&Authentication=ActiveDirectoryInteractive"

    def validate_required_apis(self) -> dict[str, bool]:
        """Check which APIs are configured. Returns dict of api_name -> is_configured."""
        return {
            "database": bool(self.sql_server and self.db_username) or self.use_sqlite,
            "polygon": bool(self.polygon_api_key),
            "alpha_vantage": bool(self.alpha_vantage_api_key),
            "coinbase": bool(self.coinbase_api_key),
            "fred": bool(self.fred_api_key),
            "openai": bool(self.openai_api_key),
            "anthropic": bool(self.anthropic_api_key),
        }

    def log_configuration_status(self) -> None:
        """Log which APIs are configured (for startup diagnostics)."""
        from loguru import logger
        api_status = self.validate_required_apis()
        configured = [k for k, v in api_status.items() if v]
        missing = [k for k, v in api_status.items() if not v]

        logger.info(f"Configured APIs: {', '.join(configured) or 'None'}")
        if missing:
            logger.warning(f"Missing API keys: {', '.join(missing)}")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    # Ensure critical directories exist
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    return settings
