import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env from your OneDrive location FIRST
ENV_PATH = r"C:\Users\tom\OneDrive\Alpha Loop LLM\API - Dec 2025.env"
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH, override=True)
else:
    load_dotenv()  # Fallback to default .env


class Settings(BaseSettings):
    """Application settings loaded from environment variables.
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
    database_url: Optional[str] = Field(default=None)
    sql_server: str = Field(default_factory=lambda: os.getenv("SQL_SERVER", "alc-sql-server.database.windows.net"))
    sql_db: str = Field(default_factory=lambda: os.getenv("SQL_DB", "alc_market_data"))
    db_username: str = Field(default_factory=lambda: os.getenv("DB_USERNAME", "CloudSAb3fcbb35"))
    db_password: str = Field(default_factory=lambda: os.getenv("DB_PASSWORD", "ALCadmin27!"))
    db_odbc_driver: str = Field(default="ODBC Driver 17 for SQL Server")

    # ==========================================================================
    # DATA APIS - Using os.getenv since dotenv is loaded above
    # ==========================================================================
    alpha_vantage_api_key: str = Field(default_factory=lambda: os.getenv("ALPHAVANTAGE_API_KEY", ""))
    # Massive.com (rebranded from Polygon.io) - API key for REST and WebSocket
    massive_api_key: str = Field(default_factory=lambda: os.getenv("PolygonIO_API_KEY", ""))
    massive_access_key: str = Field(default_factory=lambda: os.getenv("MASSIVE_ACCESS_KEY", ""))
    massive_secret_key: str = Field(default_factory=lambda: os.getenv("MASSIVE_SECRET_KEY", ""))
    massive_endpoint_url: str = Field(default_factory=lambda: os.getenv("MASSIVE_ENDPOINT_URL", "https://files.massive.com"))
    coinbase_api_key: str = Field(default_factory=lambda: os.getenv("COINBASE_API_KEY", ""))
    coinbase_api_secret: str = Field(default_factory=lambda: os.getenv("COINBASE_API_SECRET", ""))
    fred_api_key: str = Field(default_factory=lambda: os.getenv("FRED_DATA_API", os.getenv("FRED_API_KEY", "")))

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
    # FULL UNIVERSE MODE - No hardcoded tickers, pulls ALL from Massive.com
    # NO FILTERS - Get everything
    target_symbols: List[str] = Field(default_factory=list)  # Empty = use full universe
    use_full_universe: bool = Field(default=True)  # Pull ALL tickers
    train_test_split: float = Field(default=0.8)
    lookback_window: int = Field(default=60)
    time_granularity_minutes: int = Field(default=5)
    massive_lookback_hours: int = Field(default=240)
    coinbase_lookback_hours: int = Field(default=240)
    alpha_vantage_outputsize: str = Field(default="full")

    # Research ingestion / NLP
    research_paths: List[str] = Field(
        default_factory=lambda: [
            r"C:\Users\tom\Alphaloopcapital Dropbox\Tom Hogan\Archives",
            r"C:\Users\tom\Alphaloopcapital Dropbox\Tom Hogan\Current",
            r"C:\Users\tom\Alphaloopcapital Dropbox\ALC Internal Only\Agents",
        ],
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
    use_sqlite: bool = Field(default=False)  # Using Azure SQL now!

    @property
    def polygon_api_key(self) -> str:
        """Backward compatibility alias for massive_api_key."""
        return self.massive_api_key

    @property
    def polygon_lookback_hours(self) -> int:
        """Backward compatibility alias for massive_lookback_hours."""
        return self.massive_lookback_hours

    @property
    def sqlalchemy_url(self) -> str:
        if self.database_url:
            return self.database_url

        # USE SQLITE FOR NOW (faster, no network issues)
        if self.use_sqlite:
            sqlite_path = self.data_dir / "market_data.db"
            return f"sqlite:///{sqlite_path}"

        # Azure SQL
        db_name = self.sql_db.split("/")[-1] if "/" in self.sql_db else self.sql_db
        server = self.sql_server

        if not server.endswith(".database.windows.net"):
            server = f"{server}.database.windows.net"

        driver = self.db_odbc_driver.replace(" ", "+")

        if self.db_username and self.db_password:
            return f"mssql+pyodbc://{self.db_username}:{self.db_password}@{server}/{db_name}?driver={driver}"
        else:
            return f"mssql+pyodbc://@{server}/{db_name}?driver={driver}&Authentication=ActiveDirectoryInteractive"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    # Ensure critical directories exist
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    return settings
