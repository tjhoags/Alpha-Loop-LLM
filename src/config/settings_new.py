from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import Field, validator
from pydantic_settings import BaseSettings


# Ensure .env is loaded early
load_dotenv()


class Settings(BaseSettings):
    # ==========================================================================
    # DATABASE - Azure SQL Server
    # ==========================================================================
    database_url: Optional[str] = Field(
        default=None, description="Full SQLAlchemy URL. If set, overrides components below."
    )
    # Maps to your .env: SQL_SERVER=alc-sql-server, SQL_DB=alc-sql-server/alc_market_data
    sql_server: str = Field(default="alc-sql-server.database.windows.net", env="SQL_SERVER")
    sql_db: str = Field(default="alc_market_data", env="SQL_DB")
    db_username: str = Field(default="", env="DB_USERNAME")
    db_password: str = Field(default="", env="DB_PASSWORD")
    db_odbc_driver: str = Field(
        default="ODBC Driver 17 for SQL Server", env="DB_ODBC_DRIVER"
    )

    # ==========================================================================
    # DATA APIS
    # ==========================================================================
    alpha_vantage_api_key: str = Field(default="", env="ALPHA_VANTAGE_API_KEY")
    polygon_api_key: str = Field(default="", env="POLYGON_API_KEY")
    massive_access_key: str = Field("", env="MASSIVE_ACCESS_KEY")
    massive_secret_key: str = Field("", env="MASSIVE_SECRET_KEY")
    massive_endpoint_url: str = Field("https://files.massive.com", env="MASSIVE_ENDPOINT_URL")
    coinbase_api_key: str = Field(default="", env="COINBASE_API_KEY")
    fred_api_key: str = Field(default="", env="FRED_API_KEY")

    # ==========================================================================
    # AI SERVICES
    # ==========================================================================
    openai_api_key: str = Field(default="", env="OPENAI_SECRET")
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    perplexity_api_key: str = Field(default="", env="PERPLEXITY_API_KEY")
    google_api_key: str = Field(default="", env="API_KEY")

    # ==========================================================================
    # TRADING / IBKR
    # ==========================================================================
    ibkr_host: str = Field(default="127.0.0.1", env="IBKR_HOST")
    ibkr_port: int = Field(default=7497, env="IBKR_PORT")
    ibkr_client_id: int = Field(default=1, env="IBKR_CLIENT_ID")

    # Paths
    base_dir: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = base_dir / "data"
    models_dir: Path = base_dir / "models"
    logs_dir: Path = base_dir / "logs"
    vectorstore_dir: Path = base_dir / "vectorstore"

    # ==========================================================================
    # DATA / ML CONFIG
    # ==========================================================================
    target_symbols: List[str] = Field(
        default_factory=lambda: [
            "SPY", "QQQ", "IWM", "DIA",
            "AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "META", "TSLA", "AMZN",
            "BTC-USD", "ETH-USD"
        ]
    )
    use_full_universe: bool = Field(default=True, env="USE_FULL_UNIVERSE")

    train_test_split: float = Field(default=0.8)
    lookback_window: int = Field(default=60)
    time_granularity_minutes: int = Field(default=5)
    polygon_lookback_hours: int = Field(default=240)
    coinbase_lookback_hours: int = Field(default=240)
    alpha_vantage_outputsize: str = Field(default="full")
    
    # Research ingestion
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

    # Risk
    max_daily_loss_pct: float = Field(default=0.02)
    max_drawdown_pct: float = Field(default=0.05)
    max_position_size_pct: float = Field(default=0.10)
    max_positions: int = Field(default=10)
    kelly_fraction_cap: float = Field(default=0.25)

    # Grading
    min_auc: float = Field(default=0.52)
    min_accuracy: float = Field(default=0.52)
    min_sharpe_ratio: float = Field(default=1.5)
    max_validation_drawdown: float = Field(default=0.05)

    log_level: str = Field(default="INFO")

    @validator("train_test_split")
    def validate_split(cls, v: float) -> float:
        if not 0.5 < v < 0.95:
            raise ValueError("train_test_split should be between 0.5 and 0.95")
        return v

    @property
    def sqlalchemy_url(self) -> str:
        if self.database_url:
            return self.database_url
        db_name = self.sql_db.split("/")[-1] if "/" in self.sql_db else self.sql_db
        return (
            "mssql+pyodbc://{user}:{pwd}@{server}/{db}"
            "?driver={driver}".format(
                user=self.db_username,
                pwd=self.db_password,
                server=self.sql_server,
                db=db_name,
                driver=self.db_odbc_driver.replace(' ', '+'),
            )
        )

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore" # Critical fix


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    # Ensure critical directories exist
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    return settings

