"""
Settings Module - Loads and manages configuration
Author: Tom Hogan | Alpha Loop Capital, LLC
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class Settings:
    """
    Centralized configuration management for ALC-Algo.
    Loads environment variables from master_alc_env file.
    """
    
    def __init__(self, env_file_path: Optional[str] = None):
        """
        Initialize settings.
        
        Args:
            env_file_path: Path to master_alc_env file. If None, tries to load from secrets.py
        """
        self._env_vars: Dict[str, str] = {}
        self._load_environment(env_file_path)
        
    def _load_environment(self, env_file_path: Optional[str] = None):
        """Load environment variables from master_alc_env file."""
        # Try to import secrets.py if it exists
        if env_file_path is None:
            try:
                from .secrets import ENV_FILE_PATH
                env_file_path = ENV_FILE_PATH
            except ImportError:
                # Fall back to checking common locations
                possible_paths = [
                    "C:/Users/tom/Alphaloopcapital Dropbox/master_alc_env",
                    "C:/Users/tom/Dropbox/master_alc_env",
                    os.path.join(os.path.dirname(__file__), "master_alc_env"),
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        env_file_path = path
                        break
        
        if env_file_path and os.path.exists(env_file_path):
            # Load the environment file
            load_dotenv(env_file_path)
            
            # Read the file and parse it
            with open(env_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        self._env_vars[key.strip()] = value.strip().strip('"').strip("'")
        else:
            print(f"Warning: Environment file not found at {env_file_path}")
            print("Some API features may not work without proper configuration.")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get an environment variable.
        
        Args:
            key: Environment variable key
            default: Default value if key not found
            
        Returns:
            Value of the environment variable
        """
        # Try internal cache first
        value = self._env_vars.get(key)
        if value is not None:
            return value
        
        # Try OS environment
        value = os.getenv(key)
        if value is not None:
            return value
            
        return default
    
    def get_required(self, key: str) -> str:
        """
        Get a required environment variable. Raises error if not found.
        
        Args:
            key: Environment variable key
            
        Returns:
            Value of the environment variable
            
        Raises:
            ValueError: If key is not found
        """
        value = self.get(key)
        if value is None:
            raise ValueError(f"Required environment variable '{key}' not found")
        return value
    
    # Google APIs
    @property
    def google_api_key_1(self) -> Optional[str]:
        return self.get('GOOGLE_API_KEY_1')
    
    @property
    def google_api_key_2(self) -> Optional[str]:
        return self.get('GOOGLE_API_KEY_2')
    
    @property
    def google_api_key_3(self) -> Optional[str]:
        return self.get('GOOGLE_API_KEY_3')
    
    @property
    def google_vertex_project_id(self) -> Optional[str]:
        return self.get('GOOGLE_VERTEX_PROJECT_ID')
    
    # Coinbase
    @property
    def coinbase_api_key(self) -> Optional[str]:
        return self.get('COINBASE_API_KEY')
    
    @property
    def coinbase_api_secret(self) -> Optional[str]:
        return self.get('COINBASE_API_SECRET')
    
    # Alpha Vantage
    @property
    def alpha_vantage_api_key(self) -> Optional[str]:
        return self.get('ALPHA_VANTAGE_API_KEY')
    
    # Fiscal.ai
    @property
    def fiscal_ai_api_key(self) -> Optional[str]:
        return self.get('FISCAL_AI_API_KEY')
    
    # IBKR
    @property
    def ibkr_account_id(self) -> Optional[str]:
        return self.get('IBKR_ACCOUNT_ID')
    
    @property
    def ibkr_host(self) -> str:
        return self.get('IBKR_HOST', '127.0.0.1')
    
    @property
    def ibkr_port(self) -> int:
        try:
            return int(self.get('IBKR_PORT', '7497'))
        except (ValueError, TypeError):
            print(f"Warning: Invalid IBKR_PORT '{self.get('IBKR_PORT')}'. Defaulting to 7497.")
            return 7497  # Default to paper trading
    
    # Slack
    @property
    def slack_webhook_url(self) -> Optional[str]:
        return self.get('SLACK_WEBHOOK_URL')
    
    @property
    def slack_bot_token(self) -> Optional[str]:
        return self.get('SLACK_BOT_TOKEN')
    
    # Notion
    @property
    def notion_api_key(self) -> Optional[str]:
        return self.get('NOTION_API_KEY')
    
    # Dropbox
    @property
    def dropbox_access_token(self) -> Optional[str]:
        return self.get('DROPBOX_ACCESS_TOKEN')
    
    # Anthropic
    @property
    def anthropic_api_key(self) -> Optional[str]:
        return self.get('ANTHROPIC_API_KEY')
    
    # OpenAI
    @property
    def openai_api_key(self) -> Optional[str]:
        return self.get('OPENAI_API_KEY')
    
    # Perplexity
    @property
    def perplexity_api_key(self) -> Optional[str]:
        return self.get('PERPLEXITY_API_KEY')
    
    # Super.myninja
    @property
    def myninja_api_key(self) -> Optional[str]:
        return self.get('MYNINJA_API_KEY')


# Global settings instance
settings = Settings()

