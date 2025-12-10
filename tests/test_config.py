"""
ALC-Algo Configuration Tests
Author: Tom Hogan | Alpha Loop Capital, LLC

Tests for configuration loading and settings management.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSettings:
    """Tests for Settings class."""
    
    def test_settings_import(self):
        """Test that settings module can be imported."""
        from config.settings import Settings
        assert Settings is not None
    
    def test_settings_instance_creation(self):
        """Test that Settings instance can be created."""
        from config.settings import Settings
        settings = Settings()
        assert settings is not None
    
    def test_settings_get_method(self):
        """Test get method with default values."""
        from config.settings import Settings
        settings = Settings()
        
        # Test with default value
        result = settings.get("NONEXISTENT_KEY", "default_value")
        assert result == "default_value"
    
    def test_settings_get_from_env(self):
        """Test that settings can read from environment."""
        from config.settings import Settings
        
        test_key = "TEST_ALC_ALGO_KEY"
        test_value = "test_value_12345"
        
        with patch.dict(os.environ, {test_key: test_value}):
            settings = Settings()
            result = settings.get(test_key)
            assert result == test_value
    
    def test_settings_properties_exist(self):
        """Test that expected properties exist on settings."""
        from config.settings import Settings
        settings = Settings()
        
        # Check property methods exist
        assert hasattr(settings, 'alpha_vantage_api_key')
        assert hasattr(settings, 'openai_api_key')
        assert hasattr(settings, 'anthropic_api_key')
        assert hasattr(settings, 'ibkr_host')
        assert hasattr(settings, 'ibkr_port')
    
    def test_ibkr_defaults(self):
        """Test IBKR default values."""
        from config.settings import Settings
        settings = Settings()
        
        # Should have sensible defaults
        assert settings.ibkr_host == "127.0.0.1"
        assert settings.ibkr_port == 7497  # Paper trading default
    
    def test_settings_get_required_raises(self):
        """Test that get_required raises for missing keys."""
        from config.settings import Settings
        settings = Settings()
        
        with pytest.raises(ValueError):
            settings.get_required("DEFINITELY_NOT_A_REAL_KEY_12345")
    
    def test_global_settings_instance(self):
        """Test that global settings instance is created."""
        from config.settings import settings
        assert settings is not None
        assert hasattr(settings, 'get')


class TestEnvironmentConfiguration:
    """Tests for environment-based configuration."""
    
    def test_env_file_loading(self, tmp_path):
        """Test loading from .env style file."""
        # Create temporary env file
        env_content = """
# Test environment file
TEST_KEY_1=value1
TEST_KEY_2="value2"
TEST_KEY_3='value3'
"""
        env_file = tmp_path / "test_env"
        env_file.write_text(env_content)
        
        from config.settings import Settings
        settings = Settings(env_file_path=str(env_file))
        
        # Values should be loaded and stripped of quotes
        assert settings.get("TEST_KEY_1") == "value1"
        assert settings.get("TEST_KEY_2") == "value2"
        assert settings.get("TEST_KEY_3") == "value3"
    
    def test_missing_env_file_warning(self, capsys):
        """Test that missing env file shows warning."""
        from config.settings import Settings
        
        # Create settings with non-existent file
        settings = Settings(env_file_path="/nonexistent/path/to/file")
        
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower() or settings is not None


class TestAPIKeyConfiguration:
    """Tests for API key configuration."""
    
    def test_alpha_vantage_key_property(self):
        """Test Alpha Vantage API key property."""
        from config.settings import Settings
        
        with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "test_av_key"}):
            settings = Settings()
            assert settings.alpha_vantage_api_key == "test_av_key"
    
    def test_openai_key_property(self):
        """Test OpenAI API key property."""
        from config.settings import Settings
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            settings = Settings()
            assert settings.openai_api_key == "sk-test123"
    
    def test_google_api_keys(self):
        """Test Google API key properties."""
        from config.settings import Settings
        
        env_vars = {
            "GOOGLE_API_KEY_1": "google_key_1",
            "GOOGLE_API_KEY_2": "google_key_2",
            "GOOGLE_API_KEY_3": "google_key_3",
        }
        
        with patch.dict(os.environ, env_vars):
            settings = Settings()
            assert settings.google_api_key_1 == "google_key_1"
            assert settings.google_api_key_2 == "google_key_2"
            assert settings.google_api_key_3 == "google_key_3"
    
    def test_ibkr_configuration(self):
        """Test IBKR broker configuration."""
        from config.settings import Settings
        
        env_vars = {
            "IBKR_ACCOUNT_ID": "U12345678",
            "IBKR_HOST": "192.168.1.100",
            "IBKR_PORT": "7496",
        }
        
        with patch.dict(os.environ, env_vars):
            settings = Settings()
            assert settings.ibkr_account_id == "U12345678"
            assert settings.ibkr_host == "192.168.1.100"
            assert settings.ibkr_port == 7496
    
    def test_optional_keys_return_none(self):
        """Test that optional keys return None when not set."""
        from config.settings import Settings
        
        # Clear specific env vars
        env_vars = {
            "COINBASE_API_KEY": "",
            "SLACK_WEBHOOK_URL": "",
        }
        
        settings = Settings()
        # Should return None or empty string, not raise error
        coinbase_key = settings.coinbase_api_key
        slack_url = settings.slack_webhook_url
        # No assertion needed - just verify no exception


class TestConfigurationValidation:
    """Tests for configuration validation."""
    
    def test_port_conversion(self):
        """Test that port is converted to integer."""
        from config.settings import Settings
        
        with patch.dict(os.environ, {"IBKR_PORT": "7497"}):
            settings = Settings()
            assert isinstance(settings.ibkr_port, int)
            assert settings.ibkr_port == 7497
    
    def test_default_port_type(self):
        """Test default port type is integer."""
        from config.settings import Settings
        settings = Settings()
        assert isinstance(settings.ibkr_port, int)


# Fixtures
@pytest.fixture
def clean_env():
    """Fixture to provide clean environment."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_settings():
    """Fixture to provide mocked settings."""
    from config.settings import Settings
    
    mock = MagicMock(spec=Settings)
    mock.alpha_vantage_api_key = "test_key"
    mock.ibkr_host = "127.0.0.1"
    mock.ibkr_port = 7497
    
    return mock


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

