"""
Environment Variable Template
Author: Tom Hogan | Alpha Loop Capital, LLC

This file lists all required API keys and credentials.
Copy this to config/secrets.py and fill in your actual values.
"""

# Path to master_alc_env file (UPDATE THIS)
ENV_FILE_PATH = "C:/Users/tom/Alphaloopcapital Dropbox/master_alc_env"

# Expected environment keys in master_alc_env file
ENV_KEYS = {
    # Google APIs (3 keys)
    'GOOGLE_API_KEY_1': 'Primary Google Cloud API Key',
    'GOOGLE_API_KEY_2': 'Secondary Google Cloud API Key',
    'GOOGLE_API_KEY_3': 'Tertiary Google Cloud API Key',
    'GOOGLE_VERTEX_PROJECT_ID': 'Google Vertex AI Project ID',
    'GOOGLE_APPLICATION_CREDENTIALS': 'Path to Google service account JSON',
    
    # Coinbase
    'COINBASE_API_KEY': 'Coinbase API Key',
    'COINBASE_API_SECRET': 'Coinbase API Secret',
    
    # Alpha Vantage
    'ALPHA_VANTAGE_API_KEY': 'Alpha Vantage API Key',
    
    # Fiscal.ai
    'FISCAL_AI_API_KEY': 'Fiscal.ai API Key',
    
    # Finviz
    'FINVIZ_USERNAME': 'Finviz Username',
    'FINVIZ_PASSWORD': 'Finviz Password',
    
    # Interactive Brokers (IBKR)
    'IBKR_ACCOUNT_ID': 'IBKR Account ID (Live: 7496, Paper: 7497)',
    'IBKR_HOST': 'IBKR Gateway Host (default: 127.0.0.1)',
    'IBKR_PORT': 'IBKR Gateway Port (Paper: 7497, Live: 7496)',
    
    # Slack
    'SLACK_WEBHOOK_URL': 'Slack Webhook URL for notifications',
    'SLACK_BOT_TOKEN': 'Slack Bot Token',
    
    # Notion AI
    'NOTION_API_KEY': 'Notion Integration Token',
    'NOTION_DATABASE_ID': 'Notion Database ID for logging',
    
    # Dropbox
    'DROPBOX_ACCESS_TOKEN': 'Dropbox Access Token',
    'DROPBOX_REFRESH_TOKEN': 'Dropbox Refresh Token',
    
    # Claude (Anthropic)
    'ANTHROPIC_API_KEY': 'Anthropic Claude API Key',
    
    # OpenAI
    'OPENAI_API_KEY': 'OpenAI API Key',
    'OPENAI_ORG_ID': 'OpenAI Organization ID',
    
    # Perplexity
    'PERPLEXITY_API_KEY': 'Perplexity API Key',
    
    # Super.myninja AI
    'MYNINJA_API_KEY': 'Super.myninja AI API Key',
    'MYNINJA_API_URL': 'Super.myninja AI API URL',
}

# Trading Configuration
TRADING_CONFIG = {
    'PAPER_ACCOUNT': '7497',
    'LIVE_ACCOUNT': '7496',
    'DEFAULT_MODE': 'PAPER',  # Always start with paper trading
    'MARGIN_OF_SAFETY': 0.30,  # 30% required margin of safety
    'MAX_POSITION_SIZE': 0.10,  # 10% max position size
    'MAX_PORTFOLIO_HEAT': 0.20,  # 20% max portfolio heat
}

# HoagsAgent ML Protocols
ML_PROTOCOLS = {
    'VERTEX_AI': True,
    'GPT4': True,
    'CLAUDE': True,
    'GEMINI': True,
    'PERPLEXITY': True,
    'CUSTOM_FINETUNED': False,  # Enable when custom models are ready
}

# Attribution Settings
ATTRIBUTION = {
    'AUTHOR': 'Tom Hogan',
    'ORGANIZATION': 'Alpha Loop Capital, LLC',
    'DCF_MODEL_NAME': 'HOGAN MODEL',
    'NEVER_CREDIT_AI': True,
}

# Audit Configuration
AUDIT_CONFIG = {
    'LOG_ALL_ACTIONS': True,
    'USER_IDS': ['TJH', 'RT'],
    'RETENTION_DAYS': 2555,  # 7 years
    'SENSITIVE_DATA_HASHING': True,
}

