"""
Database Setup Script
Author: Tom Hogan

Initializes the PostgreSQL database with all required tables.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.core.data_logger import DataLogger

def main():
    print("=" * 60)
    print("ALC-Algo Database Setup")
    print("=" * 60)
    print()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL not set in .env file")
        print()
        print("Please add to .env:")
        print('DATABASE_URL="postgresql://user:password@localhost:5432/alc_algo"')
        return 1

    print(f"Database URL: {database_url.split('@')[1] if '@' in database_url else 'configured'}")
    print()

    print("Initializing tables...")
    logger = DataLogger()

    if logger.db_conn:
        print("✅ Database connected successfully!")
        print("✅ Tables created/verified:")
        print("   - agent_decisions")
        print("   - trades")
        print("   - portfolio_snapshots")
        print("   - market_data")
        print("   - performance_metrics")
        print()
        print("Database is ready!")
        return 0
    else:
        print("❌ Failed to connect to database")
        return 1

if __name__ == "__main__":
    sys.exit(main())
