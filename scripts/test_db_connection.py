from loguru import logger

from src.database.connection import healthcheck


def main() -> None:
    ok = healthcheck()
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    logger.add("logs/test_db_connection.log", rotation="1 MB", level="INFO")
    main()

