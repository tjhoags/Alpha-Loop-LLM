"""
Ingest research documents, chunk, embed, and build a FAISS vector store.
Uses paths from settings.research_paths.
"""

from loguru import logger

from src.nlp.ingestion import run_ingestion
# Force using the working settings file implicitly via src.nlp.ingestion
# But wait, src.nlp.ingestion needs to import settings_new too.

def main() -> None:
    logger.add("logs/ingest_research.log", rotation="20 MB", level="INFO")
    run_ingestion()


if __name__ == "__main__":
    main()
