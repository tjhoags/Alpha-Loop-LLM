"""
================================================================================
RESEARCH DOCUMENT INGESTION SYSTEM
================================================================================

WHAT THIS DOES:
    1. Reads all research documents from configured paths (PDFs, TXT, MD, DOCX, CSV)
    2. Chunks documents into manageable pieces
    3. Generates embeddings using sentence transformers
    4. Stores embeddings in a FAISS vector store for fast semantic search
    
WHY IT MATTERS:
    - Agents can query research documents to inform trading decisions
    - Semantic search finds relevant info even without exact keyword matches
    - Build knowledge base from your research, reports, memos, etc.

HOW TO USE:
================================================================================
STEP 1: Configure research paths in your .env file or settings:
    
    research_paths:
      - "C:/Users/tom/OneDrive/Research"
      - "C:/Users/tom/Dropbox/Trading/Reports"
    
STEP 2: Run this script:

    Windows (PowerShell):
    ---------------------
    cd C:\\Users\\tom\\Alpha-Loop-LLM\\Alpha-Loop-LLM-1
    .\\venv\\Scripts\\activate
    python scripts/ingest_research.py
    
    Mac (Terminal):
    ---------------
    cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
    source venv/bin/activate
    python scripts/ingest_research.py

STEP 3: Check outputs:
    - data/vectorstore/index.faiss  (the vector index)
    - data/vectorstore/docs.json    (document metadata)
    - data/vectorstore/manifest.json (summary)
    - logs/ingest_research.log      (detailed log)

SUPPORTED FILE TYPES:
    .txt, .md    - Plain text and markdown
    .pdf         - PDF documents (extracts text)
    .docx        - Microsoft Word documents
    .csv         - Spreadsheets (converted to text)

EXAMPLE QUERIES AFTER INGESTION:
    from src.nlp.vectorstore import SimpleFaissStore
    store = SimpleFaissStore("data/vectorstore")
    results = store.search("what is the DCF valuation methodology")
================================================================================
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

# Setup logging
logs_dir = PROJECT_ROOT / "logs"
logs_dir.mkdir(exist_ok=True)
logger.add(logs_dir / "ingest_research.log", rotation="20 MB", level="INFO")


def main() -> None:
    """
    Main ingestion function.
    
    This will:
    1. Load all documents from configured paths
    2. Chunk them into pieces (default 1000 chars with 200 overlap)
    3. Generate embeddings using sentence-transformers
    4. Save to FAISS vector store
    """
    logger.info("=" * 70)
    logger.info("RESEARCH DOCUMENT INGESTION")
    logger.info("=" * 70)
    
    # Import here to avoid circular imports
    try:
        from src.config.settings import get_settings
        settings = get_settings()
    except ImportError:
        # Fallback to settings_new if settings doesn't have what we need
        try:
            from src.config.settings_new import get_settings
            settings = get_settings()
        except ImportError as e:
            logger.error(f"Could not import settings: {e}")
            logger.error("Make sure src/config/settings.py exists and has get_settings()")
            return
    
    # Check research paths
    research_paths = getattr(settings, 'research_paths', [])
    if not research_paths:
        logger.warning("No research_paths configured in settings!")
        logger.warning("Add research_paths to your settings or .env file")
        logger.warning("Example: research_paths = ['C:/Users/tom/Research']")
        
        # Try default locations
        default_paths = [
            Path.home() / "OneDrive" / "Research",
            Path.home() / "Documents" / "Research",
            PROJECT_ROOT / "data" / "research",
        ]
        
        for p in default_paths:
            if p.exists():
                logger.info(f"Found default research path: {p}")
                research_paths = [str(p)]
                break
        
        if not research_paths:
            logger.error("No research paths found. Please configure research_paths in settings.")
            return
    
    logger.info(f"Research paths: {research_paths}")
    
    # Check paths exist
    valid_paths = []
    for p in research_paths:
        path = Path(p)
        if path.exists():
            valid_paths.append(path)
            logger.info(f"  [OK] {path}")
        else:
            logger.warning(f"  [MISSING] {path}")
    
    if not valid_paths:
        logger.error("No valid research paths found!")
        return
    
    # Import ingestion module
    try:
        from src.nlp.ingestion import run_ingestion
        logger.info("Starting ingestion...")
        run_ingestion()
        logger.info("Ingestion complete!")
        
    except ImportError as e:
        logger.error(f"Could not import ingestion module: {e}")
        logger.error("Make sure src/nlp/ingestion.py exists")
        return
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Verify output
    vectorstore_dir = getattr(settings, 'vectorstore_dir', PROJECT_ROOT / "data" / "vectorstore")
    if Path(vectorstore_dir).exists():
        files = list(Path(vectorstore_dir).glob("*"))
        logger.info(f"Vectorstore created with {len(files)} files:")
        for f in files:
            logger.info(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    else:
        logger.warning("Vectorstore directory not found after ingestion")


if __name__ == "__main__":
    main()
