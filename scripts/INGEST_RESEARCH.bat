@echo off
REM ============================================================================
REM RESEARCH DOCUMENT INGESTION
REM ============================================================================
REM This script ingests research documents into a vector store for semantic search.
REM
REM SUPPORTED FILE TYPES:
REM   .txt, .md   - Plain text and markdown
REM   .pdf        - PDF documents
REM   .docx       - Microsoft Word
REM   .csv        - Spreadsheets
REM
REM WHAT IT DOES:
REM   1. Reads all files from configured research paths
REM   2. Chunks documents into pieces
REM   3. Generates embeddings using sentence transformers
REM   4. Saves to FAISS vector store
REM
REM OUTPUTS:
REM   - Vector store: data/vectorstore/
REM   - Log: logs/ingest_research.log
REM ============================================================================

echo ============================================================================
echo RESEARCH DOCUMENT INGESTION
echo ============================================================================
echo.

cd /d "%~dp0.."

echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Starting document ingestion...
echo This will process all documents in your configured research paths.
echo.

python scripts/ingest_research.py

echo.
echo ============================================================================
echo Ingestion complete! Check logs/ingest_research.log for details.
echo Vector store saved to data/vectorstore/
echo ============================================================================

pause

