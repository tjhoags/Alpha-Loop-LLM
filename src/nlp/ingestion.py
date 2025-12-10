import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from loguru import logger
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

try:
    from src.config.settings import get_settings
except ImportError:
    from src.config.settings_new import get_settings
from src.nlp.vectorstore import SimpleFaissStore

try:
    import docx  # type: ignore
except ImportError:  # pragma: no cover - optional dependency guard
    docx = None


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages)


def _read_docx(path: Path) -> str:
    if docx is None:
        raise ImportError("python-docx not installed")
    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


def _read_csv(path: Path, max_rows: int = 1000) -> str:
    df = pd.read_csv(path, nrows=max_rows)
    return df.to_csv(index=False)


def _chunk_text(text: str, max_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_size)
        chunks.append(text[start:end])
        start += max_size - overlap
    return [c for c in chunks if c.strip()]


def _iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    for root in paths:
        if not root.exists():
            logger.warning(f"Path does not exist, skipping: {root}")
            continue
        if root.is_file():
            yield root
        else:
            for p in root.rglob("*"):
                if p.is_file():
                    yield p


def load_and_chunk_documents() -> List[Dict]:
    settings = get_settings()
    max_size = settings.max_chunk_size
    overlap = settings.chunk_overlap

    raw_paths = settings.research_paths
    paths = [Path(p) for p in raw_paths]

    supported_suffixes = {".txt", ".md", ".pdf", ".docx", ".csv"}
    docs: List[Dict] = []

    for path in _iter_files(paths):
        if path.suffix.lower() not in supported_suffixes:
            continue
        try:
            if path.suffix.lower() in {".txt", ".md"}:
                text = _read_text_file(path)
            elif path.suffix.lower() == ".pdf":
                text = _read_pdf(path)
            elif path.suffix.lower() == ".docx":
                text = _read_docx(path)
            elif path.suffix.lower() == ".csv":
                text = _read_csv(path)
            else:
                continue
            chunks = _chunk_text(text, max_size, overlap)
            for i, chunk in enumerate(chunks):
                docs.append(
                    {
                        "text": chunk,
                        "source_path": str(path),
                        "chunk_id": i,
                        "suffix": path.suffix.lower(),
                    },
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to read {path}: {exc}")
    logger.info(f"Prepared {len(docs)} text chunks from research corpus.")
    return docs


def embed_documents(docs: List[Dict]) -> np.ndarray:
    if not docs:
        return np.array([])
    settings = get_settings()
    model_name = settings.embedding_model_name
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings


def build_vectorstore(docs: List[Dict], embeddings: np.ndarray) -> None:
    if len(docs) != len(embeddings):
        raise ValueError("Docs and embeddings length mismatch")
    settings = get_settings()
    store = SimpleFaissStore(settings.vectorstore_dir)
    store.save(embeddings.astype(np.float32), docs)
    # Save a small manifest for transparency
    manifest_path = settings.vectorstore_dir / "manifest.json"
    summary = {
        "count": len(docs),
        "embedding_model": settings.embedding_model_name,
        "source_paths": settings.research_paths,
    }
    manifest_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Vectorstore manifest written to {manifest_path}")


def run_ingestion() -> None:
    docs = load_and_chunk_documents()
    if not docs:
        logger.warning("No documents found to ingest.")
        return
    embeddings = embed_documents(docs)
    build_vectorstore(docs, embeddings)
    logger.info("Research ingestion complete.")


if __name__ == "__main__":
    run_ingestion()

