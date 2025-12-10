from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from loguru import logger


class SimpleFaissStore:
    """
    Lightweight FAISS-backed vector store.
    Stores normalized embeddings and parallel metadata list.
    """

    def __init__(self, dir_path: Path):
        self.dir_path = dir_path
        self.index_path = dir_path / "index.faiss"
        self.meta_path = dir_path / "metadata.npy"
        self.index = None
        self.metadata: List[Dict] = []

    def _ensure_dir(self) -> None:
        self.dir_path.mkdir(parents=True, exist_ok=True)

    def save(self, embeddings: np.ndarray, metadata: List[Dict]) -> None:
        self._ensure_dir()
        if embeddings.ndim != 2:
            raise ValueError("Embeddings should be 2D array [n, dim]")
        if embeddings.shape[0] != len(metadata):
            raise ValueError("Embeddings and metadata length mismatch")

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        normed = embeddings / norms

        index = faiss.IndexFlatIP(normed.shape[1])
        index.add(normed.astype(np.float32))
        faiss.write_index(index, str(self.index_path))
        np.save(self.meta_path, np.array(metadata, dtype=object), allow_pickle=True)
        logger.info(f"Saved FAISS index with {len(metadata)} vectors to {self.index_path}")

    def load(self) -> None:
        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError("Vector store not initialized. Run ingestion first.")
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = list(np.load(self.meta_path, allow_pickle=True))
        logger.info(f"Loaded FAISS index with {len(self.metadata)} vectors.")

    def search(self, query_embeddings: np.ndarray, top_k: int = 5) -> List[List[Tuple[float, Dict]]]:
        if self.index is None:
            self.load()
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-12
        query_normed = query_embeddings / norms
        scores, idx = self.index.search(query_normed.astype(np.float32), top_k)
        results: List[List[Tuple[float, Dict]]] = []
        for row_scores, row_idx in zip(scores, idx):
            row_res = []
            for s, i in zip(row_scores, row_idx):
                if i == -1:
                    continue
                row_res.append((float(s), self.metadata[i]))
            results.append(row_res)
        return results

