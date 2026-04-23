"""
RAG Pipeline for AutoStream Agent
Handles knowledge base loading, embedding, and retrieval using FAISS + sentence-transformers.
"""

import json
import os
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class RAGPipeline:
    """
    Local RAG pipeline using FAISS vector store and sentence-transformers embeddings.
    Indexes the AutoStream knowledge base at startup and provides top-k retrieval.
    """

    def __init__(self, kb_dir: str = None, model_name: str = "all-MiniLM-L6-v2"):
        if kb_dir is None:
            kb_dir = Path(__file__).parent
        self.kb_dir = Path(kb_dir)
        self.model = SentenceTransformer(model_name)
        self.chunks: List[str] = []
        self.index: faiss.IndexFlatL2 = None
        self._build_index()

    # ------------------------------------------------------------------ #
    #  Index building                                                       #
    # ------------------------------------------------------------------ #

    def _load_knowledge_base(self) -> List[str]:
        """Load and chunk all KB files into text passages."""
        passages = []

        # Load JSON pricing file
        pricing_path = self.kb_dir / "knowledge_base_pricing.json"
        if pricing_path.exists():
            with open(pricing_path, "r") as f:
                data = json.load(f)
            passages.extend(self._chunk_pricing(data))

        # Load Markdown files
        for md_file in self.kb_dir.glob("knowledge_base_*.md"):
            with open(md_file, "r") as f:
                content = f.read()
            passages.extend(self._chunk_markdown(content))

        return passages

    def _chunk_pricing(self, data: dict) -> List[str]:
        """Convert pricing JSON into readable text chunks."""
        chunks = []
        product = data.get("product", "AutoStream")
        description = data.get("description", "")
        chunks.append(f"{product}: {description}")

        for plan in data.get("plans", []):
            features_str = "; ".join(plan.get("features", []))
            chunk = (
                f"{plan['name']}: {plan['price']}. "
                f"Features: {features_str}. "
                f"Best for: {plan.get('best_for', '')}."
            )
            chunks.append(chunk)

        for key, value in data.get("comparison", {}).items():
            chunks.append(f"Comparison ({key}): {value}")

        return chunks

    def _chunk_markdown(self, content: str, chunk_size: int = 300) -> List[str]:
        """Split markdown into overlapping text chunks."""
        # Split by headers first, then by size
        sections = []
        current_section = []

        for line in content.split("\n"):
            if line.startswith("#") and current_section:
                sections.append("\n".join(current_section).strip())
                current_section = [line]
            else:
                current_section.append(line)

        if current_section:
            sections.append("\n".join(current_section).strip())

        # Further split large sections
        chunks = []
        for section in sections:
            if not section.strip():
                continue
            words = section.split()
            if len(words) <= chunk_size // 5:
                chunks.append(section)
            else:
                # Sliding window with overlap
                step = chunk_size // 6
                for i in range(0, len(words), step):
                    chunk = " ".join(words[i: i + chunk_size // 5])
                    if chunk.strip():
                        chunks.append(chunk)

        return [c for c in chunks if len(c.strip()) > 20]

    def _build_index(self):
        """Embed all chunks and build FAISS index."""
        print("[RAG] Loading knowledge base...")
        self.chunks = self._load_knowledge_base()

        if not self.chunks:
            raise ValueError("Knowledge base is empty. Check the knowledge_base/ directory.")

        print(f"[RAG] Embedding {len(self.chunks)} chunks...")
        embeddings = self.model.encode(self.chunks, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")

        # Normalize for cosine similarity via L2
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product on normalized = cosine
        self.index.add(embeddings)

        print(f"[RAG] Index built — {self.index.ntotal} vectors, dim={dim}")

    # ------------------------------------------------------------------ #
    #  Retrieval                                                            #
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Retrieve top-k relevant passages for a query.
        Returns list of (passage, score) tuples.
        """
        query_embedding = self.model.encode([query], show_progress_bar=False)
        query_embedding = np.array(query_embedding, dtype="float32")
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks) and score > 0.1:
                results.append((self.chunks[idx], float(score)))

        return results

    def get_context(self, query: str, top_k: int = 3) -> str:
        """
        Return retrieved passages as a formatted context string
        ready to inject into an LLM prompt.
        """
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return "No relevant information found in the knowledge base."

        context_parts = []
        for i, (passage, score) in enumerate(results, 1):
            context_parts.append(f"[Source {i}] {passage}")

        return "\n\n".join(context_parts)
