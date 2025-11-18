"""Simple TF-IDF retriever for the small docs corpus.

Returns chunks in format:
{
    "chunk_id": "marketing_calendar::chunk0",
    "source": "marketing_calendar",
    "content": "..."
}

Returns top-k by cosine similarity.
"""

from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import json


class Retriever:
    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = docs_dir
        self.chunks: List[Dict[str, Any]] = []
        self._vectorizer = None
        self._matrix = None
        self._build()

    def _build(self):
        chunks = []
        for fname in os.listdir(self.docs_dir):
            path = os.path.join(self.docs_dir, fname)
            if not fname.endswith(".md"):
                continue
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                # naive chunking by paragraphs
                paras = [p.strip() for p in text.split("\n\n") if p.strip()]
                for i, p in enumerate(paras):
                    chunk_id = f"{os.path.splitext(fname)[0]}::chunk{i}"
                    chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "source": os.path.splitext(fname)[0],
                            "content": p,
                        }
                    )
        self.chunks = chunks
        texts = [c["content"] for c in chunks]
        if texts:
            self._vectorizer = TfidfVectorizer(stop_words="english")
            self._matrix = self._vectorizer.fit_transform(texts)

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        if self._matrix is None or not self.chunks:
            return []
        qv = self._vectorizer.transform([query])
        sims = cosine_similarity(qv, self._matrix)[0]
        idxs = sims.argsort()[::-1][:k]
        results = []
        for i in idxs:
            results.append(
                {
                    "chunk_id": self.chunks[i]["chunk_id"],
                    "source": self.chunks[i]["source"],
                    "content": self.chunks[i]["content"],
                    "score": float(sims[i]),
                }
            )
        return results


if __name__ == "__main__":
    r = Retriever("docs")
    print(r.retrieve("return window beverages", k=3))
