from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SearchIndex:
    vectorizer: TfidfVectorizer
    matrix: any
    texts: List[str]


def build_index(texts: List[str]) -> SearchIndex:
    cleaned = [(t or "") for t in texts]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_df=0.98)
    matrix = vectorizer.fit_transform(cleaned)
    return SearchIndex(vectorizer=vectorizer, matrix=matrix, texts=cleaned)


def search(query: str, index: SearchIndex, top_k: int = 10) -> List[Tuple[int, float]]:
    q = (query or "").strip()
    if not q:
        return []
    q_vec = index.vectorizer.transform([q])
    sims = cosine_similarity(q_vec, index.matrix).ravel()
    if sims.size == 0:
        return []
    top_k = min(int(top_k), sims.size)
    idxs = np.argpartition(-sims, range(top_k))[:top_k]
    pairs = sorted(((int(i), float(sims[i])) for i in idxs), key=lambda x: x[1], reverse=True)
    return pairs
