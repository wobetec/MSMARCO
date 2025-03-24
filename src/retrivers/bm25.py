"""
Wrapper for BM25 algorithm.
"""
from rank_bm25 import BM25Okapi
from src.datasets import Document


class BM25:
    """
    BM25 algorithm for document retrieval.
    """

    def __init__(self, documents: 'dict[str, Document]', k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.documents = [value.tokenized_text for value in documents.values()]
        self.documents_ids = list(documents.keys())
        self.bm25 = BM25Okapi(self.documents, k1=k1, b=b, epsilon=epsilon)

    def get_top_n(self, query: 'list[str]', n: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(query)
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_n = sorted_scores[:n]
        return [(self.documents_ids[i], score) for i, score in top_n]

