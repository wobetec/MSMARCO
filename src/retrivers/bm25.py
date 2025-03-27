"""
Wrapper for BM25 algorithm.
"""
import nltk
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.datasets import MSMarcoDataset
from src.retrivers.retriver import Retriver


class BM25(Retriver):
    """
    BM25 algorithm for document retrieval.
    """
    def __init__(self, dataset: MSMarcoDataset, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.documents = list(dataset.documents.values())
        self.documents_ids = list(dataset.documents.keys())

        nltk.download('punkt_tab')
        nltk.download('punkt')

        self.__tokenize_docs()

        self.bm25 = BM25Okapi(self.documents, k1=k1, b=b, epsilon=epsilon)

    def __tokenize(self, text: str) -> list[str]:
        return word_tokenize(text)

    def __tokenize_docs(self):
        self.documents = [self.__tokenize(document) for document in tqdm(self.documents, desc="Tokenizing documents", unit="doc")]

    def run(self, dataset: MSMarcoDataset, query_id: str, k: int = 10, **kwargs) -> list[tuple[str, float]]:
        query = self.__tokenize(dataset.queries[query_id])
        scores = self.bm25.get_scores(query)
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_k = sorted_scores[:k]
        return [(self.documents_ids[i], float(score)) for i, score in top_k]
