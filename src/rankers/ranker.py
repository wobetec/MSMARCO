from abc import abstractmethod

from src.algorithms import BaseAlgorithm
from src.datasets import MSMarcoDataset


class Ranker(BaseAlgorithm):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def run(self, dataset: MSMarcoDataset, query_id: str, score_docs: list[tuple[str, float]], **kwargs) -> list[tuple[str, float]]:
        pass
