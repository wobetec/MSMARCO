from abc import abstractmethod

from src.algorithms import BaseAlgorithm
from src.datasets import MSMarcoDataset


class Retriver(BaseAlgorithm):
    
    @abstractmethod
    def __init__(self, dataset: MSMarcoDataset, **kwargs):
        pass

    @abstractmethod
    def run(self, dataset: MSMarcoDataset, query_id: str, **kwargs) -> list[tuple[str, float]]:
        pass
