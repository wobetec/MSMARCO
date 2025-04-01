from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def run(self, dataset, query_id, score_docs, **kwargs) -> list[tuple[str, float]]:
        pass
