import requests
from src.datasets import MSMarcoDataset
from src.retrivers.retriver import Retriver


class VespaBM25(Retriver):
    def __init__(self, endpoint="http://localhost:8080/search/"):
        self.endpoint = endpoint

    def run(self, dataset: MSMarcoDataset, query_id: str, k: int = 10, **kwargs) -> list[tuple[str, float]]:
        query = dataset.queries[query_id]
        params = {
            "yql": f"select * from msmarco where userQuery();",
            "query": query,
            "hits": k,
            "default-index": 'content'
        }
        r = requests.get(self.endpoint, params=params)
        hits = []
        for hit in r.json()["root"]["children"]:
            docid = hit["id"].replace('id:msmarco:msmarco::', '')
            score = hit["relevance"]
            hits.append((docid, score))
        return hits
