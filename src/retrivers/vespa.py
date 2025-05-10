import requests

class VespaRetriever:
    def __init__(self, endpoint="http://localhost:8080/search/"):
        self.endpoint = endpoint

    def run(self, dataset, query_id, k=10, **kwargs):
        query = dataset.queries[query_id]
        params = {
            "yql":             f"select * from sources msmarco where userInput('{query}');",
            "hits":            k,
            "ranking.profile": "bm25-profile"
        }
        r = requests.get(self.endpoint, params=params)
        hits = []
        for hit in r.json()["root"]["children"]:
            docid = hit["id"]
            score = hit["relevance"]
            hits.append((docid, score))
        return hits
