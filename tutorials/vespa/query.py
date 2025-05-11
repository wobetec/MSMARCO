import pandas as pd
import requests
import sys

def query_vespa(query: str) -> list[tuple[str, float]]:
    VESPA_ENDPOINT = "http://localhost:8080/search/"
    params = {
        "yql": f"select * from msmarco where userQuery();",
        "query": query,
        "hits": 10,
        "default-index": 'content'
    }
    r = requests.get(VESPA_ENDPOINT, params=params)
    hits = []
    for hit in r.json()["root"]["children"]:
        docid = hit["id"].replace('id:msmarco:msmarco::', '')
        score = hit["relevance"]
        hits.append((docid, score))
    return hits

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Please provide the query.")
        sys.exit(1)

    query = ' '.join(sys.argv[1:])
    hits = query_vespa(query)
    df = pd.DataFrame(hits, columns=["docid", "score"])
    print(df)
    