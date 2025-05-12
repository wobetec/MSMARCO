import json
import os
import sys

import requests
from tqdm import tqdm

ROOT_DIR = '../../'
sys.path.append(ROOT_DIR)
from src.datasets import MSMarcoDataset
from experiments import load_datasets, DEFAULT_DATA_FOLDER


def feed_with_copy(dataset: MSMarcoDataset):
    def create_json(dataset: MSMarcoDataset) -> list[dict]:
        docs = []
        for doc_id, text in tqdm(dataset.documents.items(), desc="Creating JSON"):
            docs.append({
                "put": f"id:msmarco:msmarco::{doc_id}",
                "fields": {
                    "content": text
                }
            })
        return docs

    docs = create_json(dataset)
    temp_json = 'docs.json'
    with open(temp_json, 'w', encoding='utf-8') as f:
        json.dump(docs, f)
    
    os.system(f"docker cp {temp_json} vespa:docs.json")
    os.system("docker exec vespa vespa feed --progress 5 -t http://localhost:8080 docs.json")

    os.remove(temp_json)


def feed_with_api(dataset: MSMarcoDataset):
    VESPA_ENDPOINT = "http://localhost:8080/document/v1/msmarco/msmarco/docid/"
    for doc_id, text in tqdm(dataset.documents.items(), desc="Feeding Vespa"):
        uri  = VESPA_ENDPOINT + doc_id
        body = {"fields": {"content": text}}
        r = requests.post(uri,
                          headers={"Content-Type": "application/json"},
                          data=json.dumps(body))
        if r.status_code != 200:
            print(f"⚠️ Erro no doc {doc_id}: {r.status_code} {r.text}")


if __name__ == "__main__":

    if len(sys.argv) == 1:
        print("Please provide the method name [copy, api].")
        sys.exit(1)

    if len(sys.argv) == 2:
        print("Please provide the dataset name [none, lower, full].")
        sys.exit(1)

    dataset_name = sys.argv[2]

    dataset: MSMarcoDataset = load_datasets(data_folder=os.path.join(ROOT_DIR, DEFAULT_DATA_FOLDER))[dataset_name]

    if sys.argv[1] == "copy":
        feed_with_copy(dataset)
    elif sys.argv[1] == "api":
        feed_with_api(dataset)
