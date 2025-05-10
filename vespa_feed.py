import json
import requests
from tqdm import tqdm
from src.datasets import MSMarcoDataset

dataset = MSMarcoDataset("data/subset_msmarco_train_0")
dataset.load_data("subset_msmarco_train_0.01_99.pkl")

VESPA_ENDPOINT = "http://localhost:8080/document/v1/msmarco/msmarco/docid/"

for doc_id, text in tqdm(dataset.documents.items(), desc="Feeding Vespa"):
    uri  = VESPA_ENDPOINT + doc_id
    body = {"fields": {"content": text}}
    r = requests.post(uri,
                      headers={"Content-Type": "application/json"},
                      data=json.dumps(body))
    if r.status_code != 200:
        print(f"⚠️ Erro no doc {doc_id}: {r.status_code} {r.text}")
