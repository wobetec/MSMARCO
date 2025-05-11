"""
Wrapper for BM25 algorithm.
"""
from tqdm import tqdm
import json
import os
import subprocess

import pandas as pd
from fastbm25 import fastbm25
from nltk.tokenize import word_tokenize
from pyserini.search.lucene import LuceneSearcher
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
        tokenized_docs = [self.__tokenizer(doc) for doc in tqdm(self.documents)]
        self.bm25 = BM25Okapi(tokenized_docs, k1=k1, b=b, epsilon=epsilon)

    def __tokenizer(self, text: str) -> list[str]:
        return word_tokenize(text)

    def run(self, dataset: MSMarcoDataset, query_id: str, k: int = 10, **kwargs) -> list[tuple[str, float]]:
        query = self.__tokenizer(dataset.queries[query_id])
        scores = self.bm25.get_scores(query)
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_k = sorted_scores[:k]
        return [(self.documents_ids[i], float(score)) for i, score in top_k]


class BM25Fast(Retriver):
    """
    BM25 algorithm for document retrieval using FastBM25.
    """
    def __init__(self, dataset: MSMarcoDataset):
        self.documents = list(dataset.documents.values())
        self.documents_ids = list(dataset.documents.keys())

        tokenized_docs = [self.__tokenizer(doc) for doc in tqdm(self.documents)]
        self.bm25 = fastbm25(tokenized_docs)

    def __tokenizer(self, text: str) -> list[str]:
        return word_tokenize(text)

    def run(self, dataset: MSMarcoDataset, query_id: str, k: int = 10, **kwargs) -> list[tuple[str, float]]:
        query = self.__tokenizer(dataset.queries[query_id])
        scores = self.bm25.top_k_sentence(query)
        scores = [(self.documents_ids[index], score) for doc, index, score in scores]
        return scores[:k]


class BM25PySerini(Retriver):
    """
    BM25 algorithm for document retrieval using pyserini java implementation.
    """
    def __init__(self, dataset: MSMarcoDataset, index_folder: str, dataset_name: str):
        self.dataset = dataset
        self.index_folder = index_folder
        self.dataset_name = dataset_name

        if not os.path.exists(os.path.join(index_folder, 'output_' + dataset_name)):
            self.__create_index(dataset.documents, index_folder, dataset_name)
        self.lucene_searcher = LuceneSearcher(os.path.join(index_folder, 'output_' + dataset_name))

    def __create_index(self, documents: dict[str, str], index_folder: str, dataset_name: str):
        os.makedirs(index_folder, exist_ok=True)
        index_dataset_folder = os.path.join(index_folder, 'input_' + dataset_name)
        os.makedirs(index_dataset_folder, exist_ok=True)

        with open(os.path.join(index_dataset_folder, 'index.jsonl'), 'w') as f:
            for id, text in tqdm(documents.items()):
                f.write(json.dumps({"id": id, "contents": text}) + "\n")
        
        index_dataset_folder_output = index_dataset_folder.replace('input_', 'output_')
        os.makedirs(index_dataset_folder_output, exist_ok=True)
        cmd = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", index_dataset_folder,
            "--index", index_dataset_folder_output,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", "8",
            "--storePositions", "--storeDocvectors", "--storeRaw"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

    def run(self, dataset: MSMarcoDataset, query_id: str, k: int = 10, **kwargs) -> list[tuple[str, float]]:
        query = dataset.queries[query_id]
        scores = self.lucene_searcher.search(query, k)
        scores = [(hit.docid, hit.score) for hit in scores]
        return scores
