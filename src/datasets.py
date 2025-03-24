"""
Wrap usage of our datasets and also provide tokenizer interface.
"""
import os
import pickle
import random
from tqdm import tqdm

import nltk
from nltk.tokenize import word_tokenize


class Query:
    def __init__(self, query_id: str, text: str, tokenized_text: list[str] = None):
        self.query_id = query_id
        self.text = text
        self.tokenized_text = tokenized_text


class Document:
    def __init__(self, doc_id: str, text: str, tokenized_text: list[str] = None):
        self.doc_id = doc_id
        self.text = text
        self.tokenized_text = tokenized_text


class Qrel:
    def __init__(self, query_id: str, docs_ids: 'list[str]' = None):
        self.query_id = query_id
        if docs_ids is None:
            self.docs_ids = []
        else:
            self.docs_ids = docs_ids


class MSMarcoDataset:
    def __init__(self, data_folder: str):
        self.data_folder = data_folder

    def __load_data(self, input_file: str):
        with open(os.path.join(self.data_folder, input_file), 'rb') as f:
            dataset = pickle.load(f)

        self.queries = {key: Query(key, value.text) for key, value in tqdm(dataset['queries'].items(), desc='Loading queries')}
        self.documents = {key: Document(key, value.text) for key, value in tqdm(dataset['docs'].items(), desc='Loading documents')}
        _qrels = {}
        for qrel in tqdm(dataset['qrels'], desc='Loading qrels'):
            qrel_id = qrel.query_id
            doc_id = qrel.doc_id
            if qrel_id not in _qrels:
                _qrels[qrel_id] = [doc_id]
            else:
                _qrels[qrel_id].append(doc_id)
        self.qrels = {key: Qrel(key, value) for key, value in _qrels.items()}

    def __tokenize_data(self):
        nltk.download('punkt_tab')
        nltk.download('punkt')

        for query in tqdm(self.queries.values(), desc='Tokenizing queries'):
            query.tokenized_text = word_tokenize(query.text.lower())
        
        for doc in tqdm(self.documents.values(), desc='Tokenizing documents'):
            doc.tokenized_text = word_tokenize(doc.text.lower())

    def __split_data(self, split_ratio: float = 0.8, seed: int = 42):
        random.seed(seed)

        # Split the queries (assuming queries is a dictionary of {query_id: query_object})
        query_ids = list(self.queries.keys())  # List of query IDs

        # Shuffle query IDs to ensure a random split
        random.shuffle(query_ids)

        # Split into 80% for training, 20% for validation
        split_ratio = 0.8
        self.train_query_ids = query_ids[:int(len(query_ids) * split_ratio)]
        self.test_query_ids = query_ids[int(len(query_ids) * split_ratio):]

    def get_data(self, input_file: str):
        print(f"Loading data from {input_file}...")
        self.__load_data(input_file)
        
        print('Tokenizing data...')
        self.__tokenize_data()

        print('Split data into train and test sets...')
        self.__split_data()

        print('Data loading and processing complete.')
