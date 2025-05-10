"""
Wrap usage of our datasets and also provide interface for preprocessing.
"""
import os
import pickle
import random
from tqdm import tqdm
from nltk.stem import PorterStemmer


class MSMarcoDataset:
    def __init__(self, data_folder: str):
        self.data_folder = data_folder
        self.input_file = None

    def load_data(self, input_file: str):
        with open(os.path.join(self.data_folder, input_file), 'rb') as f:
            dataset = pickle.load(f)

        self.input_file = input_file

        self.queries = {key: value.text for key, value in tqdm(dataset['queries'].items(), desc='Loading queries')}
        self.documents = {key: value.text for key, value in tqdm(dataset['docs'].items(), desc='Loading documents')}
        self.qrels = {}
        for qrel in tqdm(dataset['qrels'], desc='Loading qrels'):
            qrel_id = qrel.query_id
            doc_id = qrel.doc_id
            if qrel_id not in self.qrels:
                self.qrels[qrel_id] = [doc_id]
            else:
                self.qrels[qrel_id].append(doc_id)

    def split_data(self, split_ratio: float = 0.8, seed: int = 42):
        random.seed(seed)
        query_ids = list(self.queries.keys())
        random.shuffle(query_ids)
        self.train_query_ids = query_ids[:int(len(query_ids) * split_ratio)]
        self.test_query_ids = query_ids[int(len(query_ids) * split_ratio):]

class PreProcessor:

    class_processors = {}

    def __init__(self, pipeline: list[callable]):
        self.pipeline = pipeline

    def preprocess_text(self, text: str) -> str:
        for func in self.pipeline:
            text = func(text)
        return text

    def preprocess(self, dataset: MSMarcoDataset):
        print("Preprocessing dataset...")
        dataset.queries = {query_id: self.preprocess_text(text) for query_id, text in tqdm(dataset.queries.items(), desc='Preprocessing queries')}
        dataset.documents = {doc_id: self.preprocess_text(text) for doc_id, text in tqdm(dataset.documents.items(), desc='Preprocessing documents')}

    @classmethod
    def lowercase(cls, text: str) -> str:
        return text.lower()
    
    @classmethod
    def remove_punctuation(cls, text: str) -> str:
        return ''.join(char for char in text if char.isalnum() or char.isspace())
    
    @classmethod
    def remove_stopwords(cls, text: str) -> str:
        stopwords = set(["the", "is", "in", "and", "to", "a"])
        return ' '.join(word for word in text.split() if word not in stopwords)
    
    @classmethod
    def stem(cls, text: str) -> str:
        if 'stemmer' not in cls.class_processors:
            cls.class_processors['stemmer'] = PorterStemmer()
        stemmer = cls.class_processors['stemmer']
        return ' '.join(stemmer.stem(word) for word in text.split())
