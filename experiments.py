import itertools
import os
import pickle
import sys
import time
from copy import deepcopy

from tqdm import tqdm

from src.datasets import MSMarcoDataset, PreProcessor
from src.rankers.monobert import MonoBatchBERT, MonoBERT
from src.rankers.sentence_transformer import SentenceTransformerSimilarity
from src.results import Result
from src.retrivers.bm25 import BM25, BM25Fast, BM25PySerini
from src.retrivers.faiss import Faiss
from src.utils.cuda import check_cuda

DEFAULT_DATA_FOLDER = 'data/subset_msmarco_train_0'
DEFAULT_DATASET_NAME = 'subset_msmarco_train_0.01_99'
DEFAULT_INDEX_FOLDER = 'data/indexes'
DEFAULT_EMBEDDINGS_FOLDER = 'data/embeddings'

K = 100

PRE_DEFINED_PREPROCS = [
    'none',
    'lower',
    'full',
]


def load_datasets(dataset_name: str = DEFAULT_DATASET_NAME, data_folder: str = DEFAULT_DATA_FOLDER) -> dict[str, MSMarcoDataset]:
    print('Loading datasets...')
    def load_pkl(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def save_pkl(file_path, data):
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    datasets_dir = os.path.join(data_folder, dataset_name)
    os.makedirs(datasets_dir, exist_ok=True)

    caminho_dataset = os.path.join(datasets_dir, 'none.pkl')
    if not os.path.exists(caminho_dataset):
        dataset = MSMarcoDataset(data_folder)
        dataset.load_data(dataset_name + '.pkl')
        dataset.split_data()
        save_pkl(caminho_dataset, dataset)
    else:
        dataset = load_pkl(caminho_dataset)

    caminho_dataset_lower = os.path.join(datasets_dir, 'lower.pkl')
    if not os.path.exists(caminho_dataset_lower):
        preprocessor = PreProcessor([
            PreProcessor.lowercase,
        ])
        dataset_lower = deepcopy(dataset)
        preprocessor.preprocess(dataset_lower)
        save_pkl(caminho_dataset_lower, dataset_lower)
    else:
        dataset_lower = load_pkl(caminho_dataset_lower)

    caminho_dataset_full = os.path.join(datasets_dir, 'full.pkl')
    if not os.path.exists(caminho_dataset_full):
        preprocessor = PreProcessor([
            PreProcessor.lowercase,
            PreProcessor.remove_punctuation,
            PreProcessor.remove_stopwords,
            PreProcessor.stem,
        ])
        dataset_full = deepcopy(dataset)
        preprocessor.preprocess(dataset_full)
        save_pkl(caminho_dataset_full, dataset_full)
    else:
        dataset_full = load_pkl(caminho_dataset_full)
    
    return {
        'none': dataset,
        'lower': dataset_lower,
        'full': dataset_full
    }


class PreDefinedLayers:
    
    # Retrivers
    @staticmethod
    def BM25(dataset: MSMarcoDataset, prev_score_docs: dict[str, list[tuple[str, float]]], *args, **kwargs) -> tuple[dict[str, list[tuple[str, float]]], list[float]]:
        bm25 = BM25(dataset)
        score_docs = {}
        times = []
        for query_id in tqdm(list(prev_score_docs.keys())):
            start_time = time.perf_counter()
            score_docs[query_id] = bm25.run(dataset, query_id, k=K)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        return score_docs, times
    
    @staticmethod
    def BM25Fast(dataset: MSMarcoDataset, prev_score_docs: dict[str, list[tuple[str, float]]], *args, **kwargs) -> tuple[dict[str, list[tuple[str, float]]], list[float]]:
        bm25_fast = BM25Fast(dataset)
        score_docs = {}
        times = []
        for query_id in tqdm(list(prev_score_docs.keys())):
            start_time = time.perf_counter()
            score_docs[query_id] = bm25_fast.run(dataset, query_id, k=K)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        return score_docs, times
    
    @staticmethod
    def BM25PySerini(dataset: MSMarcoDataset, prev_score_docs: dict[str, list[tuple[str, float]]], dataset_name: str, *args, **kwargs) -> tuple[dict[str, list[tuple[str, float]]], list[float]]:
        bm25_fast = BM25PySerini(dataset, DEFAULT_INDEX_FOLDER, dataset_name)
        score_docs = {}
        times = []
        for query_id in tqdm(list(prev_score_docs.keys())):
            start_time = time.perf_counter()
            score_docs[query_id] = bm25_fast.run(dataset, query_id, k=K)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        return score_docs, times

    @staticmethod
    def Faiss(dataset: MSMarcoDataset, prev_score_docs: dict[str, list[tuple[str, float]]], dataset_name: str, *args, **kwargs) -> tuple[dict[str, list[tuple[str, float]]], list[float]]:
        faiss = Faiss(dataset, os.path.join(DEFAULT_EMBEDDINGS_FOLDER, 'faiss', dataset_name))
        score_docs = {}
        times = []
        for query_id in tqdm(list(prev_score_docs.keys())):
            start_time = time.perf_counter()
            score_docs[query_id] = faiss.run(dataset, query_id, k=K)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        return score_docs, times

    # Rerankers
    @staticmethod
    def SentenceTransformerSimilarity(dataset: MSMarcoDataset, prev_score_docs: dict[str, list[tuple[str, float]]], dataset_name: str, *args, **kwargs) -> tuple[dict[str, list[tuple[str, float]]], list[float]]:
        sentence_transformer = SentenceTransformerSimilarity()
        sentence_transformer.encode_docs(dataset, os.path.join(DEFAULT_EMBEDDINGS_FOLDER, 'sentence_transformer', dataset_name))
        score_docs = {}
        times = []
        for query_id in tqdm(list(prev_score_docs.keys())):
            start_time = time.perf_counter()
            score_docs[query_id] = sentence_transformer.run(dataset, query_id, prev_score_docs[query_id], k=K)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        return score_docs, times

    @staticmethod
    def MonoBatchBERT(dataset: MSMarcoDataset, prev_score_docs: dict[str, list[tuple[str, float]]], dataset_name: str, *args, **kwargs) -> tuple[dict[str, list[tuple[str, float]]], list[float]]:
        monobatchbert = MonoBatchBERT()
        score_docs = {}
        times = []
        for query_id in tqdm(list(prev_score_docs.keys())):
            start_time = time.perf_counter()
            score_docs[query_id] = monobatchbert.run(dataset, query_id, prev_score_docs[query_id], k=K)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        return score_docs, times
    
    @staticmethod
    def MonoBERT(dataset: MSMarcoDataset, prev_score_docs: dict[str, list[tuple[str, float]]], dataset_name: str, *args, **kwargs) -> tuple[dict[str, list[tuple[str, float]]], list[float]]:
        monobert = MonoBERT()
        score_docs = {}
        times = []
        for query_id in tqdm(list(prev_score_docs.keys())):
            start_time = time.perf_counter()
            score_docs[query_id] = monobert.run(dataset, query_id, prev_score_docs[query_id], k=K)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        return score_docs, times


PRE_DEFINED_LAYERS_MAP = {
    'BM25': PreDefinedLayers.BM25,
    'BM25Fast': PreDefinedLayers.BM25Fast,
    'BM25PySerini': PreDefinedLayers.BM25PySerini,
    'Faiss': PreDefinedLayers.Faiss,

    'SentenceTransformerSimilarity': PreDefinedLayers.SentenceTransformerSimilarity,
    'MonoBatchBERT': PreDefinedLayers.MonoBatchBERT,
    'MonoBERT': PreDefinedLayers.MonoBERT,
}


def run_experiment(dataset_name: str, preproc_name: str, *layers):
    datasets = load_datasets(dataset_name=dataset_name)
    dataset = datasets[preproc_name]
    prev_result_score_docs = {query_id: [] for query_id in dataset.test_query_ids}
    current_layers = []
    for layer in layers:
        current_layers.append(layer)
        if layer not in PRE_DEFINED_LAYERS_MAP:
            raise ValueError(f"Unknown layer: {layer}. Expected one of {list(PRE_DEFINED_LAYERS_MAP.keys())}.")
        
        print(f"Running {layer} on {dataset_name} with preprocessor {preproc_name}...")
        if Result.exists(dataset_name, preproc_name, *layers):
            print(f"\tAlready exist")
            result = Result.load_results(dataset_name, preproc_name, *current_layers)
            prev_result_score_docs = result.score_docs
            continue

        layer_func = PRE_DEFINED_LAYERS_MAP[layer]
        score_docs, times = layer_func(dataset, prev_result_score_docs, dataset_name=dataset_name)
        prev_result_score_docs = score_docs
        result = Result(dataset_name, preproc_name, *current_layers)
        result.score_docs = score_docs
        result.times = times
        result.save_results()


if __name__ == '__main__':
    check_cuda()

    command = sys.argv[1]
    if command == 'load_datasets':
        dataset, dataset_lower, dataset_full = load_datasets()
        print("Datasets loaded successfully.")
    elif command == 'run_experiment':
        dataset_name = sys.argv[2].split('-')
        preproc_name = sys.argv[3].split('-')
        if len(set(preproc_name) - set(PRE_DEFINED_PREPROCS)) > 0:
            raise ValueError(f"Unknown preproc name: {preproc_name}. Expected one of {PRE_DEFINED_PREPROCS}.")
        layers = [layer.split('-') for layer in sys.argv[4:]]
        for layer in layers:
            if len(set(layer) - set(PRE_DEFINED_LAYERS_MAP.keys())) > 0:
                raise ValueError(f"Unknown layer: {layer}. Expected one of {list(PRE_DEFINED_LAYERS_MAP.keys())}.")
        
        experiments_args = list(itertools.product(dataset_name, preproc_name, *layers))
        for experiment_arg in tqdm(experiments_args):
            run_experiment(*experiment_arg)
        print("Experiment run successfully.")
    else:
        raise ValueError(f"Unknown command: {command}")
