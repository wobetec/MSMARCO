import sys
import os
import argparse


from src.experiments import Experiment
from copy import deepcopy
from src.datasets import MSMarcoDataset, PreProcessor
import pickle
from src.retrivers.bm25 import BM25
from src.rankers.monobert import MonoBERT
from src.utils.cuda import check_cuda


OUTPUT_DIR = 'results/raw'
data_folder = 'data/subset_msmarco_train_0'
input_file = 'subset_msmarco_train_0.01_99.pkl'

temp_data = 'data/temp_data'

monobert_model = 'castorini/monobert-large-msmarco'


def load_datasets():

    def load_pkl(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def save_pkl(file_path, data):
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    os.makedirs(temp_data, exist_ok=True)
    
    caminho_dataset = os.path.join(temp_data, 'dataset.pkl')
    if not os.path.exists(caminho_dataset):
        dataset = MSMarcoDataset(data_folder)
        dataset.load_data(input_file)
        dataset.split_data()
        save_pkl(caminho_dataset, dataset)
    else:
        dataset = load_pkl(caminho_dataset)

    caminho_dataset_lower = os.path.join(temp_data, 'dataset_lower.pkl')
    if not os.path.exists(caminho_dataset_lower):
        preprocessor = PreProcessor([
            PreProcessor.lowercase,
        ])
        dataset_lower = deepcopy(dataset)
        preprocessor.preprocess(dataset_lower)
        save_pkl(caminho_dataset_lower, dataset_lower)
    else:
        dataset_lower = load_pkl(caminho_dataset_lower)

    caminho_dataset_prepro = os.path.join(temp_data, 'dataset_prepro.pkl')
    if not os.path.exists(caminho_dataset_prepro):
        preprocessor = PreProcessor([
            PreProcessor.lowercase,
            PreProcessor.remove_punctuation,
            PreProcessor.remove_stopwords,
            PreProcessor.stem,
        ])
        dataset_prepro = deepcopy(dataset)
        preprocessor.preprocess(dataset_prepro)
        save_pkl(caminho_dataset_prepro, dataset_prepro)
    else:
        dataset_prepro = load_pkl(caminho_dataset_prepro)
    
    return dataset, dataset_lower, dataset_prepro


def run_experiment(name: str):
    dataset, dataset_lower, dataset_prepro = load_datasets()
    match name:
        case 'monobert':
            bm25 = BM25(dataset)
            monobert = MonoBERT(monobert_model, use_amp=True)
            baseline = Experiment(
                'monobert',
                {
                    'BM25': (bm25, {'k': 100}),
                    'MonoBERT': (monobert, {'k': 100}),
                },
                {},
                dataset,
                query_ids=dataset.test_query_ids
            )
            baseline.run()
            baseline.save(OUTPUT_DIR)
        case 'monobert_lower':
            bm25 = BM25(dataset_lower)
            monobert = MonoBERT(monobert_model, use_amp=True)
            baseline = Experiment(
                'monobert_lower',
                {
                    'BM25': (bm25, {'k': 100}),
                    'MonoBERT': (monobert, {'k': 100}),
                },
                {},
                dataset_lower,
                query_ids=dataset_lower.test_query_ids
            )
            baseline.run()
            baseline.save(OUTPUT_DIR)
        case 'monobert_prepro':
            bm25 = BM25(dataset_prepro)
            monobert = MonoBERT(monobert_model, use_amp=True)
            baseline = Experiment(
                'monobert_prepro',
                {
                    'BM25': (bm25, {'k': 100}),
                    'MonoBERT': (monobert, {'k': 100}),
                },
                {},
                dataset_prepro,
                query_ids=dataset_prepro.test_query_ids
            )
            baseline.run()
            baseline.save(OUTPUT_DIR)
        case _:
            raise ValueError(f"Unknown experiment name: {name}")
        

if __name__ == '__main__':
    experiments = sys.argv[1:]
    check_cuda()
    for i, experiment in enumerate(experiments):
        print(f"Running experiment ({i + 1}/{len(experiments)}): {experiment}")
        run_experiment(experiment)
