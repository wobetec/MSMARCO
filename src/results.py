"""
Control the results directory for the raw results of the layers.

RESULTS_DIR
    -> dataset_name (Used dataset name)
        -> preproc (Preprocessing name)
            -> model_name (Model class name for the first layer)
                -> model_name (Model class name for the second layer)
                    ...

All sub folders use as input the results of the previous layer.
"""
import json
import os
import pandas as pd

from src.utils.layers import get_layer_name
from src.algorithms import BaseAlgorithm

CLEAN_DIR = 'results/clean/'
RESULTS_DIR = 'results/raw/'


class Result:

    def __init__(self, dataset_name: str, preproc_name: str, *layers):
        self.dataset_name = dataset_name
        self.preproc_name = preproc_name
        self.layers = layers

        self.score_docs = {}
        self.times = []

    @property
    def file(self):
        return os.path.join(RESULTS_DIR, self.dataset_name, self.preproc_name, *self.layers, 'result.json')

    @classmethod
    def load_results(cls, dataset_name: str, preproc_name: str, *layers):
        result = Result(dataset_name, preproc_name, *layers)
        if not os.path.exists(result.file):
            raise FileNotFoundError(f"Result {result.file} does not exist")
        with open(result.file, 'r') as f:
            data = json.load(f)
        result.score_docs = data['score_docs']
        result.times = data['times']
        return result
    
    @classmethod
    def exists(cls, dataset_name: str, preproc_name: str, *layers):
        result = Result(dataset_name, preproc_name, *layers)
        return os.path.exists(result.file)

    def save_results(self):
        os.makedirs(os.path.dirname(self.file), exist_ok=True)
        with open(self.file, 'w') as f:
            json.dump({'score_docs': self.score_docs, 'times': self.times}, f)

    def get_new_layer(self, layer: str|BaseAlgorithm, score_docs: dict[str, list[list[str, float]]], times: list[float]):
        """
        Get a new layer with the same dataset and preproc name, but with the new layer name.
        """
        if isinstance(layer, BaseAlgorithm):
            layer_name = get_layer_name(layer)
        elif isinstance(layer, str):
            layer_name = layer
        else:
            raise TypeError(f"Layer must be a string or an instance of BaseAlgorithm, not {type(layer)}")
        new_layer = Result(self.dataset_name, self.preproc_name, *self.layers, layer_name)
        new_layer.score_docs = score_docs
        new_layer.times = times
        return new_layer


def load_results_raw() -> list[Result]:
    """
    Load the results from the raw results directory.
    """
    results = []
    RESULTS_DIR = 'results/raw/'
    for root, folders, files in os.walk(RESULTS_DIR):
        if 'result.json' in files:
            dataset_name, preproc_name, *layers = root.replace(RESULTS_DIR, '').split('\\')
            results.append(Result.load_results(dataset_name, preproc_name, *layers))
    
    return results


def get_results_df(
    results: list[Result],
    qrels: dict[str, list[str]],
    score_docs_metrics: dict[str, callable],
    time_metrics: dict[str, callable]
) -> pd.DataFrame:
    """
    Get the results dataframe from the results.
    """
    data = []
    max_layer = 0
    for result in results:
        dataset_name = result.dataset_name
        preproc_name = result.preproc_name
        layers = result.layers
        score_docs = result.score_docs
        times = result.times

        row = {
            'dataset': dataset_name,
            'preproc': preproc_name,
        }
        for i, layer in enumerate(layers):
            row[f'Layer {i + 1}'] = layer
        max_layer = max(max_layer, len(layers))
        
        for metric_name, metric in score_docs_metrics.items():
            row[metric_name] = metric(score_docs, qrels)
        
        for metric_name, metric in time_metrics.items():
            row[metric_name] = metric(times)

        data.append(row)

    df = pd.DataFrame(data)
    df = df.fillna('')
    df = df[
        ['dataset', 'preproc'] + \
        [f'Layer {i + 1}' for i in range(max_layer)] + \
        list(score_docs_metrics.keys()) + list(time_metrics.keys())
    ]
    return df
