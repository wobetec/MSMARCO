import json
import time
from tqdm import tqdm

from src.algorithms import BaseAlgorithm
from src.datasets import MSMarcoDataset


class Experiment:

    def __init__(
        self,
        name: str,
        stages: dict[str, tuple[BaseAlgorithm, dict]],
        desc_params: dict,
        dataset: MSMarcoDataset,
        query_ids: list[str] = None,
        input_score_docs: dict[str, list[tuple[str, float]]] = None,
    ):
        self.name = name
        self.stages = stages
        self.desc_params = desc_params
        self.dataset = dataset

        if query_ids is None and input_score_docs is None:
            raise ValueError("Either query_ids or input_score_docs must be provided.")
        elif query_ids is not None and input_score_docs is not None:
            raise ValueError("Only one of query_ids or input_score_docs must be provided.")
        elif query_ids is not None:
            if len(query_ids) == 0:
                raise ValueError("query_ids must be a non-empty list.")
            self.input_score_docs = {query_id: [] for query_id in query_ids}
        else:
            self.input_score_docs = input_score_docs

        self.times = {stage: [] for stage in self.stages.keys()}
        self.score_docs = {stage: {} for stage in self.stages.keys()}

    def run(self):
        for query_id in tqdm(self.input_score_docs.keys(), desc="Running Stages"):
            score_docs = self.input_score_docs[query_id]
            for stage_name, (algorithm, parms) in self.stages.items():
                start = time.perf_counter()
                score_docs = algorithm.run(
                    dataset=self.dataset,
                    query_id=query_id,
                    score_docs=score_docs,
                    **parms
                )
                self.score_docs[stage_name][query_id] = score_docs
                end = time.perf_counter()
                self.times[stage_name].append(end - start)
        

    def save(self, output_dir: str):
        with open(f"{output_dir}/{self.name}.json", "w") as f:
            json.dump(
                {
                    "stages": {
                        stage: {
                            "algorithm": algorithm.__class__.__name__,
                            "params": params,
                        }
                        for stage, (algorithm, params) in self.stages.items()
                    },
                    "desc_params": self.desc_params,
                    "dataset": f'{self.dataset.data_folder}/{self.dataset.input_file}',
                    "times": self.times,
                    "score_docs": self.score_docs,
                },
                f,
                indent=4,
            )
    
    @staticmethod
    def carregar_score_docs(output_dir: str, name: str, stage: str):
        with open(f'{output_dir}/{name}.json') as f:
            data = json.load(f)
        
        return data['score_docs'][stage]