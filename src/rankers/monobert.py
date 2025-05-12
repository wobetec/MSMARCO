"""
Wrapper for MonoBERT algorithm.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.rankers.ranker import Ranker
from src.datasets import MSMarcoDataset
from src.utils.cuda import get_device


class MonoBatchBERT(Ranker):

    def __init__(self, model_name: str = 'mixedbread-ai/mxbai-rerank-base-v1', device: torch.device = None):
        self.device = get_device() if device is None else device
        self.is_monobert = 'monobert' in model_name.lower()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()

    def run(self, dataset: MSMarcoDataset, query_id: str, score_docs: list[tuple[str, float]], k: int = 10, batch_size: int = 32, **kwargs) -> list[tuple[str, float]]:
        query = dataset.queries[query_id]
        inputs = [(query, dataset.documents[doc_id]) for doc_id, score in score_docs]
        batches_inputs = [inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)]

        new_score_docs = []
        for batche in batches_inputs:
            tokenized_inputs = self.tokenizer(
                batche,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            batche_scores = self.model(**tokenized_inputs.to(self.device)).logits.squeeze()
            barche_scores = batche_scores.detach().cpu().numpy().tolist()
            new_score_docs += barche_scores
        if self.is_monobert:
            new_score_docs = [x[1] - x[0] for x in new_score_docs]
        new_score_docs = [(doc_id, score) for (doc_id, _), score in zip(score_docs, new_score_docs)]
        
        return sorted(new_score_docs, key=lambda x: x[1], reverse=True)[:k]


class MonoBERT(Ranker):

    def __init__(self, model_name: str = 'castorini/monobert-large-msmarco', device: torch.device = None, use_amp: bool = False):
        self.device = get_device() if device is None else device
        self.use_amp = use_amp

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()

    def run(self, dataset: MSMarcoDataset, query_id: str, score_docs: list[tuple[str, float]], k: int = 10, **kwargs) -> list[tuple[str, float]]:
        query = dataset.queries[query_id]

        new_score_docs = []
        for doc_id, score in score_docs:
            inputs = self.tokenizer.encode_plus(
                query,
                dataset.documents[doc_id],
                max_length=512,
                truncation=True,
                return_token_type_ids=True,
                return_tensors="pt"
            )
            with torch.amp.autocast(enabled=self.use_amp, device_type=self.device.type):
                input_ids = inputs["input_ids"].to(self.device)
                token_type_ids = inputs["token_type_ids"].to(self.device)
                outputs = self.model(input_ids, token_type_ids=token_type_ids, return_dict=False)
                logits = outputs[0]

                if logits.size(1) > 1:
                    score = torch.nn.functional.log_softmax(logits, dim=1)[0, -1].item()
                else:
                    score = logits.item()
                
            new_score_docs.append((doc_id, score))
        
        return sorted(new_score_docs, key=lambda x: x[1], reverse=True)[:k]
