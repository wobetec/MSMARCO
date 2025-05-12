
"""
Wrapper for Sentence Transformer algorithm.
"""
import json
import os

from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer

from src.rankers.ranker import Ranker
from src.datasets import MSMarcoDataset
from src.utils.cuda import get_device


class SentenceTransformerSimilarity(Ranker):

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: torch.device = None):
        self.device = get_device() if device is None else device

        self.model = SentenceTransformer(model_name, device=self.device)

    def encode_docs(self, dataset: MSMarcoDataset, file: str, batch_size: int = 32, **kwargs) -> None:
        if os.path.exists(file + '.pt') and os.path.exists(file + '.json'):
            print('Loading embeddings from file...')
            self.embeddings = torch.load(file + '.pt', map_location=self.device)
            with open(file + '.json', 'r') as f:
                self.ids = json.load(f)
            return

        print('Encoding documents...')
        inputs = list(dataset.documents.items())
        batches_inputs = [inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)]
        embeddings_list = []
        for batch in tqdm(batches_inputs, desc='Encoding documents', unit='batch'):
            embeddings = self.model.encode(
                [x[1] for x in batch],
                normalize_embeddings=True,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            embeddings_list.append(embeddings)
        self.embeddings = torch.cat(embeddings_list, dim=0)
        self.ids = {x[0]: i for i, x in enumerate(inputs)}
        
        print('Saving embeddings to file...')
        os.makedirs(os.path.dirname(file + '.pt'), exist_ok=True)
        torch.save(self.embeddings, file + '.pt')
        print('Saving ids to file...')
        with open(file + '.json', 'w') as f:
            json.dump(self.ids, f)

    def run(self, dataset: MSMarcoDataset, query_id: str, score_docs: list[tuple[str, float]], k: int = 10, **kwargs) -> list[tuple[str, float]]:
        query = dataset.queries[query_id]

        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=False
        ).unsqueeze(0).to(self.device)

        docs_embeddings = torch.stack([self.embeddings[self.ids[doc_id]] for doc_id, _ in score_docs])

        new_score = torch.nn.functional.cosine_similarity(
            query_embedding,
            docs_embeddings,
        ).cpu().numpy().tolist()

        new_score_docs = [(doc_id, score) for (doc_id, _), score in zip(score_docs, new_score)]
        new_score_docs = sorted(new_score_docs, key=lambda x: x[1], reverse=True)
        return new_score_docs[:k]
