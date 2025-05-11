"""
Wrapper for Faiss algorithm.
"""
import faiss
import os

from src.datasets import MSMarcoDataset
from src.retrivers.retriver import Retriver
from src.rankers.sentence_transformer import SentenceTransformerSimilarity


class Faiss(Retriver):
    """
    Faiss algorithm for document retrieval.
    """
    def __init__(self, dataset: MSMarcoDataset, file: str):
        self.sentence_transformer = SentenceTransformerSimilarity()
        self.sentence_transformer.encode_docs(
            dataset,
            file.replace('faiss', 'sentence_transformer'),
            batch_size=32
        )
        if os.path.exists(file + '.faiss'):
            print('Loading index from file...')
            self.index = faiss.read_index(file + '.faiss')
        else:
            print('Creating index...')
            os.makedirs(os.path.dirname(file + '.faiss'), exist_ok=True)
            index_size = self.sentence_transformer.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(index_size)
            vectors = self.sentence_transformer.embeddings.cpu().numpy().astype('float32')
            faiss.normalize_L2(vectors)
            self.index.add(vectors)
            faiss.write_index(self.index, file + '.faiss')

        self.reverse_map = {i: doc_id for i, doc_id in enumerate(self.sentence_transformer.ids)}

    def run(self, dataset: MSMarcoDataset, query_id: str, k: int = 10, **kwargs) -> list[tuple[str, float]]:
        query_emebeddings = self.sentence_transformer.model.encode(
            dataset.queries[query_id],
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=False
        ).unsqueeze(0).cpu().numpy().astype('float32')
        faiss.normalize_L2(query_emebeddings)
        distances, ann = self.index.search(query_emebeddings, k)
        score_docs = [(self.reverse_map[i], float(dist)) for i, dist in zip(ann[0], distances[0])]
        return score_docs
