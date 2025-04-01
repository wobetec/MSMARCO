import random
import unittest

from src.datasets import MSMarcoDataset
from src.rankers.monobert import MonoBERT


class TestBM25(unittest.TestCase):
    def test_init(self):
        dataset = MSMarcoDataset('data/subset_msmarco_train_0')
        dataset.load_data('subset_msmarco_train_0.01_9.pkl')

        monobert = MonoBERT("castorini/monobert-large-msmarco")
        self.assertIsInstance(monobert, MonoBERT)

        # some score_docs
        query_id = list(dataset.queries.keys())[0]
        prev_score_docs = [
            (doc_id, random.uniform(0, 1))
            for doc_id in list(set(list(dataset.documents.keys())[:10] + dataset.qrels[query_id]))
        ]
        score_docs = monobert.run(dataset, query_id, prev_score_docs)
        self.assertIsInstance(score_docs, list)
        self.assertEqual(len(score_docs), len(prev_score_docs))
        self.assertTrue(all(isinstance(x, tuple) and len(x) == 2 for x in score_docs))
        self.assertTrue(all(isinstance(x[0], str) and isinstance(x[1], float) for x in score_docs)) 
