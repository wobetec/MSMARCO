import unittest
from src.datasets import MSMarcoDataset
from src.retrivers.faiss import Faiss

class TestFaiss(unittest.TestCase):
    def test_init(self):
        dataset = MSMarcoDataset('data/subset_msmarco_train_0')
        dataset.load_data('subset_msmarco_train_0.01_99.pkl')

        faiss = Faiss(dataset, 'data/embeddings/faiss/subset_msmarco_train_0.01_99')
        self.assertIsInstance(faiss, Faiss)

        score_docs = faiss.run(dataset, list(dataset.queries.keys())[0], k=10)
        self.assertIsInstance(score_docs, list)
        self.assertEqual(len(score_docs), 10)
        self.assertIsInstance(score_docs[0], tuple)
        self.assertEqual(len(score_docs[0]), 2)
        self.assertIsInstance(score_docs[0][0], str)
        self.assertIsInstance(score_docs[0][1], float)
