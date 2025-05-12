import unittest
from src.datasets import MSMarcoDataset
from src.retrivers.vespa import VespaBM25


class TestVespaB25(unittest.TestCase):
    def test_init(self):
        dataset = MSMarcoDataset('data/subset_msmarco_train_0')
        dataset.load_data('subset_msmarco_train_0.01_99.pkl')

        query_id = list(dataset.queries.keys())[0]
        vespa = VespaBM25()
        self.assertIsInstance(vespa, VespaBM25)

        score_docs = vespa.run(dataset, query_id, k=10)

        self.assertIsInstance(score_docs, list)
        self.assertEqual(len(score_docs), 10)
        self.assertIsInstance(score_docs[0], tuple)
        self.assertEqual(len(score_docs[0]), 2)
        self.assertIsInstance(score_docs[0][0], str)
        self.assertIsInstance(score_docs[0][1], float)
