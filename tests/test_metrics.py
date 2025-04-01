import math
import unittest

from src.metrics import (
    reciprocal_rank,
    precision,
    recall,
    f1,
    dcg,
    ndcg,

    mrr_score,
    map_score,
    mr_score,
    mf1_score,
    mndcg_score,
)


class TestBasicMetrics(unittest.TestCase):
    def test_reciprocal_rank(self):
        sorted_docs = [(f'doc_{i}', i) for i in range(1, 100 + 1)]
        self.assertEqual(reciprocal_rank(sorted_docs, ['doc_10'], 10), 0.1)
        self.assertEqual(reciprocal_rank(sorted_docs, ['doc_20'], 10), 0)
        self.assertEqual(reciprocal_rank(sorted_docs, ['doc_5', 'doc_10'], 10), 0.2)

    def test_precision(self):
        sorted_docs = [(f'doc_{i}', i) for i in range(1, 100 + 1)]
        self.assertAlmostEqual(precision(sorted_docs, ['doc_10'], 10), 0.1)
        self.assertAlmostEqual(precision(sorted_docs, ['doc_20'], 10), 0)
        self.assertAlmostEqual(precision(sorted_docs, ['doc_5', 'doc_10'], 10), 0.2)

    def test_recall(self):
        sorted_docs = [(f'doc_{i}', i) for i in range(1, 100 + 1)]
        self.assertAlmostEqual(recall(sorted_docs, ['doc_10'], 10), 1)
        self.assertAlmostEqual(recall(sorted_docs, ['doc_20'], 10), 0)
        self.assertAlmostEqual(recall(sorted_docs, ['doc_5', 'doc_20'], 10), 0.5)

    def test_f1(self):
        sorted_docs = [(f'doc_{i}', i) for i in range(1, 100 + 1)]
        self.assertAlmostEqual(f1(sorted_docs, ['doc_10'], 10), 2 / 11)
        self.assertAlmostEqual(f1(sorted_docs, ['doc_5', 'doc_10'], 10), 1 / 3)

    def test_dcg(self):
        sorted_docs = [(f'doc_{i}', i) for i in range(1, 100 + 1)]
        self.assertAlmostEqual(dcg(sorted_docs, ['doc_10'], 10), 1 / math.log2(10 + 1))
        self.assertAlmostEqual(dcg(sorted_docs, ['doc_20'], 10), 0)

    def test_ndcg(self):
        sorted_docs = [(f'doc_{i}', i) for i in range(1, 100 + 1)]
        self.assertAlmostEqual(ndcg(sorted_docs, ['doc_10'], 10), math.log2(1 + 1) / math.log2(10 + 1) )
        self.assertAlmostEqual(ndcg(sorted_docs, ['doc_20'], 10), 0)


class TestGroupMetrics(unittest.TestCase):
    def test_mrr_score(self):
        sorted_docs = {
            'query_1': [(f'doc_{i}', i) for i in range(1, 10 + 1)],
            'query_2': [(f'doc_{i}', i) for i in range(1, 20 + 1)],
        }
        relevant_docs = {
            'query_1': ['doc_5'],
            'query_2': ['doc_10'],
        }
        self.assertAlmostEqual(mrr_score(sorted_docs, relevant_docs, k=10), (0.2 + 0.1) / 2)

    def test_map_score(self):

        sorted_docs = {
            'query_1': [(f'doc_{i}', i) for i in range(1, 10 + 1)],
            'query_2': [(f'doc_{i}', i) for i in range(1, 20 + 1)],
        }
        relevant_docs = {
            'query_1': ['doc_5', 'doc_10'],
            'query_2': ['doc_10'],
        }
        self.assertAlmostEqual(map_score(sorted_docs, relevant_docs, k=10), (0.2 + 0.1) / 2)

    def test_mr_score(self):
        sorted_docs = {
            'query_1': [(f'doc_{i}', i) for i in range(1, 10 + 1)],
            'query_2': [(f'doc_{i}', i) for i in range(1, 20 + 1)],
        }
        relevant_docs = {
            'query_1': ['doc_5', 'doc_20'],
            'query_2': ['doc_10'],
        }
        self.assertAlmostEqual(mr_score(sorted_docs, relevant_docs, k=10), (0.5 + 1) / 2)

    def test_mf1_score(self):
        sorted_docs = {
            'query_1': [(f'doc_{i}', i) for i in range(1, 100 + 1)],
            'query_2': [(f'doc_{i}', i) for i in range(1, 100 + 1)],
        }
        relevant_docs = {
            'query_1': ['doc_5', 'doc_10'],
            'query_2': ['doc_10'],
        }
        self.assertAlmostEqual(mf1_score(sorted_docs, relevant_docs, k=10), (2 / 11 + 1 / 3) / 2)

    def test_mndcg_score(self):
        sorted_docs = {
            'query_1': [(f'doc_{i}', i) for i in range(1, 100 + 1)],
            'query_2': [(f'doc_{i}', i) for i in range(1, 100 + 1)],
        }
        relevant_docs = {
            'query_1': ['doc_10'],
            'query_2': ['doc_20'],
        }
        self.assertAlmostEqual(mndcg_score(sorted_docs, relevant_docs, k=10), (math.log2(1 + 1) / math.log2(10 + 1) + 0) / 2)
