"""
Metrics for evaluating the performance of reranking algorithms.
"""
import numpy as np
from src.datasets import Qrel


def mmr_score(sorted_docs: 'dict[str, list[str]]', qrels: 'dict[str, Qrel]', k: int = 10) -> float:
    '''
    https://mariofilho.com/mean-reciprocal-rank-mrr-em-machine-learning/
    '''
    mrr = 0
    for query_id, sorted_docs_ids in sorted_docs.items():
        relevant_docs = qrels.get(query_id, Qrel('')).docs_ids
        rank = -1
        for relevant_doc in relevant_docs:
            if relevant_doc in sorted_docs_ids:
                rank = sorted_docs_ids.index(relevant_doc) + 1
                break
        if rank > 0:
            mrr += 1 / rank
    return mrr / len(sorted_docs)


def ndcg_score(sorted_docs: 'dict[str, list[str]]', qrels: 'dict[str, Qrel]', k: int = 10) -> float:
    ''''
    https://mariofilho.com/ndcg-normalized-discounted-cumulative-gain-em-machine-learning/
    '''
    ndcg = 0
    for query_id, sorted_docs_ids in sorted_docs.items():
        relevant_docs = qrels.get(query_id, Qrel('')).docs_ids
        dcg = 0
        idcg = 0
        for i, doc_id in enumerate(sorted_docs_ids[:k]):
            if doc_id in relevant_docs:
                dcg += 1 / np.log2(i + 2)
            if doc_id in relevant_docs:
                idcg += 1 / np.log2(i + 2)
        ndcg += dcg / idcg if idcg > 0 else 0
    return ndcg / len(sorted_docs)