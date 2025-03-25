"""
Metrics for evaluating the performance of reranking algorithms.
"""
import numpy as np
from src.datasets import Qrel


def mmr_score(sorted_docs: 'dict[str, list[tuple[str, float]]]', qrels: 'dict[str, Qrel]', k: int = 10) -> float:
    '''
    https://mariofilho.com/mean-reciprocal-rank-mrr-em-machine-learning/
    '''
    mrr = 0
    for query_id, query_sorted_docs in sorted_docs.items():
        relevant_docs = qrels.get(query_id, Qrel('')).docs_ids
        rank = -1
        for i, (doc, _) in enumerate(query_sorted_docs[:k]):
            if doc in relevant_docs:
                rank = i + 1
                break
        if rank > 0:
            mrr += 1 / rank
    return mrr / len(sorted_docs)
