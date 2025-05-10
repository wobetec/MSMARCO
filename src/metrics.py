"""
Metrics for evaluating the performance of reranking algorithms.
"""
import math

import numpy as np


#------------------------------------------------------------#
# Basic metrics
#------------------------------------------------------------#
def reciprocal_rank(query_sorted_docs: list[tuple[str, float]], query_relevant_docs: list[str], k: int = 10) -> float:
    rank = -1
    for i, (doc, _) in enumerate(query_sorted_docs[:k]):
        if doc in query_relevant_docs:
            rank = i + 1
            break
    return 1 / rank if rank > 0 else 0


def precision(sorted_docs: list[tuple[str, float]], relevant_docs: list[str], k: int = 10) -> float:
    top_k_docs = [doc_id for doc_id, _ in sorted_docs[:k]]
    relevant_set = set(relevant_docs)
    relevant_count = sum(1 for doc in top_k_docs if doc in relevant_set)
    return relevant_count / k


def recall(sorted_docs: list[tuple[str, float]], relevant_docs: list[str], k: int = 10) -> float:
    top_k_docs = [doc_id for doc_id, _ in sorted_docs[:k]]
    relevant_set = set(relevant_docs)
    relevant_count = sum(1 for doc in top_k_docs if doc in relevant_set)
    return relevant_count / len(relevant_set) if len(relevant_set) > 0 else 0


def f1(sorted_docs: list[tuple[str, float]], relevant_docs: list[str], k: int = 10, beta: float = 1) -> float:
    precision_value = precision(sorted_docs, relevant_docs, k)
    recall_value = recall(sorted_docs, relevant_docs, k)
    if precision_value + recall_value == 0:
        return 0
    beta_2 = beta ** 2
    return (1 + beta_2) * (precision_value * recall_value) / (beta_2 * precision_value + recall_value)


def dcg(sorted_docs: list[tuple[str, float]], relevant_docs: list[str], k: int = 10) -> float:
    top_k_docs = [doc_id for doc_id, _ in sorted_docs[:k]]
    relevant_set = set(relevant_docs)
    dcg_value = 0
    for i, doc in enumerate(top_k_docs):
        if doc in relevant_set:
            dcg_value += 1 / math.log2(i + 2)
    return dcg_value


def ndcg(sorted_docs: list[tuple[str, float]], relevant_docs: list[str], k: int = 10) -> float:
    idcg_value = np.log2(np.arange(len(relevant_docs)) + 1 + 1).sum()
    dcg_value = dcg(sorted_docs, relevant_docs, k)
    return dcg_value / idcg_value if idcg_value > 0 else 0


#------------------------------------------------------------#
# Group Metrics
#------------------------------------------------------------#
def __score_sequencial(metric: callable, sorted_docs: list[tuple[str, float]], relevant_docs: list[str], k: int = 10, *args, **kwargs) -> float:
    score = 0
    for query_id, query_sorted_docs in sorted_docs.items():
        query_relevant_docs = relevant_docs.get(query_id, [])
        score += metric(query_sorted_docs, query_relevant_docs, k, *args, **kwargs)
    return score / len(sorted_docs) if len(sorted_docs) > 0 else 0


def mrr_score(sorted_docs: dict[str, list[tuple[str, float]]], relevant_docs: dict[str, list[str]], k: int = 10, parallel: bool = False) -> float:
    if not parallel:
        return __score_sequencial(reciprocal_rank, sorted_docs, relevant_docs, k)
    else:
        raise NotImplementedError("Parallel computation is not implemented yet.")


def map_score(sorted_docs: dict[str, list[tuple[str, float]]], relevant_docs: dict[str, list[str]], k: int = 10, parallel: bool = False) -> float:
    if not parallel:
        return __score_sequencial(precision, sorted_docs, relevant_docs, k)
    else:
        raise NotImplementedError("Parallel computation is not implemented yet.")


def mr_score(sorted_docs: dict[str, list[tuple[str, float]]], relevant_docs: dict[str, list[str]], k: int = 10, parallel: bool = False) -> float:
    if not parallel:
        return __score_sequencial(recall, sorted_docs, relevant_docs, k)
    else:
        raise NotImplementedError("Parallel computation is not implemented yet.")


def mf1_score(sorted_docs: dict[str, list[tuple[str, float]]], relevant_docs: dict[str, list[str]], k: int = 10, beta: float = 1, parallel: bool = False) -> float:
    if not parallel:
        return __score_sequencial(f1, sorted_docs, relevant_docs, k, beta)
    else:
        raise NotImplementedError("Parallel computation is not implemented yet.")


def mndcg_score(sorted_docs: dict[str, list[tuple[str, float]]], relevant_docs: dict[str, list[str]], k: int = 10, parallel: bool = False) -> float:
    if not parallel:
        return __score_sequencial(ndcg, sorted_docs, relevant_docs, k)
    else:
        raise NotImplementedError("Parallel computation is not implemented yet.")
