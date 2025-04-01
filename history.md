# MSMARCO - Reranking Task

## Week 0

We started the project by trying to understand `ir_datasets` and learning how to access `msmarco-passages-v2`.

## Week 1

We built a smaller sample of the dataset to make it easier to work with and to ensure that everyone in class is using the same dataset, making results more comparable between students.

## Week 2

The goal was to establish a baseline and develop a more complex system to test our reranking task on the dataset agreed upon by the class. We also needed to decide on the metrics to use. 

### Algorithm

First, we decided to use only `BM25` as a baseline. For the more complex model, we chose `MonoBERT`.

### Metrics

The primary metric is MRR@10, but we also compute MAP@10, MR@10, MF1@10, and MNDCG@10 to gather more insights about the algorithms.

## Week 3 - Current

This week, we decided to finalize our implementation and build a more robust architecture that will help us later. We also want to implement additional models like `Sentence Transformer` and `TourRank`. Lastly, we plan to run experiments testing dataset cleaning techniques and different algorithm pipelines, as well as measuring the time required to run the models.
