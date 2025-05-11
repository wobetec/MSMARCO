# MSMARCO

This repo is a study of reraking approachs for the MSMARCO passage ranking task. We use the [MS MARCO](https://microsoft.github.io/msmarco/) dataset and the [IR Datasets](https://ir-datasets.com/msmarco-passage-v2.html) loader to load the data.

This repo main use Python with PyTorch and HuggingFace Transformers to implement the models. We also use a litte bit of Java with pyserini interface to run BM25.

### Table of Contents

- [Table of Contents](#table-of-contents)
- [Theory](#theory)
  - [Retrivers](#retrivers)
    - [BM25](#bm25)
    - [Faiss](#faiss)
  - [Rerankers](#rerankers)
    - [MonoBERT](#monobert)
    - [SentenceTransformerSimilarity](#sentencetransformersimilarity)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
  - [Load Datasets](#load-datasets)
  - [Run Experiments](#run-experiments)
  - [Compile results](#compile-results)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Theory

We focus multistage reranking approachs. This means that we use a first stage to get a set of candidates using a fast way and then we use a second stage to rerank the candidates.

Our objective in the first layer is to maximize the recall, making sure that we get all the relevant documents. In the second layer, we use a more complex model to rerank the candidates trying to increase the precision.

### Retrivers

For the first layers, called retriver, we basicly use BM25 with differente implementations and Faiss.

#### BM25

Regarding BM25, we use 3 different implementations:

1. **BM25**: The original BM25 implementation from python `rank_bm25`
2. **BM25Fast**: A faster implementation of BM25 using `numba` and `numpy`. This implementation is faster than the original one but it is not as accurate.
3. **BM25PySerini**: A Java implementation of BM25 using `pyserini`. This implementation is the most accurate one and it is also the fastest one. We use `pyserini` to run the Java code and get the results.

A simple resume of BM25 logic is that it use words frequency, inverse document frequency and some normalizations to calculate a score for each document. It is a fast way - depending on the implementation - to get a set of candidates with very high reall - also depending on the implementation due to text preprocessing. You can see a breif comparison between implementations and the impact of preprocessing in the results notebook.

#### Faiss

Faiss is a librar that implements Facebook AI Similarity Search. It uses embeddings to calculate the similarity between documents with some clever jumps to make it even fast. It is a very fast way to get a set of candidates and like we discussed in results it have enen high MRR than our best BM25. Going further it is almost better than BM25 + Similarity.

### Rerankers

For the second layer, called reranker, we implemented and test some different models:

#### MonoBERT

MonoBERT is a BERT-based model that uses the BERT embeddings to calculate the similarity between documents. The bad thing about this model is that it is very slow since it needs to compute the embeddings for each document+query at runtime.

#### SentenceTransformerSimilarity

Here is like Faiss. It precomputes the embeddings and uses them to calculate the similarity between documents and the query embedding. The difference is that it uses a different model to compute the embeddings. It is also very fast.

## Tutorials

We provide some tutorials to run some specific models or implement some specific techniques that we learn during the project. You can find them in the [`tutorials`](/tutorials) folder.

## Installation

We recommend using a virtual environment to avoid dependency conflicts and use `python >= 3.10`.

```bash
pip install -r requirements.txt
```

Check [pytorch](https://pytorch.org/get-started/locally/) to install cuda if you want.

## Data

Our data is shared [internally](https://gvmail.sharepoint.com/sites/DatasetsProjetos). Place it in the `data` folder.

## Usage

We encapsulate the experiments run inside [`experiments.py`](experiments.py) CLI. You can run available experiments using the following commands:

### Load Datasets

First of all, you need to load the datasets. Like we said on the [Data](#data) section, we provide the data internally. After you download it, you can load the datasets using the following command:

```bash
python experiments.py load_datasets
```

### Run Experiments

We create the framework in such way that you only need to run each layer/stage once. This means that if you run BM25 with dataset X, and after you want to run BM25 with dataset X and a new layer like MonoBERT, the framework will use the BM25 results cached (it will be stored in the `results/raw` folder).

The CLI interface is like:

```bash
python experiments.py run_experiment [dataset] [preproc] [*models]
```

Where:

* `dataset` is the subset:
    - `subset_msmarco_train_0.01_99`: 1% of the queries and using relevant docs plus 99 other docs.

* `preproc` is the preprocessing function:
    - `none`
    - `lower`: lower case
    - `full`: lower case, remove stop words, remove punctuation and do stemming

* `models` are the models you want to run (separated with spaces but if you want to run multiples in a single layer and create branches use `-`):
    - BM25
    - BM25Fast
    - BM25PySerini
    - Faiss
    - SentenceTransformerSimilarity
    - MonoBatchBERT
    - MonoBERT

Here is an example to run BM25 + SenteceTransformerSimilarity:

```bash
python experiments.py subset_msmarco_train_0.01_99 none BM25 SentenceTransformerSimilarity
```

And here how to run 2 experiments at once BM25 + MonoBERT and BM25 + MonoBatchBERT:

```bash
python experiments.py subset_msmarco_train_0.01_99 none BM25 MonoBERT-MonoBatchBERT
```

### Compile results

We provide a comand to compile results to some simple tables with some usefull metrics(out main results are placed in [results](#results) section).

Use the following command to compile and append results:

```bash
python experiments.py compile_results
```

## Results

You can find the compiled results in the `results/clean` folder. The results are main stored in `csv` files. You can also find a notebook with some tables and plots of results in the [`results.ipynb`](results.ipynb).

Here is our main results:

| dataset                      | preproc | Layer 1      | Layer 2                       | mrr  | map  | mr   | mf1  | mndcg | time_avg | time_std | time_max | time_min |
|------------------------------|---------|--------------|-------------------------------|------|------|------|------|-------|----------|----------|----------|----------|
| subset_msmarco_train_0.01_99 | none    | BM25Fast     |                               | 0.19 | 0.02 | 0.18 | 0.03 | 0.18  | 0.09     | 0.07     | 0.42     | 0.00     |
| subset_msmarco_train_0.01_99 | none    | BM25         |                               | 0.24 | 0.04 | 0.36 | 0.07 | 0.26  | 0.82     | 0.28     | 3.30     | 0.38     |
| subset_msmarco_train_0.01_99 | lower   | BM25Fast     |                               | 0.38 | 0.04 | 0.37 | 0.07 | 0.37  | 0.09     | 0.08     | 0.51     | 0.00     |
| subset_msmarco_train_0.01_99 | full    | BM25Fast     |                               | 0.43 | 0.04 | 0.42 | 0.08 | 0.41  | 0.03     | 0.03     | 0.16     | 0.00     |
| subset_msmarco_train_0.01_99 | lower   | BM25         |                               | 0.47 | 0.07 | 0.66 | 0.12 | 0.51  | 0.84     | 0.26     | 2.08     | 0.38     |
| subset_msmarco_train_0.01_99 | full    | BM25         |                               | 0.52 | 0.07 | 0.72 | 0.13 | 0.56  | 0.66     | 0.19     | 1.54     | 0.37     |
| subset_msmarco_train_0.01_99 | full    | BM25PySerini |                               | 0.53 | 0.07 | 0.72 | 0.14 | 0.57  | 0.03     | 0.01     | 0.26     | 0.00     |
| subset_msmarco_train_0.01_99 | lower   | BM25PySerini |                               | 0.56 | 0.08 | 0.75 | 0.14 | 0.59  | 0.03     | 0.02     | 0.39     | 0.00     |
| subset_msmarco_train_0.01_99 | none    | BM25PySerini |                               | 0.56 | 0.08 | 0.75 | 0.14 | 0.59  | 0.04     | 0.07     | 0.99     | 0.01     |
| subset_msmarco_train_0.01_99 | none    | Faiss        |                               | 0.71 | 0.09 | 0.88 | 0.16 | 0.73  | 0.04     | 0.01     | 0.23     | 0.03     |
| subset_msmarco_train_0.01_99 | none    | BM25PySerini | SentenceTransformerSimilarity | 0.71 | 0.09 | 0.87 | 0.16 | 0.73  | 0.01     | 0.03     | 0.61     | 0.01     |

## Acknowledgements

* [MS MARCO](https://microsoft.github.io/msmarco/) for the dataset.
* [IR Datasets](https://ir-datasets.com/msmarco-passage-v2.html) for the dataset loader.

* [Transformers](https://huggingface.co/docs/transformers/index) for the models and tokenizers.
* [Rankify](https://github.com/DataScienceUIBK/Rankify) for the implementation of almost all used llm models.
* [Vespa](https://vespa.ai/)
