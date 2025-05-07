# MSMARCO

This repo is a study of raking models for the MS MARCO passage ranking task.

## Installation

We recommend using a virtual environment to avoid dependency conflicts and use `python >= 3.10`.

```bash
pip install -r requirements.txt
```

Check [pytorch](https://pytorch.org/get-started/locally/) to install cuda if you want.

## Data

Our data is shared [internally](https://gvmail.sharepoint.com/sites/DatasetsProjetos). Place it in the `data` folder.

## Usage

To basic overview of available tools take a look at [`basic_usage.ipynb`](basic_usage.ipynb).

## History

To track out evolution we're writing the seps we took in [`history.md`](history.md).

## Acknowledgements

* [MS MARCO](https://microsoft.github.io/msmarco/) for the dataset.
* [IR Datasets]() for the dataset loader.

* [Transformers](https://huggingface.co/docs/transformers/index) for the models and tokenizers.
* [Rankify](https://github.com/DataScienceUIBK/Rankify) for the implementation of almost all used llm models.

## Rodando Vespa

docker exec -it vespa bash -c "
  cd /home/vespa/application &&
  vespa-deploy prepare . &&
  vespa-deploy activate
"
 
## Query teste

curl -sG 'http://localhost:8080/search/' \
  --data-urlencode "yql=select * from sources msmarco where userInput('your example query');" \
  --data-urlencode "hits=5" \
  --data-urlencode "ranking.profile=bm25-profile" \
| jq .

