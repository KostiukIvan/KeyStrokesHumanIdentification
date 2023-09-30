# KeyStrokesHumanIdentification

## Prerequisites 
* Python 3.8.11

## Installation
Run following comands in root dir:
* `python -m venv .venv`
* `source .venv/bin/activate`
* `pip install -r requirements.txt`
* `bash scripts/download_data.sh` - it will install dataset in `dataset` dir

### To run test
`pytest -vvv src/knn_model/test_model.py`

### To start app
`python src/knn_model/run.py`
