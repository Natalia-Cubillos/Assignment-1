# Assignment-1
Assignment 1, Predictive Modelling of Eating-Out Problem

Prerequisites
Python 3.10+ (project tested with Python 3.12)
Git
DVC (pip install dvc)
Git LFS (for large files, optional)

### clone the repo
git clone https://github.com/<your-user>/Assignment-1.git
cd Assignment-1
### create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate
### install Python packages
pip install -r requirements.txt
### pull data/models tracked by DVC (downloads to local workspace)
dvc pull

### make sure the venv is active
source .venv/bin/activate
### see pipeline graph (optional)
dvc dag
### run all stages
dvc repro
### show collected metrics
dvc metrics show
### commit code and pipeline state
git add dvc.yaml dvc.lock src/*.py params.yaml 'metrics/*.json'
git commit -m "Reproduce pipeline and update metrics"
git push
### push data/models to DVC remote
dvc push

# Expected Results
regression: MSE, R2, train_time_s, infer_time_s
classification: Accuracy, train_time_s, infer_time_s












