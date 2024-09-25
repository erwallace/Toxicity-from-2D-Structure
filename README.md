## Questions
- [ ] **Normalising Tabular Data**: currently handled by Tox21Tabular, however this requires featuring the entire dataset first to the fit the scaler.
- [ ] **Handling Missing Data**: (similar to above) currently handled by Tox21Tabular, however this requires featuring the entire dataset first to find nan values and remove them.
- [ ] **Cache Validation Dataloader**: currently the validation dataloader is recreated each time the model is validated. This is inefficient and should be cached.
- [ ] **Local Imports** is there a better way to deal with local imports than installing an editable version?

## v0.1 To Do:
- [ ] sklearn models:
  - [ ] back end to run sklearn models on same architecture
  - [ ] Logistic Regression model
  - [ ] XGBoost
- [ ] simple neural network:
  - [ ] callbacks
  - [ ] cpu/gpu support
- [ ] testing:
  - [ ] unit tests for everything
  - [ ] integration tests for some things

## General Engineering:
- [ ] add logging to the project.
- [ ] version_log.md
- [x] requirements.in
- [x] gitignore
- [x] pyproject.yaml
- [ ] CI pipeline

# Setup

## Install
```
conda env create -f env.yaml
conda activate toxic2d
pip install -e .
pre-commit install
```
