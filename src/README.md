## To Do List
- [ ] **Normalising Tabular Data**: currently handled by Tox21Tabular, however this requires featuring the entire dataset first to the fit the scaler.
- [ ] **Handling Missing Data**: (similar to above) currently handled by Tox21Tabular, however this requires featuring the entire dataset first to find nan values and remove them.
- [ ] **Cache Validation Dataloader**: currently the validation dataloader is recreated each time the model is validated. This is inefficient and should be cached.
