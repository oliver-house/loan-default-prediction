# Loan Default Prediction

Predicts loan default risk using loan applications together with alternative financial data.

A blended ensemble (LightGBM + XGBoost + CatBoost) trained on ~950 features extracted from 8 relational tables, with feature engineering and 5-fold stratified cross-validation.

## Results

| Model | OOF AUC | Held-out Test AUC | Weight |
|-------|---------|-------------------|--------|
| LightGBM | 0.7894 | — | 0.12 |
| XGBoost | 0.7926 | — | 0.46 |
| CatBoost | 0.7921 | — | 0.42 |
| **Ensemble** | **0.7947** | **0.7885** | — |

## Repository Structure

```
train.py                    # Entry point: feature engineering, training, weight tuning, predictions
tune.py                     # LightGBM hyperparameter tuning with Optuna
src/
  config.py                 # Paths, constants, model parameters
  features/
    pipeline.py             # Orchestrates feature engineering
    application.py          # Main application table features
    bureau.py               # Credit bureau features
    previous_application.py # Previous Home Credit application features
    pos_cash.py             # POS cash balance features
    installments.py         # Instalment payment features
    credit_card.py          # Credit card balance features
  models/
    lgbm_model.py           # LightGBM training
    xgb_model.py            # XGBoost training
    catboost_model.py       # CatBoost training
  utils/
    helpers.py              # Logging, memory reduction, timing
```

## Data

Download the data from [here](https://www.kaggle.com/competitions/home-credit-default-risk/data) and place the following files in `data/` in the project root:

```
data/
  application_train.csv
  application_test.csv
  bureau.csv
  bureau_balance.csv
  POS_CASH_balance.csv
  credit_card_balance.csv
  previous_application.csv
  installments_payments.csv
```

## Requirements

Python 3.13. Dependencies listed in `requirements.txt`.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python train.py
```

Optionally tune LightGBM hyperparameters before training:

```powershell
python tune.py --trials 50 --folds 3
```

Best params are saved to `predictions/lgbm_best_params.json` for reference.
