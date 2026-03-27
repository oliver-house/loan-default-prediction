# Loan Default Prediction

Predicts loan default risk from loan application data and alternative sources.

## Status

In development. Feature engineering is complete across all tables: main application, credit bureau, previous applications, POS cash, instalment payments, and credit card balances (~950 features). LightGBM, XGBoost, and CatBoost models are in place with an ensemble blend, but training is not yet fully wired up.

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
