# Loan Default Prediction

Predicts loan default risk from loan application data and alternative sources.

## Status

Early development. Feature engineering is complete across all tables: main application, credit bureau, previous applications, POS cash, instalment payments, and credit card balances (~950 features). Training pipeline is scaffolded but model training is not yet wired up.

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
