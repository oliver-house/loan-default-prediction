# Loan Default Prediction

Predicts loan default risk from loan application data and alternative sources.

## Status

Early development. Feature engineering for the main application table, credit bureau, and previous applications is in place. Training pipeline is scaffolded but not yet wired up.

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
