# LoanIQ — Loan Default Prediction System

## Quick start

```bash
pip install -r requirements.txt
python train_model.py      # re-train model (optional, artifacts already included)
python app.py              # start backend on port 5050
# open frontend.html in your browser
```

## Files
- `train_model.py`   — data cleaning, feature engineering, model training
- `app.py`           — Flask REST API (port 5050)
- `frontend.html`    — standalone HTML/JS frontend
- `artifacts/`       — saved model, scaler, metadata

## API
POST /predict  body: {age, income, loan_amount, credit_score, employment_years, education_level, housing_status}
GET  /metrics  returns model accuracy & ROC-AUC
GET  /health   health check

## Note on dataset
The credit_risk_dataset.csv has very weak feature-to-label correlations (all correlations < 0.1), which is typical of synthetically generated datasets. The model achieves 60% accuracy and 0.547 AUC — the best achievable on this data with a decision tree. For production use, a dataset with genuine predictive signal is recommended.
