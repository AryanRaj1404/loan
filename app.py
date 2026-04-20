import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, json, numpy as np

app = Flask(__name__)
CORS(app)

# Load artifacts
model  = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

with open('meta.json') as f:
    meta = json.load(f)

EDU     = meta['education_order']
HOUSING = meta['housing_order']

# Feature builder
def build_features(d):
    age   = float(d['age'])
    inc   = float(d['income'])
    loan  = float(d['loan_amount'])
    cs    = float(d['credit_score'])
    emp   = float(d['employment_years'])
    edu   = float(EDU.get(d['education_level'], 0))
    hs    = float(HOUSING.get(d['housing_status'], 0))

    return [
        age, inc, loan, cs, emp, edu, hs,
        loan / (inc + 1),
        inc  / (emp + 1),
        loan / (cs  + 1),
        (cs  * inc) / (loan + 1),
        age  * emp,
    ]

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        feats = np.array([build_features(data)])
        feats_sc = scaler.transform(feats)

        # 🔥 Use probability instead of raw prediction
        proba = float(model.predict_proba(feats_sc)[0][1])

        # 🔥 Tunable threshold
        threshold = 0.4

        approved = proba < threshold
        pred = int(proba >= threshold)

        # Risk classification
        if proba < 0.30:
            risk = "Low"
        elif proba < 0.55:
            risk = "Medium"
        else:
            risk = "High"

        return jsonify({
            "default_prediction": pred,
            "default_probability": round(proba * 100, 1),
            "risk_level": risk,
            "approved": approved
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


# Metrics route
@app.route('/metrics', methods=['GET'])
def metrics():
    return jsonify(meta['metrics'])


# Health check
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


# Run server
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5050))
    app.run(host='0.0.0.0', port=port, debug=False)