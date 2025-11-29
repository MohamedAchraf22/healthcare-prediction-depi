import pickle

with open("stroke_model.pkl", "rb") as f:
    saved = pickle.load(f)

pipeline = saved["pipeline"]
model = saved["model"]
threshold = saved["threshold"]

import pandas as pd

def process_input(payload: dict):
    df = pd.DataFrame([payload])
    X = pipeline.transform(df)
    return X


def predict(payload: dict):
    X = process_input(payload)
    proba = model.predict_proba(X)[0][1]
    prediction = int(proba >= threshold)

    return {
        "stroke_probability": round(float(proba), 4),
        "final_prediction": "High Risk - Stroke Likely" if prediction == 1 else "Low Risk - No Stroke",
        "model_decision": "Positive (Stroke)" if prediction == 1 else "Negative (No Stroke)"
    }
