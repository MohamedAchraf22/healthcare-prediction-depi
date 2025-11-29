import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pipeline import preprocessing_pipeline
import pickle

df_raw = pd.read_csv("datasets/raw/dataset.csv")

X = df_raw.drop('stroke', axis=1)
y = df_raw['stroke']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessing_pipeline.fit(X_train_raw)

X_train = preprocessing_pipeline.transform(X_train_raw)
X_test = preprocessing_pipeline.transform(X_test_raw)

final_model = LogisticRegression(
    penalty='l1',
    C=0.3,
    class_weight='balanced',
    solver='saga',
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_train, y_train)

artifact = {
    "pipeline": preprocessing_pipeline,
    "model": final_model,
    "threshold": 0.6027
}

with open("stroke_model.pkl", "wb") as f:
    pickle.dump(artifact, f)

