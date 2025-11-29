import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import re

class NormalizeColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X.columns = [re.sub(r'\W+', '_', col.strip().lower()) for col in X.columns]
        return X


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Age group
        X['age_group'] = pd.cut(X['age'], bins=[0,18,40,60,100],
                                labels=['Child','Adult','Middle-Aged','Senior'])

        # BMI imputation
        X['bmi'] = X.groupby(['age_group','gender'], observed=False)['bmi'] \
                    .transform(lambda x: x.fillna(x.median()))
        X['bmi'] = X['bmi'].fillna(X['bmi'].median())

        # DO NOT drop rows — avoid breaking sample alignment
        X['bmi'] = X['bmi'].clip(12, 60)
        X.loc[X['gender'] == 'Other', 'gender'] = 'Female'

        # Derived features
        X['bmi_class'] = pd.cut(X['bmi'], bins=[0,18.5,24.9,29.9,100],
                                labels=['Underweight','Normal','Overweight','Obese'])
        X['glucose_risk'] = pd.cut(X['avg_glucose_level'], bins=[0,140,200,300],
                                   labels=['Normal','Prediabetes','Diabetes'])
        X['age_bmi_interaction'] = X['age'] * X['bmi']
        X['cardiovascular_risk_score'] = X['hypertension'] + X['heart_disease']

        X = X.drop(columns=['id'], errors='ignore')
        return X


numerical_cols = ['age', 'avg_glucose_level', 'bmi', 'age_bmi_interaction']
categorical_cols = [
    'gender', 'ever_married', 'work_type', 'residence_type',
    'smoking_status', 'age_group', 'bmi_class', 'glucose_risk'
]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

preprocessing_pipeline = Pipeline([
    ('normalize', NormalizeColumns()),
    ('feature_eng', FeatureEngineering()),
    ('preprocessor', ColumnTransformer(transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]))
])
