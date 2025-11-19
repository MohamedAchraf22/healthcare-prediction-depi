import pandas as pd
from sklearn.preprocessing import OneHotEncoder ,LabelEncoder
from sklearn.preprocessing import StandardScaler
import re

df = pd.read_csv("datasets/raw/dataset.csv")

def normalize_columns(df):
    df.columns = [re.sub(r'\W+', '_', col.strip().lower()) for col in df.columns]
    return df

df = normalize_columns(df)

df_clean = df.copy()

# Create age groups for more accurate imputation
df_clean['age_group'] = pd.cut(df_clean['age'], bins=[0, 18, 40, 60, 100], labels=['Child', 'Adult', 'Middle-Aged', 'Senior'])
    
# Impute BMI with median by age group and gender
df_clean['bmi'] = df_clean.groupby(['age_group', 'gender'], observed=False)['bmi'].transform(
    lambda x: x.fillna(x.median())
)

# For any remaining missing values, use overall median
df_clean['bmi'] = df_clean['bmi'].fillna(df_clean['bmi'].median())

df_clean = df_clean.drop_duplicates(subset=['id'], keep='first')
df_clean = df_clean[(df_clean['bmi'] >= 12) & (df_clean['bmi'] <= 60)]

# Drop 'Other' gender if present
df_clean = df_clean[df_clean['gender'] != 'Other']

# Feature engineering
df_clean.drop(['id'],axis=1,inplace=True)
df_clean['bmi_class'] = pd.cut(df_clean['bmi'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
df_clean['glucose_risk'] = pd.cut(df_clean['avg_glucose_level'], bins=[0, 140, 200, 300], labels=['Normal', 'Prediabetes', 'Diabetes'])

df_clean['age_bmi_interaction'] = df_clean['age'] * df_clean['bmi']
df_clean['cardiovascular_risk_score'] = df_clean['hypertension'] + df_clean['heart_disease']

# Encoding

categorical_cols = [
    'gender', 'ever_married', 'work_type', 'residence_type', 
    'smoking_status', 'age_group', 'bmi_class', 'glucose_risk'
]
df_clean = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)

# Feature Scaling

numerical_cols = ['age', 'avg_glucose_level', 'bmi', 'age_bmi_interaction']
df_scaled = df_clean.copy()
scaler = StandardScaler()
df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])

# Save Dataset
output_name = "dataset.csv"
df_scaled.to_csv("datasets/processed/" + output_name , index=False)