import pandas as pd

df = pd.read_csv("datasets/raw/dataset.csv")
df_clean = df.copy()

# Create age groups for more accurate imputation
df_clean['age_group'] = pd.cut(df_clean['age'], 
                                     bins=[0, 18, 35, 50, 65, 100],
                                     labels=['<18', '18-35', '36-50', '51-65', '65+'])
    
# Impute BMI with median by age group and gender
df_clean['bmi'] = df_clean.groupby(['age_group', 'gender'], observed=False)['bmi'].transform(
    lambda x: x.fillna(x.median())
)

# For any remaining missing values, use overall median
df_clean['bmi'] = df_clean['bmi'].fillna(df_clean['bmi'].median())

# Remove duplicates 
df_clean = df_clean.drop_duplicates(subset=['id'], keep='first')

# Handle outliers
df_clean = df_clean[(df_clean['bmi'] >= 12) & (df_clean['bmi'] <= 60)]
    
# Drop category 'Other' from gender
df_clean = df_clean.drop(df_clean[df_clean['gender'] == 'Other'].index, axis=0)

# Remove temp column
df_clean = df_clean.drop('age_group', axis=1)


# Feature engineering
df_clean.drop(['id'],axis=1,inplace=True)
df_clean['age_group'] = pd.cut(df_clean['age'], bins=[0, 18, 40, 60, 100], labels=['Child', 'Adult', 'Middle-Aged', 'Senior'])
df_clean['bmi_class'] = pd.cut(df_clean['bmi'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
df_clean['glucose_risk'] = pd.cut(df_clean['avg_glucose_level'], bins=[0, 140, 200, 300], labels=['Normal', 'Prediabetes', 'Diabetes'])

df_clean['age_bmi_interaction'] = df_clean['age'] * df_clean['bmi']
df_clean['cardiovascular_risk_score'] = df_clean['hypertension'] + df_clean['heart_disease']
df_clean['married_and_adult'] = ((df_clean['ever_married'] == 'Yes') & (df_clean['age'] > 18)).astype(int)


output_name = "dataset.csv"
df_clean.to_csv("datasets/processed/" + output_name)
