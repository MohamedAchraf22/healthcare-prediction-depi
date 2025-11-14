import pandas as pd
# from sklearn.preprocessing import LabelEncoder

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

# # Encode categorical features - May be added again later
# encoder = LabelEncoder()
# categorical_cols = df_cleaned.select_dtypes(include=['object']).columns.tolist()
# for col in categorical_cols:
#     df_cleaned[col] = encoder.fit_transform(df_cleaned[col])


output_name = "dataset.csv"
df_clean.to_csv("datasets/processed/" + output_name)
