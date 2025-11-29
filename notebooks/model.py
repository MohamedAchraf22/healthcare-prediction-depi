#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score, roc_auc_score, f1_score, confusion_matrix, precision_score


# In[3]:


df = pd.read_csv("datasets/processed/dataset.csv")

df.head()


# In[4]:


X = df.drop('stroke', axis=1)
y = df['stroke']


# In[ ]:


if 'Unnamed: 0' in X.columns:
    X = X.drop('Unnamed: 0', axis=1)



# In[6]:


X_train, X_test, y_train, y_test = train_test_split( X, y,test_size=0.2,random_state=42, stratify=y)


# In[7]:


if 'Unnamed: 0' in X_train.columns:
    X_train = X_train.drop('Unnamed: 0', axis=1)

if 'Unnamed: 0' in X_test.columns:
    X_test = X_test.drop('Unnamed: 0', axis=1)


# In[8]:


# numerical_cols = ['age', 'avg_glucose_level', 'bmi', 'age_bmi_interaction']
# scaler = StandardScaler()


# # In[9]:


# X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
# X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])


# In[10]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

# ------------------------------------------------------------------

log_reg = LogisticRegression(
    penalty='l1',
    C=0.3,
    class_weight='balanced',
    solver='saga',
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)
log_reg.fit(X_train, y_train)

y_proba = log_reg.predict_proba(X_test)[:, 1]



# ------------------------------------------------------------------
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

target_recall = 0.78
valid_idx = np.where(recalls >= target_recall)[0]

f1_scores = 2 * precisions[valid_idx] * recalls[valid_idx] / (
    precisions[valid_idx] + recalls[valid_idx] + 1e-12)

best_idx_global = valid_idx[np.argmax(f1_scores)]
BEST_THRESHOLD = thresholds[best_idx_global]

y_pred_custom = (y_proba >= BEST_THRESHOLD).astype(int)



# ------------------------------------------------------------------
print("Logistic Regression (L1 + C=0.3 + class_weight='balanced')")
print("="*80)
print(classification_report(y_test, y_pred_custom, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_custom))
print(f"\nBest Threshold = {BEST_THRESHOLD:.4f}  →  Recall ≥ {target_recall}")