#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install scikit-learn


# In[26]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score


# In[3]:


data = pd.read_csv("kaggledata.csv")


# In[4]:


data.head


# In[5]:


def label_encode_columns(df, columns):
    label_encoder = LabelEncoder()
    for col in columns:
        df[col] = label_encoder.fit_transform(df[col])
    return df


# In[6]:


df = pd.DataFrame(data)

columns_to_encode = ['country', 'gender']
df_encoded = label_encode_columns(df, columns_to_encode)

print(df_encoded)


# In[7]:


features  = ['credit_score', 'country', 'gender', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary']


# In[8]:


X = data[features].copy()
y = data['churn'].copy()


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=324)


# In[28]:


# Random Forest Algorithm 
classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

accuracy1 = accuracy_score(y_test, y_pred)
conf_matrix1 = confusion_matrix(y_test, y_pred)

results2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})


# In[29]:


accuracy1


# In[30]:


print(conf_matrix1)


# In[31]:


classification_report(y_test,y_pred)


# In[32]:


n_folds = 5  
cv_scores = cross_val_score(classifier, X_train, y_train, cv=n_folds)

print("Cross-validation scores:", cv_scores)
print("Mean accuracy:", cv_scores.mean())


# In[33]:


# Logistic Regression Classifier

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test,y_pred)

#note: accuracy = no of correct predictions / total number of predictions 

results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})


# In[22]:


results


# In[23]:


accuracy


# In[24]:


conf_matrix


# In[25]:


class_report


# In[34]:


n_folds = 5  
cv_scores = cross_val_score(classifier, X_train, y_train, cv=n_folds)

print("Cross-validation scores:", cv_scores)
print("Mean accuracy:", cv_scores.mean())


# In[18]:


# Logistic Regression Classifier 

classifier_lr = LogisticRegression(random_state=0)

classifier_lr.fit(X_train, y_train)

coefficients = classifier_lr.coef_[0]

feature_names = X_train.columns.tolist()

indices = np.argsort(np.abs(coefficients))[::-1]

print("Feature ranking:")
for f in range(X_train.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, feature_names[indices[f]], coefficients[indices[f]]))


# In[19]:



classifier_rf = RandomForestClassifier(n_estimators=100, random_state=0)

classifier_rf.fit(X_train, y_train)

importances = classifier_rf.feature_importances_

feature_names = X_train.columns.tolist()  # Assuming X_train is a pandas DataFrame

indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for f in range(X_train.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))


# In[20]:


import matplotlib.pyplot as plt

top_features = 5

plt.figure(figsize=(10, 6))
plt.title("Top {} Feature Importances".format(top_features))
plt.bar(range(top_features), importances[indices][:top_features], color="skyblue", align="center")
plt.xticks(range(top_features), [feature_names[i] for i in indices][:top_features], rotation=45, ha="right")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()


# In[21]:


from sklearn.decomposition import PCA

pca = PCA(n_components=7)  # Number of components to keep
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


# In[22]:


pd.DataFrame(X_test_pca)


# In[23]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression()
rfe = RFE(estimator=logistic_regression, n_features_to_select=5)  
rfe.fit(X_train, y_train)

selected_features = X_train.columns[rfe.support_]
print("Selected Features:")
print(selected_features)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

param_grid = { 
    'n_estimators': [100, 200, 300], 
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None]
}

model = RandomForestClassifier(random_state=0) 

grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=5)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_ 
best_score = grid_search.best_score_

print("Best Parameters:", best_params) 
print("Best Score (Accuracy):", best_score)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred)) 
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# 

# In[ ]:




