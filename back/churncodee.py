pip install scikit-learn

from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv("kaggledata.csv")

data.head

def label_encode_columns(df, columns):
    label_encoder = LabelEncoder()
    for col in columns:
        df[col] = label_encoder.fit_transform(df[col])
    return df

df = pd.DataFrame(data)

columns_to_encode = ['country', 'gender']
df_encoded = label_encode_columns(df, columns_to_encode)

print(df_encoded)

features  = ['credit_score', 'country', 'gender', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary']

X = data[features].copy()
y = data['churn'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=324)

# Linear Regression
reg = LinearRegression()
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print(r2_score(y_test,y_pred))
print(mean_squared_error(y_test, y_pred))

# Elastic Net Regression
reg = ElasticNet(alpha=0.00001,l1_ratio=0.1)
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print(r2_score(y_test,y_pred))
print(mean_squared_error(y_test, y_pred))

# Random Forest Algorithm 
classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(r2_score(y_test,y_pred))
print(mean_squared_error(y_test, y_pred))

accuracy1 = accuracy_score(y_test, y_pred)
results2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

accuracy1

# Logistic Regression Classifier

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

#note: accuracy = no of correct predictions / total number of predictions 

results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

results
accuracy

# Logistic Regression Classifier 

classifier_lr = LogisticRegression(random_state=0)

classifier_lr.fit(X_train, y_train)

coefficients = classifier_lr.coef_[0]

feature_names = X_train.columns.tolist()

indices = np.argsort(np.abs(coefficients))[::-1]

print("Feature ranking:")
for f in range(X_train.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, feature_names[indices[f]], coefficients[indices[f]]))

classifier_rf = RandomForestClassifier(n_estimators=100, random_state=0)

classifier_rf.fit(X_train, y_train)

importances = classifier_rf.feature_importances_

feature_names = X_train.columns.tolist()  # Assuming X_train is a pandas DataFrame

indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for f in range(X_train.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))

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

from sklearn.decomposition import PCA

pca = PCA(n_components=7)  # Number of components to keep
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

pd.DataFrame(X_test_pca)

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression()
rfe = RFE(estimator=logistic_regression, n_features_to_select=5)  
rfe.fit(X_train, y_train)

selected_features = X_train.columns[rfe.support_]
print("Selected Features:")
print(selected_features)

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
