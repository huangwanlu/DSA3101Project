pip install scikit-learn

from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
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

data = pd.read_csv("kaggledata.csv")

data.head

# Column Encoding 

def label_encode_columns(df, columns):
    label_encoder = LabelEncoder()
    for col in columns:
        df[col] = label_encoder.fit_transform(df[col])
    return df


df = pd.DataFrame(data)
columns_to_encode = ['country', 'gender']
df_encoded = label_encode_columns(df, columns_to_encode)
print(df_encoded)


features  = ['credit_score', 'country', 'gender', 'age', 'tenure', 'balance', 'products_number',
             'credit_card', 'active_member', 'estimated_salary']

X = data[features].copy()
y = data['churn'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=324)


# Logistic Regression Classifier

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

#note: accuracy = no of correct predictions / total number of predictions 

results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

results
accuracy

# Logistic Regression Feature Importance Scores 

classifier_lr = LogisticRegression(random_state=0)

classifier_lr.fit(X_train, y_train)
coefficients = classifier_lr.coef_[0]

feature_names = X_train.columns.tolist()
indices = np.argsort(np.abs(coefficients))[::-1]

print("Feature ranking:")
for f in range(X_train.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, feature_names[indices[f]], coefficients[indices[f]]))
