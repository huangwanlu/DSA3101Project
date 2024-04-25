from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv('churn_data.csv')

def label_encode_columns(df, columns):
    label_encoder = LabelEncoder()
    for col in columns:
        df[col] = label_encoder.fit_transform(df[col])
    return df

df = pd.DataFrame(data)

# Specify columns to be label encoded
columns_to_encode = ['country', 'gender']

# Apply label encoding to specified columns
df_encoded = label_encode_columns(df, columns_to_encode)

features  = ['credit_score', 'country', 'gender', 'age', 'tenure', 
            'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary']

X = data[features].copy()
y = data['churn'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=324)

# Random Forest Algorithm 
classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

model_pkl_file = "final_model.pkl"

with open(model_pkl_file, 'wb') as file:
    pickle.dump(classifier, file)