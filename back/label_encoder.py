import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#Read in training data
data = pd.read_csv('churn_data.csv')

#Fitting Label Encoder for 'country' column
country_le = LabelEncoder()
data['country'] = country_le.fit_transform(data['country'])

#Exporting Label Encoder for 'country' column
with open('country_encoder.pkl', 'wb') as file:
    pickle.dump(country_le, file)

#Fitting Label Encoder for 'gender' column
gender_le = LabelEncoder()
data['gender'] = gender_le.fit_transform(data['gender'])

#Exporting Label Encoder for 'gender' column
with open('gender_encoder.pkl', 'wb') as file:
    pickle.dump(gender_le, file)