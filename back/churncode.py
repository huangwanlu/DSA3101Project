{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: scikit-learn in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (1.4.1.post1)\n",
      "Requirement already satisfied: joblib>=1.2.0 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from scikit-learn) (1.3.2)\n",
      "Requirement already satisfied: threadpoolctl>=2.0.0 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from scikit-learn) (3.4.0)\n",
      "Requirement already satisfied: numpy<2.0,>=1.19.5 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from scikit-learn) (1.23.1)\n",
      "Requirement already satisfied: scipy>=1.6.0 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from scikit-learn) (1.12.0)\n",
      "\u001b[33mWARNING: You are using pip version 21.3.1; however, version 24.0 is available.\n",
      "You should consider upgrading via the '/usr/local/bin/python3.10 -m pip install --upgrade pip' command.\u001b[0m\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install scikit-learn\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet, LogisticRegression\n",
    "from sklearn.model_selection import train_test_split, GridSearchCV\n",
    "from sklearn.metrics import r2_score\n",
    "from sklearn.metrics import mean_squared_error\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = pd.read_csv(\"kaggledata.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<bound method NDFrame.head of       customer_id  credit_score  country  gender  age  tenure    balance  \\\n",
       "0        15634602           619   France  Female   42       2       0.00   \n",
       "1        15647311           608    Spain  Female   41       1   83807.86   \n",
       "2        15619304           502   France  Female   42       8  159660.80   \n",
       "3        15701354           699   France  Female   39       1       0.00   \n",
       "4        15737888           850    Spain  Female   43       2  125510.82   \n",
       "...           ...           ...      ...     ...  ...     ...        ...   \n",
       "9995     15606229           771   France    Male   39       5       0.00   \n",
       "9996     15569892           516   France    Male   35      10   57369.61   \n",
       "9997     15584532           709   France  Female   36       7       0.00   \n",
       "9998     15682355           772  Germany    Male   42       3   75075.31   \n",
       "9999     15628319           792   France  Female   28       4  130142.79   \n",
       "\n",
       "      products_number  credit_card  active_member  estimated_salary  churn  \n",
       "0                   1            1              1         101348.88      1  \n",
       "1                   1            0              1         112542.58      0  \n",
       "2                   3            1              0         113931.57      1  \n",
       "3                   2            0              0          93826.63      0  \n",
       "4                   1            1              1          79084.10      0  \n",
       "...               ...          ...            ...               ...    ...  \n",
       "9995                2            1              0          96270.64      0  \n",
       "9996                1            1              1         101699.77      0  \n",
       "9997                1            0              1          42085.58      1  \n",
       "9998                2            1              0          92888.52      1  \n",
       "9999                1            1              0          38190.78      0  \n",
       "\n",
       "[10000 rows x 12 columns]>"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "def label_encode_columns(df, columns):\n",
    "    label_encoder = LabelEncoder()\n",
    "    for col in columns:\n",
    "        df[col] = label_encoder.fit_transform(df[col])\n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "      customer_id  credit_score  country  gender  age  tenure    balance  \\\n",
      "0        15634602           619        0       0   42       2       0.00   \n",
      "1        15647311           608        2       0   41       1   83807.86   \n",
      "2        15619304           502        0       0   42       8  159660.80   \n",
      "3        15701354           699        0       0   39       1       0.00   \n",
      "4        15737888           850        2       0   43       2  125510.82   \n",
      "...           ...           ...      ...     ...  ...     ...        ...   \n",
      "9995     15606229           771        0       1   39       5       0.00   \n",
      "9996     15569892           516        0       1   35      10   57369.61   \n",
      "9997     15584532           709        0       0   36       7       0.00   \n",
      "9998     15682355           772        1       1   42       3   75075.31   \n",
      "9999     15628319           792        0       0   28       4  130142.79   \n",
      "\n",
      "      products_number  credit_card  active_member  estimated_salary  churn  \n",
      "0                   1            1              1         101348.88      1  \n",
      "1                   1            0              1         112542.58      0  \n",
      "2                   3            1              0         113931.57      1  \n",
      "3                   2            0              0          93826.63      0  \n",
      "4                   1            1              1          79084.10      0  \n",
      "...               ...          ...            ...               ...    ...  \n",
      "9995                2            1              0          96270.64      0  \n",
      "9996                1            1              1         101699.77      0  \n",
      "9997                1            0              1          42085.58      1  \n",
      "9998                2            1              0          92888.52      1  \n",
      "9999                1            1              0          38190.78      0  \n",
      "\n",
      "[10000 rows x 12 columns]\n"
     ]
    }
   ],
   "source": [
    "df = pd.DataFrame(data)\n",
    "\n",
    "# Specify columns to be label encoded\n",
    "columns_to_encode = ['country', 'gender']\n",
    "\n",
    "# Apply label encoding to specified columns\n",
    "df_encoded = label_encode_columns(df, columns_to_encode)\n",
    "\n",
    "# Display the encoded DataFrame\n",
    "print(df_encoded)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "features  = ['credit_score', 'country', 'gender', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = data[features].copy()\n",
    "y = data['churn'].copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=324)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.14034246737875633\n",
      "0.13520672181628607\n"
     ]
    }
   ],
   "source": [
    "# Linear Regression\n",
    "reg = LinearRegression()\n",
    "reg.fit(X_train,y_train)\n",
    "y_pred = reg.predict(X_test)\n",
    "print(r2_score(y_test,y_pred))\n",
    "print(mean_squared_error(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.14034255265436524\n",
      "0.1352067084041596\n"
     ]
    }
   ],
   "source": [
    "# Elastic Net Regression\n",
    "reg = ElasticNet(alpha=0.00001,l1_ratio=0.1)\n",
    "reg.fit(X_train,y_train)\n",
    "y_pred = reg.predict(X_test)\n",
    "print(r2_score(y_test,y_pred))\n",
    "print(mean_squared_error(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.15437302004867126\n",
      "0.133\n"
     ]
    }
   ],
   "source": [
    "# Random Forest Algorithm \n",
    "classifier = RandomForestClassifier(n_estimators=100, random_state=0)\n",
    "classifier.fit(X_train, y_train)\n",
    "y_pred = classifier.predict(X_test)\n",
    "print(r2_score(y_test,y_pred))\n",
    "print(mean_squared_error(y_test, y_pred))\n",
    "\n",
    "accuracy1 = accuracy_score(y_test, y_pred)\n",
    "\n",
    "results2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.867"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/sklearn/linear_model/_logistic.py:469: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    }
   ],
   "source": [
    "# Logistic Regression Classifier\n",
    "\n",
    "classifier = LogisticRegression(random_state=0)\n",
    "classifier.fit(X_train, y_train)\n",
    "y_pred = classifier.predict(X_test)\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "\n",
    "#note: accuracy = no of correct predictions / total number of predictions \n",
    "\n",
    "results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Actual</th>\n",
       "      <th>Predicted</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>4158</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1915</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6225</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8017</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9847</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6640</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5450</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8167</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1156</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4368</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>2000 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      Actual  Predicted\n",
       "4158       0          0\n",
       "1915       0          0\n",
       "6225       0          0\n",
       "8017       0          1\n",
       "9847       0          0\n",
       "...      ...        ...\n",
       "6640       0          0\n",
       "5450       0          0\n",
       "8167       0          0\n",
       "1156       0          0\n",
       "4368       0          0\n",
       "\n",
       "[2000 rows x 2 columns]"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.804"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Feature ranking:\n",
      "1. feature active_member (-0.173029)\n",
      "2. feature gender (-0.109613)\n",
      "3. feature products_number (-0.065913)\n",
      "4. feature age (0.052720)\n",
      "5. feature tenure (-0.040215)\n",
      "6. feature credit_card (-0.033840)\n",
      "7. feature country (0.013412)\n",
      "8. feature credit_score (-0.005463)\n",
      "9. feature balance (0.000005)\n",
      "10. feature estimated_salary (0.000001)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/sklearn/linear_model/_logistic.py:469: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    }
   ],
   "source": [
    "# Logistic Regression Classifier \n",
    "\n",
    "classifier_lr = LogisticRegression(random_state=0)\n",
    "\n",
    "# Train the classifier\n",
    "classifier_lr.fit(X_train, y_train)\n",
    "\n",
    "# Extract coefficients\n",
    "coefficients = classifier_lr.coef_[0]\n",
    "\n",
    "feature_names = X_train.columns.tolist()\n",
    "\n",
    "# Get the indices of features sorted by absolute coefficient values\n",
    "indices = np.argsort(np.abs(coefficients))[::-1]\n",
    "\n",
    "# Print the feature ranking\n",
    "print(\"Feature ranking:\")\n",
    "for f in range(X_train.shape[1]):\n",
    "    print(\"%d. feature %s (%f)\" % (f + 1, feature_names[indices[f]], coefficients[indices[f]]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Feature ranking:\n",
      "1. feature age (0.236266)\n",
      "2. feature credit_score (0.146967)\n",
      "3. feature estimated_salary (0.145848)\n",
      "4. feature balance (0.137502)\n",
      "5. feature products_number (0.133879)\n",
      "6. feature tenure (0.081124)\n",
      "7. feature active_member (0.044073)\n",
      "8. feature country (0.037921)\n",
      "9. feature credit_card (0.018867)\n",
      "10. feature gender (0.017553)\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# Instantiate the random forest classifier\n",
    "classifier_rf = RandomForestClassifier(n_estimators=100, random_state=0)\n",
    "\n",
    "# Train the classifier\n",
    "classifier_rf.fit(X_train, y_train)\n",
    "\n",
    "# Extract feature importances\n",
    "importances = classifier_rf.feature_importances_\n",
    "\n",
    "# Get the names of features\n",
    "feature_names = X_train.columns.tolist()  # Assuming X_train is a pandas DataFrame\n",
    "\n",
    "# Get the indices of features sorted by importance\n",
    "indices = np.argsort(importances)[::-1]\n",
    "\n",
    "# Print the feature ranking\n",
    "print(\"Feature ranking:\")\n",
    "for f in range(X_train.shape[1]):\n",
    "    print(\"%d. feature %s (%f)\" % (f + 1, feature_names[indices[f]], importances[indices[f]]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 720x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "top_features = 5\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.title(\"Top {} Feature Importances\".format(top_features))\n",
    "plt.bar(range(top_features), importances[indices][:top_features], color=\"skyblue\", align=\"center\")\n",
    "plt.xticks(range(top_features), [feature_names[i] for i in indices][:top_features], rotation=45, ha=\"right\")\n",
    "plt.xlabel(\"Feature\")\n",
    "plt.ylabel(\"Importance\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.decomposition import PCA\n",
    "\n",
    "# Instantiate PCA\n",
    "pca = PCA(n_components=7)  # Number of components to keep\n",
    "\n",
    "# Fit PCA to the training data\n",
    "X_train_pca = pca.fit_transform(X_train)\n",
    "\n",
    "# Transform the test data using the trained PCA\n",
    "X_test_pca = pca.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>6</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>33568.384265</td>\n",
       "      <td>-46246.689425</td>\n",
       "      <td>-199.340488</td>\n",
       "      <td>-1.040667</td>\n",
       "      <td>5.003675</td>\n",
       "      <td>-0.776035</td>\n",
       "      <td>-0.365003</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-69779.382643</td>\n",
       "      <td>-97785.496899</td>\n",
       "      <td>146.419028</td>\n",
       "      <td>-13.454905</td>\n",
       "      <td>-1.851475</td>\n",
       "      <td>1.326976</td>\n",
       "      <td>-0.838933</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>54270.615713</td>\n",
       "      <td>-94782.666263</td>\n",
       "      <td>52.411299</td>\n",
       "      <td>-7.064306</td>\n",
       "      <td>-3.951181</td>\n",
       "      <td>0.231984</td>\n",
       "      <td>0.497547</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>36609.643648</td>\n",
       "      <td>-52785.712587</td>\n",
       "      <td>191.629081</td>\n",
       "      <td>10.939050</td>\n",
       "      <td>0.044312</td>\n",
       "      <td>0.196739</td>\n",
       "      <td>-0.487468</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>-71597.378123</td>\n",
       "      <td>-72927.106156</td>\n",
       "      <td>46.615897</td>\n",
       "      <td>-4.501787</td>\n",
       "      <td>-3.891650</td>\n",
       "      <td>-0.682220</td>\n",
       "      <td>0.309801</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1995</th>\n",
       "      <td>71233.165026</td>\n",
       "      <td>-79755.061117</td>\n",
       "      <td>-30.329927</td>\n",
       "      <td>-3.166229</td>\n",
       "      <td>0.024294</td>\n",
       "      <td>-0.806440</td>\n",
       "      <td>-0.381333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1996</th>\n",
       "      <td>-75407.707244</td>\n",
       "      <td>-20826.511231</td>\n",
       "      <td>-152.973353</td>\n",
       "      <td>-5.579098</td>\n",
       "      <td>-0.955450</td>\n",
       "      <td>-0.669221</td>\n",
       "      <td>0.187787</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1997</th>\n",
       "      <td>72317.650410</td>\n",
       "      <td>-15954.438719</td>\n",
       "      <td>-61.781553</td>\n",
       "      <td>-15.293460</td>\n",
       "      <td>-2.009316</td>\n",
       "      <td>0.195560</td>\n",
       "      <td>-0.287904</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1998</th>\n",
       "      <td>-80207.427517</td>\n",
       "      <td>44802.489677</td>\n",
       "      <td>35.544509</td>\n",
       "      <td>10.317140</td>\n",
       "      <td>2.032278</td>\n",
       "      <td>-0.675183</td>\n",
       "      <td>0.268443</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1999</th>\n",
       "      <td>56570.613650</td>\n",
       "      <td>75657.231348</td>\n",
       "      <td>3.871147</td>\n",
       "      <td>-6.402622</td>\n",
       "      <td>-0.060294</td>\n",
       "      <td>0.199679</td>\n",
       "      <td>-0.419561</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>2000 rows × 7 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                 0             1           2          3         4         5  \\\n",
       "0     33568.384265 -46246.689425 -199.340488  -1.040667  5.003675 -0.776035   \n",
       "1    -69779.382643 -97785.496899  146.419028 -13.454905 -1.851475  1.326976   \n",
       "2     54270.615713 -94782.666263   52.411299  -7.064306 -3.951181  0.231984   \n",
       "3     36609.643648 -52785.712587  191.629081  10.939050  0.044312  0.196739   \n",
       "4    -71597.378123 -72927.106156   46.615897  -4.501787 -3.891650 -0.682220   \n",
       "...            ...           ...         ...        ...       ...       ...   \n",
       "1995  71233.165026 -79755.061117  -30.329927  -3.166229  0.024294 -0.806440   \n",
       "1996 -75407.707244 -20826.511231 -152.973353  -5.579098 -0.955450 -0.669221   \n",
       "1997  72317.650410 -15954.438719  -61.781553 -15.293460 -2.009316  0.195560   \n",
       "1998 -80207.427517  44802.489677   35.544509  10.317140  2.032278 -0.675183   \n",
       "1999  56570.613650  75657.231348    3.871147  -6.402622 -0.060294  0.199679   \n",
       "\n",
       "             6  \n",
       "0    -0.365003  \n",
       "1    -0.838933  \n",
       "2     0.497547  \n",
       "3    -0.487468  \n",
       "4     0.309801  \n",
       "...        ...  \n",
       "1995 -0.381333  \n",
       "1996  0.187787  \n",
       "1997 -0.287904  \n",
       "1998  0.268443  \n",
       "1999 -0.419561  \n",
       "\n",
       "[2000 rows x 7 columns]"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.DataFrame(X_test_pca)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/sklearn/linear_model/_logistic.py:469: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/sklearn/linear_model/_logistic.py:469: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/sklearn/linear_model/_logistic.py:469: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Selected Features:\n",
      "Index(['country', 'gender', 'age', 'products_number', 'active_member'], dtype='object')\n"
     ]
    }
   ],
   "source": [
    "from sklearn.feature_selection import RFE\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "# Instantiate logistic regression classifier\n",
    "logistic_regression = LogisticRegression()\n",
    "\n",
    "# Instantiate RFE with logistic regression classifier and number of features to select\n",
    "rfe = RFE(estimator=logistic_regression, n_features_to_select=5)  # Specify the number of features to select\n",
    "\n",
    "# Fit RFE to the training data\n",
    "rfe.fit(X_train, y_train)\n",
    "\n",
    "# Get selected features\n",
    "selected_features = X_train.columns[rfe.support_]\n",
    "\n",
    "# Print selected features\n",
    "print(\"Selected Features:\")\n",
    "print(selected_features)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best Parameters: {'max_depth': 5}\n",
      "Best Score (R^2): 0.30128338483643474\n",
      "R^2 Score: 0.3090106534480904\n",
      "Mean Squared Error: 0.10867863167834771\n"
     ]
    }
   ],
   "source": [
    "# param_grid = {\n",
    "#     'fit_intercept': [True, False],\n",
    "#     'normalize': [True, False]\n",
    "# }\n",
    "\n",
    "param_grid = {\n",
    "    'max_depth': [None, 5, 10]\n",
    "}\n",
    "\n",
    "# param_grid = { \n",
    "#     'n_estimators': [100, 200, 300], \n",
    "#     'max_depth': [None, 5, 10] \n",
    "#  }\n",
    "\n",
    "#model = RandomForestRegressor(random_state=0) \n",
    "#model = LinearRegression() \n",
    "model = DecisionTreeRegressor(random_state=0) \n",
    "\n",
    "grid_search = GridSearchCV(model, param_grid, scoring='r2', cv=5)\n",
    "\n",
    "grid_search.fit(X_train, y_train)\n",
    "\n",
    "best_params = grid_search.best_params_ \n",
    "best_score = grid_search.best_score_\n",
    "\n",
    "print(\"Best Parameters:\", best_params) \n",
    "print(\"Best Score (R^2):\", best_score)\n",
    "\n",
    "best_model = grid_search.best_estimator_\n",
    "\n",
    "y_pred = best_model.predict(X_test)\n",
    "\n",
    "print(\"R^2 Score:\", r2_score(y_test, y_pred)) \n",
    "print(\"Mean Squared Error:\", mean_squared_error(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3.10.0 64-bit",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  },
  "orig_nbformat": 4,
  "vscode": {
   "interpreter": {
    "hash": "7e1998ff7f8aa20ada591c520b972326324e5ea05489af9e422744c7c09f6dad"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
