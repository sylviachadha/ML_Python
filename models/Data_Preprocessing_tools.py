# Tool 1 - Importing libraries
#####################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Tool 2 Importing dataset
#####################################################

os.chdir('/Users/sylviachadha/Desktop/All_Datasets/ML')
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(X)
print(y)

# Tool 3 - Handle missing data
#####################################################

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

print(X)

# Tool 4 - Encoding categorical variables
#####################################################

# Encoding Independent variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X)

# Encoding dependent variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(y)

# Tool 5 - Split into Train and Test
#####################################################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Tool 6 - Feature Scaling
#####################################################

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print(X_train)

print(X_test)






