# Libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Dataset

dataset = pd.read_csv('Your Dataset Path')
print(dataset.head())
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 13].values

# Splitting dataset into trainSet and testSet

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit(X_train)
X_test = sc_X.fit_transform(X_test)
