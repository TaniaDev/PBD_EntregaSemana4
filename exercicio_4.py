import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('weatherAUS.csv')

independent = dataset.iloc[:10, [2, 3, 7, 8, 11]].dropna()
dependent = dataset.iloc[:10, -1]

ind = independent.values
dep = dependent.values

transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])], remainder='passthrough')
ind = np.array(transformer.fit_transform(ind))

print(ind)

ind_train, ind_test, dep_train, dep_test = train_test_split(ind, dep, test_size=0.2, random_state=0)

linearRegression = LinearRegression()

linearRegression.fit(ind_train, dep_train)

dep_pred = linearRegression.predict(ind_test)

np.set_printoptions(precision=2)
dep_pred_col = dep_pred.reshape(len(dep_pred), 1)
dep_test_col = dep_test.reshape(len(dep_pred), 1)
print(np.concatenate((dep_pred_col, dep_test_col), axis=1))

print (f'{linearRegression.intercept_:.2f}')
for c in linearRegression.coef_:
  print (f'{c:.2f} ')