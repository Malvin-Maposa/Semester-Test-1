import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv('tourism_data_500_points.csv')
df

df.isnull().sum()

df['Location'].unique()

col = ['Location', 'Date']

df_encoder = pd.get_dummies(df, columns = col)
df_encoder

X = df_encoder.drop('Number_of_Visitors', axis = 1)
y = df_encoder['Number_of_Visitors']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

model = RandomForestRegressor()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2_score = metrics.r2_score(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)

print("R2 Score: ", r2_score)
print("MSE: ", mse)
print("MAE: ", mae)

with open('results.txt', 'w') as f:
    f.write(f"R2 score: {r2_score}\n")
    f.write(f"Mean Squared Error: {mse}\n")
    f.write(f"Mean Absolute Error: {mae}\n")


