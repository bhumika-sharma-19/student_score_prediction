import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

data = pd.read_csv("student.csv")
print("Columns in CSV : ",data.columns)

data.columns=data.columns.str.strip()

X=data[['Hours']]
y=data['Score']

model = LinearRegression()
model.fit(X,y)

predicted_score = model.predict(X)

mae = mean_squared_error(y,predicted_score)
mse = mean_squared_error(y,predicted_score)
rmse = np.sqrt(mse)

print("mean absolute EEROR (MAE) : ",mae)
print("mean squared ERROR (MSE): ",mse)
print("Root Mean Squared ERROR (RMSE): ",rmse)

new_hour = float(input("enter study hour = "))
new_pred = model.predict([[new_hour]])
print(f"prediction for {new_hour} is score = {new_pred[0]:.2f}")