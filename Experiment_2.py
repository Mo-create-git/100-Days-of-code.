import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

print("Dataset Preview:")
print(df.head())

# Select Features (X) and Target (y)
X = df[['lstat', 'rm', 'age']]
y = df['medv']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train Model
model = LinearRegression()
model.fit(X_train, y_train)
# Model Parameters
print("\nIntercept:", model.intercept_)
print("Coefficients:", model.coef_)
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print("\nCoefficient Table:\n", coef_df)
# Predictions
y_pred = model.predict(X_test)
# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("\nModel Evaluation:")

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)
# Visualization
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_pred, color="blue")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Prices")
plt.show()