import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# step 1: Training data
X = np.array([600, 700, 800, 900, 1000]).reshape(-1, 1)
y = np.array([150, 180, 200, 210, 240])
z = y / 1000  # Convert prices to $1000s for easier interpretation


# step 2: Create and train the model
model = LinearRegression()
model.fit(X, y ,)

# step 3: Make predictions
predicted_price = model.predict([[850]])
print(f"Predicted price for an 850 sq ft house: ${predicted_price[0]*1000:.2f}")

# step 4: plotting 
plt.scatter(X, y, color='brown', label='Training Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.scatter(850, predicted_price, color='green', label='Prediction (850 sq ft)', marker='X',
s=100)
plt.xlabel("Size (sq ft)")
plt.ylabel("Price ($1000s)")
plt.title("House Price Prediction using Linear Regression")
plt.legend()
plt.grid(True)
plt.show()
