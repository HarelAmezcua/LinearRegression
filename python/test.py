import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score, explained_variance_score

# Importing and viewing relevant information about dataset
data_frame = pd.read_csv(r"C:\Users\arath\OneDrive\Documents\GitHub\LinearRegression\datasets\df_regresion_lineal_1.csv")
print(data_frame.head()) 
print(data_frame.info())

# Transforming data_frame to numpy arrays
x_numpy = np.asanyarray(data_frame['x']).reshape(-1, 1)  # Reshape to make it a 2D array
y_numpy = np.asanyarray(data_frame['y'])

# Creating sklearn model
linear_regression_model = linear_model.LinearRegression()
linear_regression_model.fit(x_numpy, y_numpy)

y_predicted = linear_regression_model.predict(x_numpy)

print("Coefficients: Intercept = {}, Slope = {}".format(linear_regression_model.intercept_, linear_regression_model.coef_))

# Optionally, you might want to plot the results
plt.scatter(x_numpy, y_numpy, color='blue')  # Original data
plt.plot(x_numpy, y_predicted, color='red')  # Predicted line
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()


# Evaluate the model
print("Mean Absolute Error: ", mean_absolute_error(y_numpy, y_predicted))
print("Mean Squared Error: ", mean_squared_error(y_numpy, y_predicted))
print("R2 Score: ", r2_score(y_numpy, y_predicted))
