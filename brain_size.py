# Implementing linear regression from scratch
# Dataset used - contains head size and brain weight of different people (credit for dataset: https://github.com/FeezyHendrix/LinearRegressionfromscrath)

# import libraries
import inline as inline
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('braindataset.csv')
print(dataset.head())

# initializing our inputs and outputs
X = dataset['Head Size(cm^3)'].values
Y = dataset['Brain Weight(grams)'].values

# mean of our inputs and outputs
X_mean = np.mean(X)
Y_mean = np.mean(Y)

# total number of values
n = len(X)

# using the formula to calculate the b1 and b0
numeretor = 0
denomenator = 0

for i in range(n):
    numeretor += (X[i] - X_mean) * (Y[i] - Y_mean)
    denomenator += (X[i] - X_mean) ** 2

b1 = numeretor / denomenator
b0 = Y_mean - (b1 * X_mean)

# printing the coefficient
print(b1, b0)
# output : 0.26342933948939945 325.57342104944223
# Now we have our bias coefficient(b) and scale factor(m)

# We now have a linear model, lets plot it graphically

# plotting values
x_max = np.max(X) + 100
x_min = np.min(X) - 100

# calculating line values of x and y
x = np.linspace(x_min, x_max, 1000)
y = b0 + b1 * x

# plotting line
plt.plot(x, y, color='red', label='Linear Regression')


# x-axis label
plt.xlabel('Head Size (cm^3)')
# y-axis label
plt.ylabel('Brain Weight (grams)')
plt.legend()
plt.show()

# We need to able to measure how good our model is (accuracy). There are many methods to achieve this but we would
# implement Root mean squared error
rmse = 0
for i in range(n):
    y_pred = b0 + b1 * X[i]
    rmse += (Y[i] - y_pred) ** 2

rmse = np.sqrt(rmse / n)
print("Root Mean Square Error: ", rmse)