import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# S = suite, C = card
headings = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'hand']

# import data
train = pd.read_csv("../../../Documents/Data/poker-hand-training-true.txt", names=headings)
test = pd.read_csv("../../../Documents/Data/poker-hand-testing.txt", names=headings)

print(train.head())
print(train.shape)
print(test.shape)

X = train.drop('hand', axis=1)
# print(X.head())

y = train['hand'].values
# print(y.head())

s1 = train['S1'].values
c1 = train['C1'].values
s2 = train['S2'].values
c2 = train['C2'].values
s3 = train['S3'].values
c3 = train['C3'].values
s4 = train['S4'].values
c4 = train['C4'].values
s5 = train['S5'].values
c5 = train['C5'].values

m = len(s1)
x0 = np.ones(m)
X = np.array([x0, s1, c1, s2, c2, s3, c3, s4, c4, s5, c5]).T
B = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
Y = np.array(y)
alpha = 0.0001

def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
    return J

inital_cost = cost_function(X, Y, B)
print(inital_cost)


def gradient_descent(X, Y, B, alpha, iterations):
    cost_history = [0] * iterations
    m = len(Y)

    for iteration in range(iterations):
        # Hypothesis Values
        h = X.dot(B)
        # Difference b/w Hypothesis and Actual Y
        loss = h - Y
        # Gradient Calculation
        gradient = X.T.dot(loss) / m
        # Changing Values of B using Gradient
        B = B - alpha * gradient
        # New Cost Value
        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost

    return B, cost_history

# 100000 Iterations
newB, cost_history = gradient_descent(X, Y, B, alpha, 100000)

# New Values of B
print(newB)

# Final Cost of new B
print(cost_history[-1])

# Model Evaluation - RMSE
def rmse(Y, Y_pred):
    rmse = np.sqrt(sum((Y - Y_pred) ** 2) / len(Y))
    return rmse

Y_pred = X.dot(newB)
print("root mean square error: ", rmse(Y, Y_pred))

# Confusion matrix using scykit learn
