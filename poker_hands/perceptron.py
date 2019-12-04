import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# S = suite, C = card
headings = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'hand']

# import data
train = pd.read_csv("../../../Documents/Data/poker-hand-training-true.txt", names=headings)
test = pd.read_csv("../../../Documents/Data/poker-hand-testing.txt", names=headings)

feature_set = train.drop('hand', axis=1)
labels = train['hand'].values
labels = labels.reshape(len(labels), 1)

# feature_set.describe()
# scaler = StandardScaler()
# scaler = MinMaxScaler()
# feature_set = scaler.fit_transform(feature_set)

print(feature_set.mean(axis=0))
print(feature_set.std(axis=0))
# feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
# labels = np.array([[1,0,0,1,1]])
# labels = labels.reshape(5,1)

print(feature_set.shape)
print(labels.shape)


# defining some hyper parameters for our neural network
# np.random.seed(420)
weights = np.random.rand(10, 1)
bias = np.random.rand(1)
lr = 0.00001


# activation function (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# derivative of sigmoid function
def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

# #
for epoch in range(1000):
    inputs = feature_set

    # step 1: feed forward
    XW = np.dot(feature_set, weights) + bias

    # step 2: feed forward
    z = sigmoid(XW)

    # step 1: back propagation
    error = z - labels
    # print(error.sum())

    # step 2: back propagation
    dcost_dpred = error
    dpred_dz = sigmoid_der(z)

    z_delta = dcost_dpred * dpred_dz

    inputs = feature_set.T
    weights -= lr * np.dot(inputs, z_delta)

    for num in z_delta:
        bias -= lr * num



single_point = np.array([1,10,1,11,1,13,1,12,1,1])
result = sigmoid(np.dot(single_point, weights) + bias)
print(result)