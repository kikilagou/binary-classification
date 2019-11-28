import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# S = suite, C = card
headings = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'hand']

# import data
train = pd.read_csv("../../../Documents/Data/poker-hand-training-true.txt", names=headings)
test = pd.read_csv("../../../Documents/Data/poker-hand-testing.txt", names=headings)

print(train.head())
print(train.shape)
print(test.shape)

X = train.drop('hand', axis=1)
print(X.head())

Y = train['hand']
print(Y.head())
