import numpy as np
from sklearn.linear_model import LinearRegression

# Defining some data to work with
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))  # regressors/input
y = np.array([5, 20, 14, 32, 22, 38])  # predictors/output

# Create a model and fit it
model = LinearRegression().fit(x, y)

r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)

print('intercept:', model.intercept_)
print('slope:', model.coef_)

# Predict response (once there is a satisfactory model you can use it to predict new/existing data
y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')

# y = c + mx
# y_pred = model.intercept_ + model.coef_ * x