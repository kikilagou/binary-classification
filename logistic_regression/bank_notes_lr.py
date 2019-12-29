import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

FEATURES = ['variance of Wavelet Transformed image', 'skewness of Wavelet Transformed image', 'curtosis of Wavelet Transformed image', 'entropy of image']
NAMES = FEATURES + ['class']
CLASS = 'class'


def load_and_prep_data(data_file):
    """
    Method to load and prepare the dataset for use
    :param data_file: the raw data
    :return: the data split into groups
    """
    X = pd.read_csv(data_file, names=NAMES)

    y = X[CLASS]
    X_no_class = X.drop('class', axis=1)

    return X, X_no_class, y

def format_data(data):
    intercept = np.ones((data.shape[0], 1))
    formatted = np.hstack((intercept, data))

    return formatted


def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))


def log_likelihood(features, y, weights):
    scores = np.dot(features, weights)
    log_likelihood = np.sum(y*scores - np.log(1 + np.exp(scores)))
    return log_likelihood


def logistic_regression(X, y, num_steps, learning_rate):
    format_data(X)
    weights = np.zeros(X.shape[1])
    predictions = None

    for step in range(num_steps):
        scores = np.dot(X, weights)
        predictions = sigmoid(scores)

        output_error = y.T - predictions
        gradient = np.dot(X.T, output_error)
        weights += learning_rate * gradient

    return predictions, weights


def calculate_accuracy(predictions, labels):
    """
    Calculates the accuracy between the predictions and the actual classification
    :param predictions_data: The latest predictions, based on the latest weights, made
    :param data_class: The genuine classification of the patient
    :return: The accuracy percentage
    """
    rounded_predictions = np.round(predictions)

    rounded_predictions_list = rounded_predictions.tolist()
    class_list = labels.values.tolist()

    correct = 0
    for i in range(len(class_list)):
        if class_list[i] == rounded_predictions_list[i]:
            correct += 1

    accuracy = correct / len(class_list)
    return accuracy


if __name__ == '__main__':
    X, X_no_class, y = load_and_prep_data("../data/banknote.data")

    # From scratch
    p, w = logistic_regression(X_no_class, y, 1000000, 0.01)
    print("Accuracy: {}".format(calculate_accuracy(p, y)))

    # Comparison to Sk-Learn’s LogisticRegression
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(fit_intercept=True, C=1e15)
    clf.fit(X_no_class, y)
    print('Accuracy from sk-learn: {0}'.format(clf.score(X_no_class, y)))

# Accuracy: 0.9278425655976676
# Accuracy from sk-learn: 0.9919825072886297