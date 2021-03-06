import pandas as pd
import numpy as np
from sklearn import preprocessing


FEATURES = ['number_of_times_pregnant', 'plasma_glucose_concentration', 'diastolic_blood_pressure', 'triceps_skin_fold_thickness', '2-hour_serum_insulin', 'body_mass_index',
            'Diabetes pedigree function', 'age']
NAMES = FEATURES + ['class']
CLASS = 'class'


def load_and_prep_data(data_file):
    """
    Method to load and prepare the dataset for use
    """
    X = pd.read_csv(data_file, names=NAMES)

    y = X[CLASS]
    X_no_class = X.drop(CLASS, axis=1)

    x = X_no_class.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    X_no_class_scaled = pd.DataFrame(x_scaled)

    return X, X_no_class_scaled, y

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
    X, X_no_class_scaled, y = load_and_prep_data("../data/diabetes.data")

    # From scratch
    p, w = logistic_regression(X_no_class_scaled, y, 1000000, 0.01)
    print("Accuracy: {}".format(calculate_accuracy(p, y)))

    # Comparison to Sk-Learn’s LogisticRegression
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(fit_intercept=True, C=1e15)
    clf.fit(X_no_class_scaled, y)
    print('Accuracy from sk-learn: {0}'.format(clf.score(X_no_class_scaled, y)))


