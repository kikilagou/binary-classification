import pandas as pd
from copy import deepcopy
from sklearn.model_selection import train_test_split

NAMES = ['id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'type']
FEATURES = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
TRAIN_TEST_RATIO = 0.8
k = 3
CLASS = NAMES[10]


def load_and_prep_data(csv):
    """
    Method to load and prepare the dataset for use
    :param csv: the raw data
    :return: the finalised dataframe
    """
    X = pd.read_csv(csv, names=NAMES)
    X = X.drop('id', axis=1)

    y = X[CLASS]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-TRAIN_TEST_RATIO, random_state=4)

    return X, X_train, X_test, y_train, y_test


def calc_euclidean_distance(X_train, unseen_point):
    """
    Calculate the distance between the query instance and all the training examples
    Get the neighbours
    :return:
    """
    arr = []
    X_train_dist = deepcopy(X_train)
    for index, row in X_train.iterrows():
        sum = 0
        for ind, r in unseen_point.iterrows():
            for i in range(len(FEATURES)):
                sum += pow(row[i] - r[i], 2)
        arr.append(sum)
    X_train_dist['dist'] = arr

    X_train_dist.sort_values(by='dist', ascending=True, inplace=True)
    neighbours = X_train_dist.head(n=k)

    return neighbours

def predict_class(neighbours):
    mode = neighbours.loc[:, CLASS].mode()

    return mode


if __name__ == '__main__':
    _, X_train, X_test, _, _ = load_and_prep_data("../data/glass.data")

    correct = misclassified = 0
    for i in range(len(X_test)):
        row = X_test.iloc[[i]]
        neighbours = calc_euclidean_distance(X_train, row)
        classification_prediction = predict_class(neighbours).iloc[0]
        actual_classification = row[CLASS].iloc[0]
        if actual_classification == classification_prediction:
            correct += 1
        else:
            misclassified += 1

    accuracy = correct / len(X_test.index) * 100
    print("Accuracy: {}%".format(accuracy))
    print('Misclassified samples: {}'.format(misclassified))
