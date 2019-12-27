import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

FEATURES = ['id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'type']
TRAIN_TEST_RATIO = 0.8

def load_and_prep_data(csv):
    """
    Method to load and prepare the dataset for use
    :param csv: the raw data
    :return: the finalised dataframe
    """
    data = pd.read_csv(csv, names=FEATURES)

    x = data.drop('type', axis=1)
    x = x.drop('id', axis=1)
    y = data['type']

    print(x.head())
    print(y.head())

    # split the data set into test and training sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=1-TRAIN_TEST_RATIO)

    # remove the class column into a separate dataframe from the features
    # randomly sample data to split into training and test sets

    # format data so that it is all to the same decimal point
    # drop partially filled columns
    # get rid of extreme values or wrong values

    # use histogram to review the distribution and reduce skewness of the data

    return data

def data_exploration(data):
    """
    This method performs basic data exploration on the dataset.
    Goals are to view:
    - Data distributions
    - Identify skewed predictors
    - Identify outliers
    :param data: the dataset
    :return:
    """
    num_bins = 11
    n, bins, patches = plt.hist(data, num_bins, facecolor='blue', alpha=0.5)
    plt.show()

data = load_and_prep_data("glass.data")
# data_exploration(data)