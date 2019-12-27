import numpy as np
import pandas as pd

FEATURES = ['id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'type']
test_train_ratio = 0.8

def load_and_prep_data(csv):
    """
    Method to load and prepare the dataset for use
    :param csv: the raw data
    :return: the finalised dataframe
    """
    data = pd.read_csv(csv, names=FEATURES)

    # drop the id column
    # split the data set into test and training sets
    # remove the class column into a separate dataframe from the features
    # randomly sample data to split into training and test sets

    # format data so that it is all to the same decimal point
    # drop partially filled columns
    # get rid of extreme values or wrong values

    # use histogram to review the distribution and reduce skewness of the data

    print(data.head())
    return data

def data_exploration(data):
    """
    This method performs basic data exploration on the dataset
    :param data: the dataset
    :return:
    """


data = load_and_prep_data("glass.data")