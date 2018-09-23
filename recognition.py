# from py_computer_vision import video_to_img as v2i
import video_to_img as v2i
from sklearn.neural_network import MLPClassifier as Mlpc
import pandas as pd
import sklearn
import numpy as np


RANDOM_STATE = 42
CLF_HIDDEN_LAYERS = (6, 6, 6)  # Simple arbitary choose.
# Simple as more accurate with less data (reduces variance)


# import the data set, by default from a csv
# :param data, used to specify a data frame calling funtion would prefer to use
def create_data(data=None):
    # #####_____Crockery_____#####
    # Bowls, Plates, Trays, Cups, bread tray, other?
    if data is None:
        data = v2i.load_data(v2i.DATA_FILE)
    return data


# trains a classifying neural network based on data
# uses subset of data to create a test set
# SAI APP CHALLENGE NOTES -- this function requires data quantities larger than we currently have access to so is
#       currently not a viable option for classifying.
# :param data, the data set to be used by the function
def train_nueral_network(data):
    X = data.drop(columns=["Type_o_Object"])
    y = pd.DataFrame(data, columns=["Type_o_Object"])
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE)  # test small large to compensate for small data set
    y_train, y_test = y_train.astype(dtype=np.float64), y_test.astype(dtype=np.float64)

    # create the classifying network
    clf = Mlpc(alpha=1e-5, hidden_layer_sizes=CLF_HIDDEN_LAYERS, random_state=RANDOM_STATE)
    # random_state set to make deterministic; alpha=1e-5 is low L2 to as over-fitting is avoided by using a simple model

    clf.fit(X_train, np.ravel(y_train))
    train_accuracy = clf.score(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)
    print("for the neural network; training accuracy was {}.  The test accuracy was {}".
            format(train_accuracy, test_accuracy))


# trains a classifying neural network based on data
# uses subset of data to create a test set
# SAI APP CHALLENGE NOTES --
# :param data, the data set to be used by the function
def train_knn(data):
    X = data.drop(columns=["Type_o_Object"])
    y = pd.DataFrame(data, columns=["Type_o_Object"])
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE)  # test small large to compensate for small data set
    y_train, y_test = y_train.astype(dtype=np.float64), y_test.astype(dtype=np.float64)

    # create the classifying network
    knnc = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3, p=2)
    # random_state set to make deterministic; alpha=1e-5 is low L2 to as over-fitting is avoided by using a simple model

    knnc.fit(X_train, y_train)
    train_accuracy = knnc.score(X_train, y_train)
    test_accuracy = knnc.score(X_test, y_test)
    print("for the k nearest neighbours classifier; training accuracy was {}.  The test accuracy was {}".
            format(train_accuracy, test_accuracy))


# trains a classifying neural network based on data
# uses subset of data to create a test set
# SAI APP CHALLENGE NOTES --
# :param data, the data set to be used by the function


# #####_____Cutlery_____#####
# Forks, Knifes, Spoons, Teaspoons
# use a deep neural network


# #####_____Foils_____#####
# (camera easy QR code)

def main():
    data = create_data()
    print(data.columns.values)
    #train_knn(data)
    train_nueral_network(data)
    #train_lr(data)

main()

