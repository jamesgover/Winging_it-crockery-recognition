# from py_computer_vision import video_to_img as v2i
import video_to_img as v2i
from sklearn.neural_network import MLPClassifier as Mlpc
import pandas as pd
import sklearn
import sklearn.neighbors as skn
import sklearn.linear_model as sk_lin
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


def pre_process_data(data):
    X = data.drop(columns=["Type_o_Object"])
    y = pd.DataFrame(data, columns=["Type_o_Object"])
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE)  # test small large to compensate for small data set
    y_train, y_test = y_train.astype(dtype=np.float64), y_test.astype(dtype=np.float64)
    return X_train, X_test, np.ravel(y_train), y_test;


# trains a classifying neural network based on data
# uses subset of data to create a test set
# SAI APP CHALLENGE NOTES -- this function requires data quantities larger than we currently have access to so is
#       currently not a viable option for classifying.
# :param data, the data set to be used by the function
def train_neural_network(data):
    X_train, X_test, y_train, y_test = pre_process_data(data)

    # create the classifying network
    clf = Mlpc(alpha=1e-5, hidden_layer_sizes=CLF_HIDDEN_LAYERS, random_state=RANDOM_STATE)
    # random_state set to make deterministic; alpha=1e-5 is low L2 to as over-fitting is avoided by using a simple model

    clf.fit(X_train, y_train)
    train_accuracy = clf.score(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)
    print("for the neural network; training accuracy was {}.  The test accuracy was {}".
            format(train_accuracy, test_accuracy))


# trains a classifying K nearest neighbours model based on "data"
# uses subset of data to create a test set
# SAI APP CHALLENGE NOTES -- This has no optimisation as optimising without significant amounts of data
#    and without working machine data will be mostly inaccurate anyway.
#    With large scale real data optimisation can be achieved by:
#       - varying the choice of k
#       - applying some form of normalisation to data
#       - refine and be more selective in which measures we use.
# currently with our data set this only scored 0.5714285714285714% test accuracy.
#       This is mostly correlated to the nature of our data set being no consistent
#       and not having enough data to heavily optimise.
# :param data, the data set to be used by the function
def train_knn(data):
    X_train, X_test, y_train, y_test = pre_process_data(data)

    # create the classifying network
    knnc = skn.KNeighborsClassifier(n_neighbors=3, p=2)
    # random_state set to make deterministic; alpha=1e-5 is low L2 to as over-fitting is avoided by using a simple model

    knnc.fit(X_train, y_train)
    train_accuracy = knnc.score(X_train, y_train)
    test_accuracy = knnc.score(X_test, y_test)
    print("for the k nearest neighbours classifier; training accuracy was {}.  The test accuracy was {}".
            format(train_accuracy, test_accuracy))


# trains a logistic regression classifier based on data
# uses subset of data to create a test set
# SAI APP CHALLENGE NOTES -- Not guaranteed accurate for data that is not linearly separable.
#       Once larger sets of more accurate data are obtained this can be verified and the accuracy better assessed
#       with the small data set we are using is appears to be barely satisfactory but feasible enough to justify
#       further investigation with full scale data.
# with out test dataset we had this at 0.8571428571428571% test accuracy which is respectable and can be improved
# through optimisation of processing of the images
# :param data, the data set to be used by the function
def train_lr(data):
    X_train, X_test, y_train, y_test = pre_process_data(data)

    lr_model = sk_lin.LogisticRegression()
    lr_model.fit(X_train, y_train)
    train_accuracy = lr_model.score(X_train, y_train)
    test_accuracy = lr_model.score(X_test, y_test)
    print("for the logistic regression model; training accuracy was {}.  The test accuracy was {}".
          format(train_accuracy, test_accuracy))


# trains a classifying neural network based on data
# uses subset of data to create a test set
# SAI APP CHALLENGE NOTES --
# :param data, the data set to be used by the function
def main():
    # if a dataset for training and testing does not exist create one by querying the user to identify objects
    try:
        loaded_data = pd.read_csv(filepath_or_buffer=v2i.DATA_FILE)
    except:
        v2i.process_sample_photos()  # this is to create the data set if it does not exist

    # load the dataset (either the one just created or the pre-existing dataset);
    data = create_data()
    X_train, X_test, y_train, y_test = pre_process_data(data)
    print("the labels of the X vector are: " + str(X_train.columns.values[:4]))
    print("used to predict the object type")
    print("\nTESTING LEARNING METHODS")
    train_knn(data)  # k nearest neighbours model
    train_neural_network(data)  # neural network classifier model
    train_lr(data)  # logistic regression


main()

