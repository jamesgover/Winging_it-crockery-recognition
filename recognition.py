from py_computer_vision import video_to_img as v2i
from sklearn.neural_network import MLPClassifier as Mlpc
import pandas as pd
import sklearn

RANDOM_STATE = 42
CLF_HIDDEN_LAYERS = (6, 6, 6)  # Simple arbitary choose.
# Simple as more accurate with less data (reduces variance)


def create_and_train(data=None):
    ''' #####_____Crockery_____#####
    # Bowls, Plates, Trays, Cups, bread tray, other? '''
    if data is None:
        data = v2i.load_data(v2i.DATA_FILE)

    # create the classifying network
    clf = Mlpc(alpha=1e-5, hidden_layer_sizes=CLF_HIDDEN_LAYERS, random_state=RANDOM_STATE)
    # random_state set to make deterministic; alpha=1e-5 is low L2 to as over-fitting is avoided by using a simple model
    X = data.drop(columns=["Type_o_Object"])
    y = pd.DataFrame(data, columns=["Type_o_Object"])
    #print(X, y)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.80, random_state=RANDOM_STATE)  # test size large to compensate for small data set
    clf.fit(X_train, y_train)
    train_accuracy = clf.score(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)
    print("the training accuracy was {}.  The test accuracy was {}".
          format(train_accuracy, test_accuracy))


# #####_____Cutlery_____#####
# Forks, Knifes, Spoons, Teaspoons
# use a deep neural network


# #####_____Foils_____#####
# (camera easy QR code)

create_and_train()

