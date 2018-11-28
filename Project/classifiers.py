import pandas as pd
import numpy as np
from __init__ import *
from sklearn import metrics, linear_model, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

def calculate_metrics(true_values, predicted_values):
    """
    Calculate precision, recall, f1-score and support based on classifier output

    :param true_values: actual class output
    :param predicted_values: predicted class output
    :return: calculated metrics
    """
    logger.info('Calculated metrics')
    y_true = true_values.astype('int')
    y_pred = predicted_values.astype('int')
    result = "accuracy = " + str(metrics.accuracy_score(y_true=y_true, y_pred=y_pred)) + "\n"\
             + str(metrics.classification_report(y_true=y_true, y_pred=y_pred))
    logger.info(result)



def svm_model(train, test):

    clf = svm.SVC(kernel='linear')
    clf.fit(X=train[train.columns[:-1]], y=train[attribute].astype('int'))

    predicted = clf.predict(X=test[test.columns[:-1]])
    predicted = pd.DataFrame(predicted)

    calculate_metrics(test[attribute], predicted)

def knn_model(train, test):

    knn = KNeighborsClassifier()

    knn.fit(X=train[train.columns[:-1]], y=train[attribute].astype('int'))

    predicted = knn.predict(X=test[test.columns[:-1]])
    predicted = pd.DataFrame(predicted)

    calculate_metrics(test[attribute], predicted)

def mlp_model(train, test):

    mlp = MLPClassifier()

    mlp.fit(X=train[train.columns[:-1]], y=train[attribute].astype('int'))

    predicted = mlp.predict(X=test[test.columns[:-1]])
    predicted = pd.DataFrame(predicted)

    calculate_metrics(test[attribute], predicted)


def logistic_model(train, test):

    log_model = linear_model.LogisticRegression()

    print(train[train.columns[:-1]].isnull())
    train.columns[:-1] = pd.to_numeric(train.columns[:-1], errors='coerce')
    train = train.dropna(subset=[train.columns[:-1]])
    df['x'] = df['x'].astype(int)

    log_model.fit(X=coba, y=train[attribute].astype(int))

    predicted = log_model.predict(X=test[test.columns[:-1]])
    predicted = pd.DataFrame(predicted)

    calculate_metrics(test[attribute], predicted)

def random_forest_model(train, test):

    forest_model = RandomForestClassifier()

    forest_model.fit(X=train[train.columns[:-1]].as_matrix(),
                     y=train[attribute].astype('int').as_matrix())

    predicted = forest_model.predict(X=test[test.columns[:-1]])

    calculate_metrics(test[attribute], predicted)

def naive_bayes_model(train, test):

    gnb_model = GaussianNB()

    gnb_model.fit(X=train[train.columns[:-1]].as_matrix(),
                  y=train[attribute].astype('int').as_matrix())

    predicted = gnb_model.predict(X=test[test.columns[:-1]])

    calculate_metrics(test[attribute], predicted)
