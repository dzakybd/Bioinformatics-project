import pandas as pd
import build_dataset
from __init__ import *
from imblearn.combine import SMOTETomek, SMOTEENN
from classifiers import logistic_model, random_forest_model, naive_bayes_model, knn_model, mlp_model, svm_model
import warnings
warnings.simplefilter(action='ignore')


def over_under_sampling(data):
    column_names = data.columns[:-1]
    smote_tomek = SMOTEENN(ratio='auto')
    features, label = smote_tomek.fit_sample(data[data.columns[:-1]], data[attribute].as_matrix())

    data = pd.DataFrame(features)
    data.columns = column_names
    data[attribute] = label

    logger.info(data)
    return data

def main():
    logger.info("We work on "+attribute+" classification")
    train, test = build_dataset.build()
    train = over_under_sampling(train)

    logger.info('Applying Logistic Regression')
    logistic_model(train, test)

    logger.info('Applying Random Forest')
    random_forest_model(train, test)

    logger.info('Applying Gaussian Naive Bayes')
    naive_bayes_model(train, test)

    logger.info('Applying KNN')
    knn_model(train, test)

    logger.info('Applying SVM')
    svm_model(train, test)

    logger.info('Applying MLP')
    mlp_model(train, test)

if __name__ == '__main__':
    main()
