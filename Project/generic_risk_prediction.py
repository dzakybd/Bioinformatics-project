import build_dataset
from classifiers import logistic_model, random_forest_model, naive_bayes_model, knn_model, mlp_model, svm_model
from __init__ import *

import warnings
warnings.simplefilter(action='ignore')

def main():
    """
    Execute generic classification methods on DNA methylation data

    :return: metrics of each classifier
    """
    logger.info("We work on "+attribute+" classification")
    train, test = build_dataset.build()

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
