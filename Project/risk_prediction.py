import build_dataset
import numpy as np
from classifiers import logistic_model, knn_model, mlp_model, svm_model
from __init__ import *
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore')


def visualization(name, classifier_result):
    plt.cla()
    plt.clf()
    y_pos = np.arange(len(classifier_result))
    plt.bar(y_pos, classifier_result.values(), align='center')
    plt.xticks(y_pos, classifier_result.keys())
    plt.ylabel('Accuracy')
    plt.ylim(0.0, 1.0)
    plt.title(name)
    rects = plt.axes().patches
    labels = ['%.2f' % elem for elem in classifier_result.values()]
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        plt.axes().text(rect.get_x() + rect.get_width() / 2, height + 5, label, ha='center', va='bottom')
    figure_path = os.path.join(result_location, name+".png")
    plt.savefig(figure_path, format="png")


def main():
    """
    Execute generic classification methods on DNA methylation data
    """
    logger.info("We work on "+attribute+" classification")
    train, test = build_dataset.build()
    classifier_result = {}

    logger.info('Applying Logistic Regression')
    name, acc = logistic_model(train, test, "l1")
    classifier_result[name] = acc
    name, acc = logistic_model(train, test, "l2")
    classifier_result[name] = acc
    visualization("LR", classifier_result)

    classifier_result.clear()
    logger.info('Applying KNN')
    name, acc = knn_model(train, test, 2)
    classifier_result[name] = acc
    name, acc = knn_model(train, test, 3)
    classifier_result[name] = acc
    name, acc = knn_model(train, test, 4)
    classifier_result[name] = acc
    name, acc = knn_model(train, test, 5)
    classifier_result[name] = acc
    visualization("KNN", classifier_result)

    classifier_result.clear()
    logger.info('Applying SVM')
    name, acc = svm_model(train, test, "linear")
    classifier_result[name] = acc
    name, acc = svm_model(train, test, "rbf")
    classifier_result[name] = acc
    name, acc = svm_model(train, test, "poly", 2)
    classifier_result[name] = acc
    name, acc = svm_model(train, test, "poly", 3)
    classifier_result[name] = acc
    name, acc = svm_model(train, test, "poly", 4)
    classifier_result[name] = acc
    visualization("SVM", classifier_result)

    classifier_result.clear()
    logger.info('Applying MLP')
    name, acc = mlp_model(train, test, "logistic", (100,))
    classifier_result[name] = acc
    name, acc = mlp_model(train, test, "logistic", (100, 100))
    classifier_result[name] = acc
    name, acc = mlp_model(train, test, "logistic", (100, 100, 100))
    classifier_result[name] = acc
    name, acc = mlp_model(train, test, "relu", (100,))
    classifier_result[name] = acc
    name, acc = mlp_model(train, test, "relu", (100, 100))
    classifier_result[name] = acc
    name, acc = mlp_model(train, test, "relu", (100, 100, 100))
    classifier_result[name] = acc
    visualization("MLP", classifier_result)


if __name__ == '__main__':
    main()
