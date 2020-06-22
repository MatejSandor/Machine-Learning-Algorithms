import numpy
from collections import Counter


def euclidean_distance(x1, x2):
    return numpy.sqrt(numpy.sum((x1-x2)**2))


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, x):
        predicted_labels = [self._predict(element) for element in x]
        return numpy.array(predicted_labels)

    def _predict(self, x):
        distances = [euclidean_distance(x, train_element) for train_element in self.x_train]
        k_indices = numpy.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

