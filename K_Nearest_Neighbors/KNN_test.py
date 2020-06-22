from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy
from matplotlib.colors import ListedColormap

from K_Nearest_Neighbors.KNN import KNN

color_map = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


def accuracy(y_true, y_pred):
    accuracy_measurement = numpy.sum(y_true == y_pred) / len(y_true)
    return accuracy_measurement


iris = datasets.load_iris()
x, y = iris.data, iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

plt.figure()
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=color_map, edgecolor='k', s=20)
plt.show()

k = 3
clf = KNN(k=k)
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)

print("custom KNN classification accuracy", accuracy(y_test, predictions))
