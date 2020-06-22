import numpy


class LinearRegression:

    def __init__(self, lr=0.001, n_iterations=1000):
        self.lr = lr
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.weights = numpy.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_predicted = numpy.dot(x, self.weights) + self.bias

            dw = (1 / n_samples) * numpy.dot(x.T, (y_predicted - y))
            db = (1 / n_samples) * numpy.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, x):
        pass
