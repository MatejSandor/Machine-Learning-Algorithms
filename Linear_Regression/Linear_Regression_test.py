import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from Linear_Regression.LinearRegression import LinearRegression


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


x, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

regression = LinearRegression(lr=0.01)
regression.fit(x_train, y_train)
predicted = regression.predict(x_test)

mse = mean_squared_error(y_test, predicted)
print(mse)

y_pred_line = regression.predict(x)
color_map = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(x_train, y_train, color=color_map(0.9), s=10)
m2 = plt.scatter(x_test, y_test, color=color_map(0.5), s=10)
plt.plot(x, y_pred_line, color='black', linewidth=2, label="Prediction")
plt.show()
