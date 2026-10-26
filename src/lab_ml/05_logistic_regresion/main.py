import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from logistic_regression import LogisticRegressionGD

# How to run: 
# python3 src\lab_ml\logistic_regresion\main.py

def training(X, y, lrgd):
    lrgd.fit(X, y)

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolors='black')

def plot_errors(ada):
    plt.plot(range(1, len(ada.losses_) + 1), ada.losses_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Mean squared error')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Running Logistic Regression")
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    print('Class labels:', np.unique(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
    y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

    lrgd = LogisticRegressionGD(n_iter=1000, eta=0.3, random_state=1)
    training(X_train_01_subset, y_train_01_subset, lrgd)
    plot_decision_regions(X_train_01_subset, y_train_01_subset, classifier=lrgd)
    plt.title('Logistic Regression - Gradient descent')    
    plt.xlabel('Petal standarized length [cm]')
    plt.ylabel('Petal standarized width [cm]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    plot_errors(lrgd)


