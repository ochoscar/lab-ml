import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from adaline import Adaline
from adaline_sgd import AdalineSGD

# How to run: 
# python3 src\lab_ml\perceptron\main.py

def read_data():
    s = "E:\\datasets\\ml\\iris\\iris.data"
    df = pd.read_csv(s, header=None, encoding="utf-8")
    df.tail()
    return df

def select_setosa_versicolor(df):
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 0, 1)
    return y

def extract_sepal_petals(df):
    X = df.iloc[0:100, [0, 2]].values
    return X

def plot_data(X):
    plt.scatter(X[:50, 0], X[:50, 1],
                color='red', marker='o', label='Setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1],
                    color='blue', marker='s', label='Versicolor')
    plt.xlabel('Sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()

def compare_trainings_without_standarization(X, y):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ada1 = Adaline(n_iter=15, eta=0.1).fit(X, y)
    ax[0].plot(range(1, len(ada1.losses_) + 1), np.log10(ada1.losses_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Mean squared error)')
    ax[0].set_title('Adaline - Learning rate 0.1')

    ada2 = Adaline(n_iter=15, eta=0.0001).fit(X, y)
    ax[1].plot(range(1, len(ada2.losses_) + 1), ada2.losses_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Mean squared error')
    ax[1].set_title('Adaline - Learning rate 0.0001')
    plt.show()

def standarize(X):
    X_std = np.copy(X)
    X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
    X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    return X_std

def training(X, y, ada):
    ada.fit(X, y)

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
    plt.title('Adaline - Gradient descent')    
    plt.xlabel('Sepal standarized length [cm]')
    plt.ylabel('Petal standarized length [cm]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_errors(ada):
    plt.plot(range(1, len(ada.losses_) + 1), ada.losses_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Mean squared error')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Running perceptron")
    df = read_data()
    y = select_setosa_versicolor(df)
    X = extract_sepal_petals(df)
    #compare_trainings_without_standarization(X, y)

    X_std = standarize(X)
    
    #ada = Adaline(n_iter=20, eta=0.5)
    #training(X_std, y, ada)
    #plot_decision_regions(X_std, y, classifier=ada)
    #plot_errors(ada)

    ada_sgd = AdalineSGD(n_iter=20, eta=0.01, random_state=1)
    training(X_std, y, ada_sgd)
    plot_decision_regions(X_std, y, classifier=ada_sgd)
    plot_errors(ada_sgd)


