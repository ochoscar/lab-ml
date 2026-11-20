import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt

# How to run:
# mac: python3 src/lab_ml/08_L1_regularization_feature_selection/l1.py

# Read data set
df_wine = pd.read_csv('/Users/oscochoa/dataset/wine/wine.data', header=None)
df_wine.columns = ['Class label',
                    'Alcohol',
                    'Malic acid',
                    'Ash',
                    'Alcalinity of ash',
                    'Magnesium',
                    'Total phenols',
                    'Flavanoids',
                    'Nonflavanoid phenols',
                    'Proanthocyanins',
                    'Color intensity',
                    'Hue',
                    'OD280/OD315 of diluted wines',
                    'Proline']

print('Class labels:', df_wine['Class label'].unique())
print(df_wine.head())

# Separate training and test sets
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# Standardize
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

# Classification
lr = OneVsRestClassifier(
    LogisticRegression(
        C=1.0,
        solver='liblinear',
        l1_ratio=1.0
    )
)
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))

# Weight and bias per class
intercepts = np.array([
    estimator.intercept_[0]
    for estimator in lr.estimators_
])
coefficients = np.array([
    estimator.coef_[0]
    for estimator in lr.estimators_
])
print(intercepts)
print(coefficients)

# Plot c vs features

fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan',
          'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue',
          'gray', 'indigo', 'orange']
weights, params = [], []
for c in np.arange(-4., 6.):
    lr = OneVsRestClassifier(
        LogisticRegression(
            C=10.**c,
            solver='liblinear',
            l1_ratio=1.0
        )
    )
    lr.fit(X_train_std, y_train)
    coefficients = np.array([
        estimator.coef_[0]
        for estimator in lr.estimators_
    ])
    weights.append(coefficients[1])
    params.append(10**c)
weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column], label=df_wine.columns[column + 1], color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim(10**(-5), 10**5)
plt.ylabel('Weigth coefficient')
plt.xlabel('C Inverse regularization strength')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center',
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)
plt.show()