# Gradient Boosting 1999
- [Gradient boosting|WIKI](https://en.wikipedia.org/wiki/Gradient_boosting)
- [Friedman, J. H. (February 1999). "Greedy Function Approximation: A Gradient Boosting Machine"](https://jerryfriedman.su.domains/ftp/trebst.pdf)
- [Practical Federated Gradient Boosting Decision Trees(2019)](https://arxiv.org/abs/1911.04206)
## Gradient Boosting實作大車拚
- Python 硬派實作
- 使用scikit-learn
- 使用XGBoost(eXtreme Gradient Boosting)
  - [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/) 
  - [Hands-On Gradient Boosting with XGBoost and scikit-learn(2020)](https://www.packtpub.com/product/hands-on-gradient-boosting-with-xgboost-and-scikit-learn/9781839218354)
- 使用LightGBM (Light Gradient Boosting Machine) 2017
  - 微軟團隊[LightGBM: A Highly Efficient Gradient Boosting Decision Tree ](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html).
  - [LightGBM’s documentation](https://lightgbm.readthedocs.io/en/v3.3.2/)
  - [Predicting Diabetes Using LightGBM and SHAP]()
  - [Machine Learning Models for Data-Driven Prediction of Diabetes by Lifestyle Type()]()
- 使用CatBoost 2017
  - Anna Veronika Dorogush, Andrey Gulin, Gleb Gusev, Nikita Kazeev, Liudmila Ostroumova Prokhorenkova, Aleksandr Vorobev "Fighting biases with dynamic boosting". arXiv:1706.09516, 2017.
  - [CatBoost: unbiased boosting with categorical features](https://arxiv.org/abs/1706.09516)
  - Anna Veronika Dorogush, Vasily Ershov, Andrey Gulin "CatBoost: gradient boosting with categorical features support". Workshop on ML Systems at NIPS 2017.
  - [官方網址](https://yandex.com/dev/catboost/)  [GITHUB](https://github.com/catboost/catboost)
  - [你听过 CatBoost 吗？本文教你如何使用 CatBoost 进行快速梯度提升](https://aijishu.com/a/1060000000143372) 
# 迴歸
- 資料集 [Diabetes(糖尿病) Dataset| Kaggle](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)
- [sklearn.datasets.load_diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)
### Python 硬派實作 see ch6.6
```python
# --- SECTION 1 ---
# Libraries and data loading
from copy import deepcopy
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

import numpy as np

diabetes = load_diabetes()

train_size = 400
train_x, train_y = diabetes.data[:train_size], diabetes.target[:train_size]
test_x, test_y = diabetes.data[train_size:], diabetes.target[train_size:]

np.random.seed(123456)

# --- SECTION 2 ---
# Create the ensemble

# Define the ensemble's size, learning rate and decision tree depth
ensemble_size = 50
learning_rate = 0.1
base_classifier = DecisionTreeRegressor(max_depth=3)

# Create placeholders for the base learners and each step's prediction
base_learners = []

# Note that the initial prediction is the target variable's mean

previous_predictions = np.zeros(len(train_y)) + np.mean(train_y)

# Create the base learners
for _ in range(ensemble_size):
    # Start by calcualting the pseudo-residuals
    errors = train_y - previous_predictions

    # Make a deep copy of the base classifier and train it on the
    # pseudo-residuals
    learner = deepcopy(base_classifier)
    learner.fit(train_x, errors)

    # Predict the residuals on the train set
    predictions = learner.predict(train_x)

    # Multiply the predictions with the learning rate and add the results
    # to the previous prediction
    previous_predictions = previous_predictions + learning_rate*predictions

    # Save the base learner
    base_learners.append(learner)

# --- SECTION 3 ---
# Evaluate the ensemble

# Start with the train set's mean
previous_predictions = np.zeros(len(test_y)) + np.mean(train_y)

# For each base learner predict the pseudo-residuals for the test set and
# add them to the previous prediction, after multiplying with the learning rate
for learner in base_learners:
    predictions = learner.predict(test_x)
    previous_predictions = previous_predictions + learning_rate*predictions

# --- SECTION 4 ---
# Print the metrics
r2 = metrics.r2_score(test_y, previous_predictions)
mse = metrics.mean_squared_error(test_y, previous_predictions)

print('Gradient Boosting:')
print('R-squared: %.2f' % r2)
print('MSE: %.2f' % mse)
```
### 使用scikit-learn
```python
 Libraries and data loading
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics

import numpy as np

diabetes = load_diabetes()

train_size = 400
train_x, train_y = diabetes.data[:train_size], diabetes.target[:train_size]
test_x, test_y = diabetes.data[train_size:], diabetes.target[train_size:]

np.random.seed(123456)

# --- SECTION 2 ---
# Create the ensemble
ensemble_size = 200
learning_rate = 0.1
ensemble = GradientBoostingRegressor(n_estimators=ensemble_size,
                                     learning_rate=learning_rate)

# --- SECTION 3 ---
# Evaluate the ensemble
ensemble.fit(train_x, train_y)
predictions = ensemble.predict(test_x)

# --- SECTION 4 ---
# Print the metrics
r2 = metrics.r2_score(test_y, predictions)
mse = metrics.mean_squared_error(test_y, predictions)

print('Gradient Boosting:')
print('R-squared: %.2f' % r2)
print('MSE: %.2f' % mse)


import matplotlib.pyplot as plt
diffs = [ensemble.train_score_[i] - ensemble.train_score_[i-1] for i in range(1, len(ensemble.train_score_))]

fig, ax1 = plt.subplots()
ax1.plot(ensemble.train_score_, linestyle='--', label='Errors (Left axis)')


ax2 = ax1.twinx()
ax2.plot(diffs, label='Errors Differences (Right axis)')
fig.legend()
```
### 使用XGBoost(eXtreme Gradient Boosting)
```python
# Libraries and data loading
from sklearn.datasets import load_diabetes
from xgboost import XGBRegressor
from sklearn import metrics

import numpy as np

diabetes = load_diabetes()

train_size = 400
train_x, train_y = diabetes.data[:train_size], diabetes.target[:train_size]
test_x, test_y = diabetes.data[train_size:], diabetes.target[train_size:]

np.random.seed(123456)

# --- SECTION 2 ---
# Create the ensemble
ensemble_size = 200
ensemble = XGBRegressor(n_estimators=ensemble_size, n_jobs=4,
                        max_depth=1, learning_rate=0.1,
                        objective ='reg:squarederror')

# --- SECTION 3 ---
# Evaluate the ensemble
ensemble.fit(train_x, train_y)
predictions = ensemble.predict(test_x)

# --- SECTION 4 ---
# Print the metrics
r2 = metrics.r2_score(test_y, predictions)
mse = metrics.mean_squared_error(test_y, predictions)

print('Gradient Boosting:')
print('R-squared: %.2f' % r2)
print('MSE: %.2f' % mse)
```

# 分類
- 資料集: Optical Recognition of Handwritten Digits Data Set
- [sklearn.datasets.load_digits](sklearn.datasets.load_digits)

### 使用scikit-learn `GradientBoostingClassifier`   see CH 6.8
```python
# Libraries and data loading
import numpy as np

from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier  # 使用scikit-learn版本
from sklearn import metrics


digits = load_digits()

train_size = 1500
train_x, train_y = digits.data[:train_size], digits.target[:train_size]
test_x, test_y = digits.data[train_size:], digits.target[train_size:]

np.random.seed(123456)

# --- SECTION 2 ---
# Create the ensemble
ensemble_size = 200
learning_rate = 0.1

ensemble = GradientBoostingClassifier(n_estimators=ensemble_size,
                                      learning_rate=learning_rate)

# --- SECTION 3 ---
# Train the ensemble
ensemble.fit(train_x, train_y)

# --- SECTION 4 ---
# Evaluate the ensemble
ensemble_predictions = ensemble.predict(test_x)

ensemble_acc = metrics.accuracy_score(test_y, ensemble_predictions)

# --- SECTION 5 ---
# Print the accuracy
print('Boosting: %.2f' % ensemble_acc)


import matplotlib.pyplot as plt
diffs = [ensemble.train_score_[i] - ensemble.train_score_[i-1] for i in range(1, len(ensemble.train_score_))]

fig, ax1 = plt.subplots()
ax1.plot(ensemble.train_score_, linestyle='--', label='Errors (Left axis)')


ax2 = ax1.twinx()
ax2.plot(diffs, label='Errors Differences (Right axis)')
fig.legend()
```
### 使用XGBoost(eXtreme Gradient Boosting) see CH6.10
```python
# Libraries and data loading
from sklearn.datasets import load_digits
from xgboost import XGBClassifier
from sklearn import metrics

import numpy as np

digits = load_digits()

train_size = 1500
train_x, train_y = digits.data[:train_size], digits.target[:train_size]
test_x, test_y = digits.data[train_size:], digits.target[train_size:]

np.random.seed(123456)
# --- SECTION 2 ---
# Create the ensemble
ensemble_size = 100
ensemble = XGBClassifier(n_estimators=ensemble_size, n_jobs=4)

# --- SECTION 3 ---
# Train the ensemble
ensemble.fit(train_x, train_y)

# --- SECTION 4 ---
# Evaluate the ensemble
ensemble_predictions = ensemble.predict(test_x)

ensemble_acc = metrics.accuracy_score(test_y, ensemble_predictions)

# --- SECTION 5 ---
# Print the accuracy
print('Boosting: %.2f' % ensemble_acc)

```
# [Gradient Boosting Algorithm: A Complete Guide for Beginners](https://www.analyticsvidhya.com/blog/2021/09/gradient-boosting-algorithm-a-complete-guide-for-beginners/)
- [資料集 Income classification@kaggle](https://www.kaggle.com/lodetomasi1995/income-classification)
  - [Random Forest Classifier + Feature Importance](https://www.kaggle.com/code/prashant111/random-forest-classifier-feature-importance) 
- [GITHUB](https://github.com/AnshulSaini17/Income_evaluation)
