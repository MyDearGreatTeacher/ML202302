# Regression Analysis迴歸分析

# 推薦書籍
- [Python Machine Learning: Machine Learning and Deep Learning with Python, scikit-learn, and TensorFlow, 3/e](https://www.packtpub.com/product/python-machine-learning-third-edition/9781789955750)
  - [github](https://github.com/rasbt/python-machine-learning-book-3rd-edition) 
  - [Python 機器學習 (上), 3/e](https://www.tenlong.com.tw/products/9789864345182?list_name=srh)
    - 推薦章節:第10章：以迴歸分析預測連續目標變數

# 大綱
- DATASET:  房價預測問題
- Exploratory data analysis (EDA) 探索房屋數據集
- 使用RANdom SAmple Consensus (RANSAC)找出強固的迴歸模型
- 評估線性迴歸模型的效能
  - from sklearn.metrics import r2_score 
- 使用正規化方法(regularized methods)做迴歸
  - from sklearn.linear_model import Ridge
  - from sklearn.linear_model import Lasso 
- 將線性迴歸模型轉成曲線－多項式迴歸polynomial regression
  - from sklearn.preprocessing import PolynomialFeatures
- 更多Nonlinear regression非線性迴歸
  - random forest nonlinear  regression使用隨機森林處理非線性關係
    - from sklearn.ensemble import RandomForestRegressor 
  - Decision tree regression
    - from sklearn.tree import DecisionTreeRegressor
  - support vector machine (SVM) nonlinear regression
    - [Support Vector Machines for Classification and Regression, S. R. Gunn ..(1998)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.579.6867&rep=rep1&type=pdf)
    - [sklearn.svm.SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR)

# 教學主軸 
- 原理解說
- 實作
  - 使用PYTHON
  - 使用scikit-learn
- 應用

## 進階研讀statsmodels
- [statsmodels](https://www.statsmodels.org/stable/index.html)
  - statsmodels is a Python module that provides classes and functions for the estimation of many different statistical models, as well as for conducting statistical tests, and statistical data exploration. 
- [statsmodels範例學習](https://www.statsmodels.org/stable/examples/index.html)
- statsmodels@colab
```python
import statsmodels
statsmodels.__version__
# 0.10.2
```
- [測試Ordinary Least Squares及各種變形](https://www.statsmodels.org/stable/examples/notebooks/generated/ols.html)

## DATASET:  房價預測問題
- Boston House Prices
  - Harrison, D. and Rubinfeld, D.L. `Hedonic prices and the demand for clean air', J. Environ. Economics & Management, vol.5, 81-102, 1978. 
  - http://lib.stat.cmu.edu/datasets/boston
  - https://www.kaggle.com/datasets/vikrishnan/boston-house-prices
  - [sklearn.datasets.load_boston](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html)
- California housing dataset
- [UC Irvine Machine Learning Repository(有612個資料集)](https://archive-beta.ics.uci.edu/)
```
CRIM: Per capita crime rate by town
ZN: Proportion of residential land zoned for lots over 25,000 sq. ft.
INDUS: Proportion of non-retail business acres per town
CHAS: Charles River dummy variable (= 1 if tract bounds river and 0 otherwise)
NOX: Nitric oxide concentration (parts per 10 million)
RM: Average number of rooms per dwelling
AGE: Proportion of owner-occupied units built prior to 1940
DIS: Weighted distances to five Boston employment centers
RAD: Index of accessibility to radial highways
TAX: Full-value property tax rate per $10,000
PTRATIO: Pupil-teacher ratio by town
B: 1000(Bk – 0.63)2, where Bk is the proportion of [people of African American descent] by town
LSTAT: Percentage of lower status of the population
MEDV: Median value of owner-occupied homes in $1000s
```

## 分析
- [What impacts Boston Housing Prices](https://medium.com/li-ting-liao-tiffany/python-%E5%BF%AB%E9%80%9F%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-boston-housing%E6%B3%A2%E5%A3%AB%E9%A0%93%E6%88%BF%E5%83%B9-9c535fb7ceb7)
- [Boston House Price Dataset: Evaluating the performance RandomForest and OLS regression models(2021)](https://www.youtube.com/watch?v=LvsFtFkIoX4)
  - https://sites.google.com/view/vinegarhill-datalabs/introduction-to-machine-learning/random-forest-and-ols
  - https://www.youtube.com/watch?v=CfuZl-2vrsg
- [Boston House Prices Data Set Analysis and Visualization | Machine Learning Regression Model(2020)](https://www.youtube.com/watch?v=CGQTT-swK7U) 
  - [程式碼](https://github.com/Prianca25/Machine-Learning/blob/master/Boston%20Model%20Deployment.ipynb)  
- [boston-housing-dataset分析@github](https://github.com/topics/boston-housing-dataset)
- [boston-housing-dataset分析@paperswithcode](https://paperswithcode.com/dataset/the-boston-housing-dataset)
