# 機器學習演算法 
- [開發套件 scikit-learn](https://scikit-learn.org/stable/)

## 監督式學習Supervised learning  
- [scikit-learn支援的演算法](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)

## 回歸Regression  
- Linear Regression線性回歸
  - Ordinary Least Squares
  - 👍[有數學公式:Simple Linear Regression in Python (From Scratch)](https://towardsdatascience.com/simple-linear-regression-in-python-numpy-only-130a988c0212)
  - [sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
        - Attributes
        - fit(X, y[, sample_weight]):Fit linear model.
        - predict(X): Predict using the linear model.
  - [不用套件硬功夫](https://github.com/m0-k1/Linear-Regression_model)
  - [使用各種套件](https://github.com/tatwan/Linear-Regression-Implementation-in-Python)
    - Manual with Gradient Descent
    - Using Scipy
    - Using Scikit-Learn
    - Using Statsmodel
  - YOUTUBE 教學影片
    - [Regression Analysis | Full Course](https://www.youtube.com/watch?v=0m-rs2M7K-Y) 
    - 範例研讀
      - [07_Simple_and_Multiple_Regression](https://github.com/sandipanpaul21/Machine-Learning-in-Python-Code/blob/master/07_Simple_and_Multiple_Regression.ipynb) 
        - [範例筆記](https://github.com/sandipanpaul21/ML-Notes-Daywise)
      - [karthickai/Linear-Regression](https://github.com/karthickai/Linear-Regression) 
    - 專題
      - 房價預測問題[Boston House Prices](https://www.kaggle.com/datasets/vikrishnan/boston-house-prices) 
      - 薪水預測[KAGGLE: Salary data - Simple linear regression](https://www.kaggle.com/datasets/karthickveerakumar/salary-data-simple-linear-regression) 
- Multiple Linear Regression線性回歸
- Polynomial Regression 使用numpy方法 [Polynomial Regression](https://www.w3schools.com/python/python_ml_polynomial_regression.asp)
- Ridge regression 
- Lasso regression
- Nearest Neighbors Regression 

## 分類Classification ==> 二元分類(有病|沒病,正常|異常) VS 多元分類(不同等級A|B|C|D|E|...的水果)
- Nearest Neighbors
  - 👍[A Simple Introduction to K-Nearest Neighbors Algorithm](https://towardsdatascience.com/a-simple-introduction-to-k-nearest-neighbors-algorithm-b3519ed98e)
  - [K-Nearest Neighbor(KNN) Algorithm for Machine Learning](https://www.javatpoint.com/k-nearest-neighbor-algorithm-for-machine-learning)
  - [sklearn.neighbors.KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
    - from sklearn.neighbors import KNeighborsClassifier  
  - 範例 [k-NN on Iris Dataset](https://towardsdatascience.com/k-nn-on-iris-dataset-3b827f2591e)
  - Kaggle範例 [Iris data visualization and KNN classification](https://www.kaggle.com/code/skalskip/iris-data-visualization-and-knn-classification)
  - Kaggle範例[K-Nearest Neighbor with Iris Data set](https://www.kaggle.com/code/susree64/k-nearest-neighbor-with-iris-data-set/notebook)
  - Kaggle資料集 [Pistachio Dataset](https://www.kaggle.com/datasets/muratkokludataset/pistachio-dataset)
    - OZKAN IA., KOKLU M. and SARACOGLU R. (2021). Classification of Pistachio Species Using Improved K-NN Classifier. Progress in Nutrition, Vol. 23, N. 2, pp. DOI:10.23751/pn.v23i2.9686. (Open Access) https://www.mattioli1885journals.com/index.php/progressinnutrition/article/view/9686/9178
    - SINGH D, TASPINAR YS, KURSUN R, CINAR I, KOKLU M, OZKAN IA, LEE H-N., (2022). Classification and Analysis of Pistachio Species with Pre-Trained Deep Learning Models, Electronics, 11 (7), 981. https://doi.org/10.3390/electronics11070981. (Open Access) 
  - YOUTUBE 教學影片
    - [2.6 K-nearest neighbors in Python (L02: Nearest Neighbor Methods)](https://www.youtube.com/watch?v=PtjeiDpHss8)
- Naive Bayes
- Decision Trees決策樹
- [Support Vector Machines 支援向量機](https://scikit-learn.org/stable/modules/svm.html#support-vector-machines)
  - SVC, NuSVC and LinearSVC are classes capable of performing binary and multi-class classification on a dataset.

## 效能評估 see [Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
    - The `sklearn.metrics` module implements several loss, score, and utility functions to measure classification performance
    - 3.3.2.2. Accuracy score
    - 3.3.2.7. Classification report

## 效能調教 
- [sklearn.model_selection](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection): gridsearchCV 
- 交叉驗證 cross validation

## 專題
   - 智慧金融之信用卡詐欺偵測[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) 
   - 人力資源管理:員工流失率預測模型[Human Resources Analytics: A Descriptive Analysis](https://www.kaggle.com/code/colara/human-resources-analytics-a-descriptive-analysis)
   - 客戶流失率預測模型
