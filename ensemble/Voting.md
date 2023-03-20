# Voting


# 
- 三個base learner(基礎學習器)：
  - 感知器（具有單個神經元的神經網絡） ==>  sklearn.linear_model   Perceptron
  - 支持向量機( SVM ) ==> sklearn.svm   SVC
  - 最近鄰  ==> sklearn.neighbors   KNeighborsClassifier
- breast_cancer分類數據集 
```python
# Import the required libraries
from sklearn import datasets, linear_model, svm, neighbors
from sklearn.metrics import accuracy_score
from numpy import argmax
# Load the dataset
breast_cancer = datasets.load_breast_cancer()
x, y = breast_cancer.data, breast_cancer.target
```
## base learner(基礎學習器)
```python

# Instantiate the learners (classifiers)
learner_1 = neighbors.KNeighborsClassifier(n_neighbors=5)
learner_2 = linear_model.Perceptron(tol=1e-2, random_state=0)
learner_3 = svm.SVC(gamma=0.001)
```
## 獨立訓練
```python
# Split the train and test samples
test_samples = 100
x_train, y_train = x[:-test_samples], y[:-test_samples]
x_test, y_test = x[-test_samples:], y[-test_samples:]

# Fit learners with the train data
learner_1.fit(x_train, y_train)
learner_2.fit(x_train, y_train)
learner_3.fit(x_train, y_train)
```

## 進行預測
```python

# Each learner predicts the classes of the test data
predictions_1 = learner_1.predict(x_test)
predictions_2 = learner_2.predict(x_test)
predictions_3 = learner_3.predict(x_test)
```

## HARD VOTING
```python
# We combine the predictions with hard voting
hard_predictions = []

# For each predicted sample
for i in range(test_samples):
    # Count the votes for each class
    counts = [0 for _ in range(2)]
    counts[predictions_1[i]] = counts[predictions_1[i]]+1
    counts[predictions_2[i]] = counts[predictions_2[i]]+1
    counts[predictions_3[i]] = counts[predictions_3[i]]+1
    
    # Find the class with most votes
    final = argmax(counts)
    
    # Add the class to the final predictions
    hard_predictions.append(final) 
```

```PYTHON
# Accuracies of base learners
print('L1:KNeighborsClassifier', accuracy_score(y_test, predictions_1))
print('L2:Perceptron', accuracy_score(y_test, predictions_2))
print('L3:SVC', accuracy_score(y_test, predictions_3))
# Accuracy of hard voting
print('-'*30)
print('Hard Voting:', accuracy_score(y_test, hard_predictions))
```
# 使用 scikit-learn
-  sklearn.ensemble  VotingClassifier

```PYTHON
# Import the required libraries
from sklearn import datasets, linear_model, svm, neighbors
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
# Load the dataset
breast_cancer = datasets.load_breast_cancer()
x, y = breast_cancer.data, breast_cancer.target

# Split the train and test samples
test_samples = 100
x_train, y_train = x[:-test_samples], y[:-test_samples]
x_test, y_test = x[-test_samples:], y[-test_samples:]


# Instantiate the learners (classifiers)
learner_1 = neighbors.KNeighborsClassifier(n_neighbors=5)
learner_2 = linear_model.Perceptron(tol=1e-2, random_state=0)
learner_3 = svm.SVC(gamma=0.001)
```


```PYTHON
# Instantiate the voting classifier
voting = VotingClassifier([('KNN', learner_1),
                           ('Prc', learner_2),
                           ('SVM', learner_3)])

```


```PYTHON
# Fit classifier with the training data
voting.fit(x_train, y_train)


# Predict the most voted class
hard_predictions = voting.predict(x_test)


# Accuracy of hard voting
print('-'*30)
print('Hard Voting:', accuracy_score(y_test, hard_predictions))
```

# SOFT VOTING
- Scikit-learn 也允許進行軟投票。
- 唯一的要求是基礎學習者實現該predict_proba功能。
- Perceptron根本不實現該函數，而SVC僅在傳遞參數時才產生概率probability=True。
- Perceptron中實現的樸素貝葉斯分類器sklearn.naive_bayes。

要實際使用軟投票，VotingClassifier必須使用參數初始化對象voting='soft'。
```PYTHON
# Import the required libraries
from sklearn import datasets, naive_bayes, svm, neighbors
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score


# Load the dataset
breast_cancer = datasets.load_breast_cancer()
x, y = breast_cancer.data, breast_cancer.target

# Split the train and test samples
test_samples = 100
x_train, y_train = x[:-test_samples], y[:-test_samples]
x_test, y_test = x[-test_samples:], y[-test_samples:]

#  
# Instantiate the learners (classifiers)
learner_1 = neighbors.KNeighborsClassifier(n_neighbors=5)
learner_2 = naive_bayes.GaussianNB() # 
learner_3 = svm.SVC(gamma=0.001, probability=True)  # 請注意 probability=True為了讓GaussianNB 能夠產生概率

#  
# Instantiate the voting classifier
voting = VotingClassifier([('KNN', learner_1),
                           ('NB', learner_2),
                           ('SVM', learner_3)],
                            voting='soft')  # 請注意 voting='soft'

# 
# Fit classifier with the training data
voting.fit(x_train, y_train)
learner_1.fit(x_train, y_train)
learner_2.fit(x_train, y_train)
learner_3.fit(x_train, y_train)

# 
# Predict the most probable class
hard_predictions = voting.predict(x_test)

#  
# Get the base learner predictions
predictions_1 = learner_1.predict(x_test)
predictions_2 = learner_2.predict(x_test)
predictions_3 = learner_3.predict(x_test)

#  
# Accuracies of base learners
print('L1:', accuracy_score(y_test, predictions_1))
print('L2:', accuracy_score(y_test, predictions_2))
print('L3:', accuracy_score(y_test, predictions_3))

# Accuracy of hard voting
print('-'*30)
print('Hard Voting:', accuracy_score(y_test, hard_predictions))
```
