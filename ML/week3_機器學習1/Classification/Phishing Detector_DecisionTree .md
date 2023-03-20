
# Phishing Detector.ipynb

- [資料來源](https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/tree/master/Chapter03/sources)
- [參考書籍:Hands-On Artificial Intelligence for Cybersecurity(2019)Alessandro Parisi](https://www.packtpub.com/product/hands-on-artificial-intelligence-for-cybersecurity/9781789804027)
  - chapter 5. Ham or Spam? Detecting Email Cybersecurity Threats with AI




!wget https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/master/Chapter03/datasets/phishing_dataset.csv


```python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings 
warnings.simplefilter('ignore')

phishing_dataset = np.genfromtxt('./phishing_dataset.csv', delimiter=',', dtype=np.int32)
samples = phishing_dataset[:,:-1]
targets = phishing_dataset[:, -1]

from sklearn.model_selection import train_test_split

training_samples, testing_samples, training_targets, testing_targets = train_test_split(
         samples, targets, test_size=0.2, random_state=0)
```
## DecisionTree Phishing Detector
```
from sklearn import tree
tree_classifier = tree.DecisionTreeClassifier()

tree_classifier.fit(training_samples, training_targets)

predictions = tree_classifier.predict(testing_samples)

accuracy = 100.0 * accuracy_score(testing_targets, predictions)
print ("Decision Tree accuracy: " + str(accuracy))
```
## LogisticRegression Phishing Detector
```
log_classifier = LogisticRegression()
log_classifier.fit(training_samples, training_targets)

predictions = log_classifier.predict(testing_samples)
accuracy = 100.0 * accuracy_score(testing_targets, predictions)
print ("Logistic Regression accuracy: " + str(accuracy))
```
