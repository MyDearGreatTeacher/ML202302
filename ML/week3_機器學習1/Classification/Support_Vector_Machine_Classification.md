

# Support Vector Machine Classification breast cancer data set


```PYTHON
## Import libraries and load data
# Commented out IPython magic to ensure Python compatibility.
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

#Get the Data
#We'll use the built in breast cancer dataset from Scikit Learn. Note the load function:


from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

#The data set is presented in a dictionary form

cancer.keys()

#We can grab information and arrays out of this dictionary to create data frame and understand the features
#The description of features are as follows


print(cancer['DESCR'])

#Show the feature names

cancer['feature_names']

## Set up the DataFrame

df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df.info()

df.describe()

#Is there any missing data?

np.sum(pd.isnull(df).sum()) # Sum of the count of null objects in all columns of data frame

# What are the 'target' data in the data set?

cancer['target']

# Adding the target data to the DataFrame

df['Cancer'] = pd.DataFrame(cancer['target'])
df.head()

## Exploratory Data Analysis探索式資料分析

### Check the relative counts of benign (0) vs malignant (1) cases of cancer


sns.set_style('whitegrid')
sns.countplot(x='Cancer',data=df,palette='RdBu_r')

### Run a 'for' loop to draw boxlots of all the mean features (first 10 columns) for '0' and '1' CANCER OUTCOME"""

l=list(df.columns[0:10])
for i in range(len(l)-1):
    sns.boxplot(x='Cancer',y=l[i], data=df, palette='winter')
    plt.figure()

#Not all the features seperate out the cancer predictions equally clearly
#For example, from the following two plots it is clear that smaller area generally is indicative of positive cancer detection, 
#while nothing concrete can be said from the plot of mean smoothness**

f,(ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(12,6))
ax1.scatter(df['mean area'],df['Cancer'])
ax1.set_title("Cancer cases as a function of mean area", fontsize=15)
ax2.scatter(df['mean smoothness'],df['Cancer'])
ax2.set_title("Cancer cases as a function of mean smoothness", fontsize=15)

## Training and prediction

### Train Test Split


df_feat = df.drop('Cancer',axis=1) # Define a dataframe with only features
df_feat.head()

df_target = df['Cancer'] # Define a dataframe with only target results i.e. cancer detections
df_target.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_feat, df_target, test_size=0.30, random_state=101)

y_train.head()

### Train the Support Vector Classifier"""

from sklearn.svm import SVC

model = SVC()

model.fit(X_train,y_train)

### Predictions and Evaluations

predictions = model.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

#Notice that we are classifying everything into a single class! 
#This means our model needs to have it parameters adjusted (it may also help to normalize the data)
print(confusion_matrix(y_test,predictions))

#As expected, the classification report card is bad

print(classification_report(y_test,predictions))

## Gridsearch
## Finding the right parameters (like what C or gamma values to use) is a tricky task! 
## But luckily, Scikit-learn has the functionality of trying a bunch of combinations and see what works best, built in with GridSearchCV! 
## The CV stands for cross-validation.
## GridSearchCV takes a dictionary that describes the parameters that should be tried and a model to train. 
## The grid of parameters is defined as a dictionary, where the keys are the parameters and the values are the settings to be tested.** 


param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}

from sklearn.model_selection import GridSearchCV

## One of the great things about GridSearchCV is that it is a meta-estimator. 
## It takes an estimator like SVC, and creates a new estimator, that behaves exactly the same - in this case, like a classifier. 
## You should add refit=True and choose verbose to whatever number you want, higher the number, 
## the more verbose (verbose just means the text output describing the process)."""

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=1)

## First, it runs the same loop with cross-validation, to find the best parameter combination. 
## Once it has the best combination, it runs fit again on all data passed to fit (without cross-validation), 
## to built a single new model using the best parameter setting."""

# May take awhile!
grid.fit(X_train,y_train)

## You can inspect the best parameters found by GridSearchCV in the best\_params\_ attribute, and the best estimator in the best\_estimator\_ attribute

grid.best_params_

grid.best_estimator_

# Then you can re-run predictions on this grid object just like you would with a normal model

grid_predictions = grid.predict(X_test)

## Now print the confusion matrix to see improved predictions

print(confusion_matrix(y_test,grid_predictions))

## Classification report shows improved F1-score

print(classification_report(y_test,grid_predictions))

## Another set of parameters for GridSearch

param_grid = {'C': [50,75,100,125,150], 'gamma': [1e-2,1e-3,1e-4,1e-5,1e-6], 'kernel': ['rbf']} 
grid = GridSearchCV(SVC(tol=1e-5),param_grid,refit=True,verbose=1)
grid.fit(X_train,y_train)

grid.best_estimator_

grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
