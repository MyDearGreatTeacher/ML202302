# 載入資料和資料預處理
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer()
X, y = breast_cancer['data'], breast_cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                   test_size=0.33, random_state=2)

model_pl = Pipeline([
    ('preprocess', StandardScaler()),
    ('model', LogisticRegression())
])
param_grid = [
    {'model':[LogisticRegression()], 'model__penalty': ['l1', 'l2'], 
     'model__C':[0.001,0.01,1,5,10]},
    {'model':[SVC()], 'model__kernel':['linear','rbf'], 
     'model__C': [0.1, 1, 10, 100, 1000],
     'model__gamma': [1, 0.1, 0.01, 0.001, 0.0001]},
    {'model':[KNeighborsClassifier()], 
     'model__n_neighbors':[5,10,15,20,25]},
    {'model':[DecisionTreeClassifier()], 
     'model__min_samples_split':[5, 10, 15, 20, 30]}
]
gs = GridSearchCV(model_pl, param_grid=param_grid,
                  cv=10, return_train_score=True)
gs.fit(X_train, y_train)
score = gs.best_estimator_.score(X_test, y_test)
print('最佳模型', gs.best_params_['model'])
print('最佳交叉驗證的結果', gs.best_score_)
print('最後測試集的結果', score)