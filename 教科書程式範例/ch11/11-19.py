from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

models = [LogisticRegression(), SVC(), 
          KNeighborsClassifier(), DecisionTreeClassifier()]
param_lr = {'logisticregression__penalty': ['l1', 'l2'],
            'logisticregression__C':[0.001,0.01,1,5,10]}
param_svc = {
    'svc__kernel':['linear','rbf'],
    'svc__C': [0.1, 0.5, 0.8, 1, 5],
    'svc__gamma': np.arange(0.2, 1, 0.2)
}
param_knn = {'kneighborsclassifier__n_neighbors':[5,10,15,20,25]}
param_tree = {'decisiontreeclassifier__min_samples_split':[5, 10, 15, 20, 30]}
params = [param_lr, param_svc, param_knn, param_tree]
scores = {}
for model, param in list(zip(models, params)):
    print(f'Model {model.__class__.__name__} 正在進行學習和預測...')
    model_pl = make_pipeline(StandardScaler(), model)
    gs = GridSearchCV(model_pl, param_grid=param, cv=5)
    gs.fit(X_train, y_train)
    score = gs.best_estimator_.score(X_test, y_test)
    data = {
        'train_score': gs.best_score_,
        'param': gs.best_params_,
        'test_score': score
    }
    scores[model.__class__.__name__] = data
df_gs_results = pd.DataFrame(scores, index=['train_score', 'test_score']).T
df_gs_results