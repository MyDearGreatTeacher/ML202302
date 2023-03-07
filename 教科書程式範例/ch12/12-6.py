from sklearn.model_selection import RandomizedSearchCV
vc = VotingClassifier([
    ('lr', model_pl_lr),    
    ('svc', model_pl_svc), 
    ('tree', model_pl_tree), 
    ('knn', model_pl_knn)], 
    voting='soft', weights=[2, 2, 1, 1])
weights = {'weights':mesh}
np.random.seed(42)
rgs = RandomizedSearchCV(vc, param_distributions=weights, 
                         n_iter=30, cv=10, random_state=42)
rgs.fit(X_train, y_train)
print('訓練集的預測結果', rgs.best_score_)
print('測試集預測結果',rgs.score(X_test, y_test))
print('最佳權重選擇',rgs.best_params_)