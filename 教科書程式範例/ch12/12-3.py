model_pl_svc = make_pipeline(StandardScaler(), SVC(probability=True))
vc = VotingClassifier([
    ('lr', model_pl_lr),    
    ('svc', model_pl_svc), 
    ('tree', model_pl_tree), 
    ('knn', model_pl_knn)], 
    voting='soft')
vc.fit(X_train, y_train)
train_score = vc.score(X_train, y_train)
test_score = vc.score(X_test, y_test)
print('訓練集的預測結果', train_score)
print('測試集的預測結果', test_score)