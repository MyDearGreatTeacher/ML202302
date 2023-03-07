from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# 載入VotingClassfier投票組合器 
from sklearn.ensemble import VotingClassifier
model_pl_lr = make_pipeline(StandardScaler(), LogisticRegression())
model_pl_svc = make_pipeline(StandardScaler(), SVC())
model_pl_knn = make_pipeline(StandardScaler(), KNeighborsClassifier())
model_pl_tree = make_pipeline(DecisionTreeClassifier(max_depth=10))
vc = VotingClassifier([
    ('lr', model_pl_lr),    
    ('svc', model_pl_svc), 
    ('tree', model_pl_tree), 
    ('knn', model_pl_knn)], 
    voting='hard')
vc.fit(X_train, y_train)
train_score = vc.score(X_train, y_train)
test_score = vc.score(X_test, y_test)
print('訓練集的預測結果', train_score)
print('測試集的預測結果', test_score)