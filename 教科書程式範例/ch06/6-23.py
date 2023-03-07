from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
model_pl = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear'))
model_pl.fit(X_train, y_train)
y_pred = model_pl.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm, index=['實際1', '實際2'], columns=['預測1', '預測2']))
print()
print('整體正確率:',accuracy_score(y_test, y_pred).round(2))