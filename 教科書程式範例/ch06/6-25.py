y_test_proba = model_pl.predict_proba(X_test.iloc[:5])
pd.DataFrame(y_test_proba, columns=['預測1的機率', '預測2的機率'])