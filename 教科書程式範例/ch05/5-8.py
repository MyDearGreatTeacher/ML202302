pd.DataFrame(zip(X.columns, model.coef_), columns=['變數','係數']).\
sort_values(by='係數', ascending=False)