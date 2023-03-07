reg = model_pl.named_steps['linearregression']
pd.DataFrame(zip(X.columns, reg.coef_), columns=['變數','係數']).\
sort_values(by='係數', ascending=False)