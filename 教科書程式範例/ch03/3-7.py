from sklearn.preprocessing import StandardScaler, MinMaxScaler
ss = StandardScaler()
X_num_impute_ss = ss.fit_transform(X_num_impute)
X_num_impute_ss