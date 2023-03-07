from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, 2)
selector.fit(X_train, y_train)
selector.get_support()