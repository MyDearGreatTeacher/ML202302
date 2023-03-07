from sklearn.preprocessing import KBinsDiscretizer
# 欄位
X_col_num = ['Fare', 'Age']
X_col_bin = ['SibSp', 'Parch']
X_col_cat = ['Pclass', 'Sex', 'Embarked']
# 資料管道器
num_pl = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler()
)
bin_pl = make_pipeline(
    SimpleImputer(strategy='mean'),
    KBinsDiscretizer(n_bins=5, encode='ordinal'),
)
cat_pl = make_pipeline(
    SimpleImputer(strategy='constant', fill_value='missing'),
    OneHotEncoder()
)
# 合併後的資料管道器
data_pl = ColumnTransformer([
    ('num', num_pl, X_col_num),
    ('bin', bin_pl, X_col_bin),
    ('cat', cat_pl, X_col_cat)
])
# 模型預測
model_pl = make_pipeline(data_pl, SVC())
model_pl.fit(X_train, y_train)
y_pred = model_pl.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print('整體正確率:',accuracy_score(y_test, y_pred).round(2))