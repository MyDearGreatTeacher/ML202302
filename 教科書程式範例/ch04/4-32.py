# 先將原本的二次方欄位刪除
X_train.drop('RM2', axis=1, inplace=True)
X_test.drop('RM2', axis=1, inplace=True)

# 觀察前五筆資料與手動增加的二次方項是相同的，多出來的1不用去理它
from sklearn.preprocessing import PolynomialFeatures
polynomial = PolynomialFeatures(degree=2)
x_poly = polynomial.fit_transform(X_train)
x_poly[:5]