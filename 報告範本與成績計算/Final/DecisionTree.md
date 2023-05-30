# DecisionTree



# 如何建立決策數? == > Attribute Selection(屬性選擇)
- 理論
  - Quinlan's earlier ID3 algorithm
- 資訊增益(Information Gain)（使用`熵(Entropy) `計算）
- 吉尼係數(Gini)（使用`不純度(impurity)`計算）
- 參看底下說明 [Day 22 : 決策樹](https://ithelp.ithome.com.tw/articles/10276079)


## 使用scikit-learn開發
- [API 参考中文版說明](https://scikit-learn.org.cn/lists/3.html)
- [sklearn.tree.ExtraTreeClassifier中文版說明](https://scikit-learn.org.cn/view/786.html)
- [sklearn.tree.ExtraTreeClassifier](https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/modules/generated/sklearn.tree.ExtraTreeClassifier.html)

- 範例
```python
from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# 如果是2.0版的話,from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, random_state=0)

from sklearn.ensemble import ExtraTreesClassifier
extra_tree = ExtraTreesClassifier(criterion='entropy',
n_estimators=10,
random_state=1,
n_jobs=-1)

extra_tree.fit(X_train, y_train)
extra_tree.score(X_train, y_train)
extra_tree.feature_importances_ # 特徵權重
```
