# DBSCAN 1996
- Density-based spatial clustering of applications with noise (DBSCAN)
- [重要值得看的wiki](https://en.wikipedia.org/wiki/DBSCAN) 
- [DBSCAN Clustering Algorithm in Machine Learning(2022)](https://www.kdnuggets.com/2020/04/dbscan-clustering-algorithm-machine-learning.html#:~:text=low%20point%20density.-,Density%2DBased%20Spatial%20Clustering%20of%20Applications%20with%20Noise%20(DBSCAN),is%20containing%20noise%20and%20outliers.)
- [【機器學習】基於密度的聚類演算法 DBSCAN(2020)](https://jason-chen-1992.weebly.com/home/-dbscan)
# [DBSCAN 範例 from sklearn.cluster.DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
```python
from sklearn.cluster import DBSCAN
import numpy as np

X = np.array([[1, 2], [2, 2], [2, 3],[8, 7], [8, 8], [25, 80]])

clustering = DBSCAN(eps=3, min_samples=2).fit(X)

clustering.labels_

clustering
```
- [Demo of DBSCAN clustering algorithm官方網站範例]()
# [DBSCAN 範例](https://github.com/pyinvest/ml_toturial/blob/master/DBSCAN.ipynb)
```python
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN

# 下載資料
iris=datasets.load_iris()
X=iris.data
X=X[:,2:4]

# 重要參數: eps: 半徑 min_samples: 最少點數
clustering=DBSCAN(eps=0.3,min_samples=10).fit(X)
clustering

clustering.labels_

plt.scatter(X[:,0],X[:,1],c=clustering.labels_)

y=iris.target
plt.scatter(X[:,0],X[:,1],c=y)
```

```python
iris=datasets.load_iris()
X=iris.data
X
clustering=DBSCAN(eps=0.3,min_samples=10).fit(X)
clustering.labels_
```
