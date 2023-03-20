# 區域性異常因子(Local Outlier Factor) 2000
- Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000, May). LOF: identifying density-based local outliers. In Proceedings of the 2000 ACM SIGMOD international conference on Management of data (pp. 93–104).
- M. Goldstein. FastLOF: An Expectation-Maximization based Local Outlier detection algorithm. ICPR, 2012

= [机器学习-异常检测算法（二）：Local Outlier Factor](https://zhuanlan.zhihu.com/p/28178476)


# [sklearn.neighbors.LocalOutlierFactor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)
```python
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

X = [[-1.1], [0.2], [101.1], [0.3]]

clf = LocalOutlierFactor(n_neighbors=2)

clf.fit_predict(X)

clf.negative_outlier_factor_
```
## 其他範例
- [Outlier detection with Local Outlier Factor (LOF)](https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection.html)
# [機器學習_學習筆記系列(96)：區域性異常因子(Local Outlier Factor)](https://tomohiroliu22.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E5%AD%B8%E7%BF%92%E7%AD%86%E8%A8%98%E7%B3%BB%E5%88%97-96-%E5%8D%80%E5%9F%9F%E6%80%A7%E7%95%B0%E5%B8%B8%E5%9B%A0%E5%AD%90-local-outlier-factor-a141c2450d4a)
```python
import os 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from tqdm.notebook import tqdm
```
## 使用手寫數字資料集(MNIST Dataset)
```
from sklearn.datasets import load_digits
digits = load_digits()
X=(digits.data/16)
y=digits.target
plt.rcParams["figure.figsize"] = (18,18)
plt.gray() 
for i in range(100):
    plt.subplot(20, 20, i + 1)
    plt.imshow(digits.images[i], cmap=plt.cm.gray, vmax=16, interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.show() 
```
## Principal Component Anlysis
```python
def PCA(X,n_components,N):
    X_center=X-np.mean(X,axis=0)
    W,D,V=np.linalg.svd(X_center.T)
    X_embedded=np.dot(X_center,W[:,:n_components])
    return X_embedded
```
## Reachability distance
```python
N=X.shape[0]
k=5
distance_matrix = cdist(X,X,"euclidean")
k_distance=np.sort(distance_matrix,axis=0)[k+1]
k_distance_matrix=np.outer(np.ones(N),k_distance)
reach_distacne=np.maximum(distance_matrix,k_distance_matrix)
```
## local reachability density
```
sort_index=np.argsort(distance_matrix,axis=1)[:,1:k+1]
IRD=np.zeros(N)
for i in range(N):
    IRD[i]=1/np.mean(reach_distacne[i,sort_index[i]])
```
## LOF
```
LOF=np.zeros(N)
for i in range(N):
    LOF[i]=np.mean(IRD[sort_index[i]])/IRD[i]
```
```
X_emb=PCA(X,2,N)
plt.scatter(X_emb[:,0],X_emb[:,1],c=LOF, s=100,cmap='Blues')
plt.show()
```
