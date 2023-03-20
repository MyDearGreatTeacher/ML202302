# TSNE:t-隨機鄰近嵌入法  [WIKI](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)
- t-distributed stochastic neighbor embedding
- Van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of machine learning research, 9(11)
- 👍[Van der Maaten官方網址](https://lvdmaaten.github.io/tsne/) 
  - 有許多補充資料 
- [Visualizing Data Using t-SNE(2013)](https://www.youtube.com/watch?v=RJVL80Gg3lA&list=UUtXKDgv1AVoG88PLl8nGXmw)
- [機器學習_學習筆記系列(78)：t-隨機鄰近嵌入法(t-distributed stochastic neighbor embedding)](https://tomohiroliu22.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E5%AD%B8%E7%BF%92%E7%AD%86%E8%A8%98%E7%B3%BB%E5%88%97-78-t-%E9%9A%A8%E6%A9%9F%E9%84%B0%E8%BF%91%E5%B5%8C%E5%85%A5%E6%B3%95-t-distributed-stochastic-neighbor-embedding-a0ed57759769)
- [[筆記] 如何使用 t-SNE 進行降維](https://mortis.tech/2019/11/program_note/664/)
- [利用降維技巧檢視資料分群狀態：PCA, tSNE, SVD, SOM](https://ithelp.ithome.com.tw/m/articles/10278992)
- [淺談降維方法中的 PCA 與 t-SNE](https://medium.com/d-d-mag/%E6%B7%BA%E8%AB%87%E5%85%A9%E7%A8%AE%E9%99%8D%E7%B6%AD%E6%96%B9%E6%B3%95-pca-%E8%88%87-t-sne-d4254916925b)
- [t-SNE：可视化效果最好的降维算法](https://zhuanlan.zhihu.com/p/327699974?utm_id=0)
- 教學影片[t-SNE(T-distributed Stochastic Neighbourhood Embedding)](https://www.youtube.com/playlist?list=PLupD_xFct8mHqCkuaXmeXhe0ajNDu0mhZ)
- t-SNE is a tool to visualize high-dimensional data. 
- It converts similarities between data points to joint probabilities and tries to minimize the Kullback-Leibler divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data. 
- t-SNE has a cost function that is not convex, i.e. with different initializations we can get different results.

# Student t-distribution
- [WIKI](https://en.wikipedia.org/wiki/Student%27s_t-distribution) [中文版WIKI](https://zh.wikipedia.org/wiki/%E5%8F%B8%E5%BE%92%E9%A0%93t%E5%88%86%E5%B8%83)
- 在機率論及統計學中用於根據小樣本來估計母體呈常態分布且標準差未知的期望值。
- 若母體標準差已知，或是樣本數足夠大時（依據中央極限定理漸進常態分布），則應使用常態分布來進行估計。

# [sklearn.manifold.TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE)
```python

import numpy as np
from sklearn.manifold import TSNE

X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
X_embedded = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=3).fit_transform(X)

X_embedded.shape
```
