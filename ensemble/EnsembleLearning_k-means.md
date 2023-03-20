# Ensemble Learning with k-means
- [Hands-On Ensemble Learning with Python: Build highly optimized ensemble machine learning models using scikit-learn and Keras](https://www.packtpub.com/product/hands-on-ensemble-learning-with-python/9781789612851) [GITHUB](https://github.com/PacktPublishing/Hands-On-Ensemble-Learning-with-Python)
  - 繁體中譯本[集成式學習：Python 實踐！整合全部技術，打造最強模型](https://www.tenlong.com.tw/products/9789863126942?list_name=srh) CH8
- !pip install openensembles
```python
# Libraries and data loading
import openensembles as oe
import numpy as np
import pandas as pd
import sklearn.metrics

from sklearn.datasets import load_breast_cancer

## TSNE降維
#from sklearn.manifold import TSNE
#t = TSNE()

bc = load_breast_cancer()

# Create the data object
cluster_data = oe.data(pd.DataFrame(bc.data), bc.feature_names)

# cluster_data = oe.data(pd.DataFrame(t.fit_transform(bc.data)), [0,1])

np.random.seed(123456)
```

- 8.3/8.4 ==> oe_vote.py  VS oe_vote_tsne.py 
```python
# --- SECTION 3 ---
# Create the ensembles and calculate the homogeneity score
for K in [2, 3, 4, 5, 6, 7]:
    for ensemble_size in [3, 4, 5]:
        ensemble = oe.cluster(cluster_data)
        for i in range(ensemble_size):
            name = f'kmeans_{ensemble_size}_{i}'
            ensemble.cluster('parent', 'kmeans', name, K)

        preds = ensemble.finish_majority_vote(threshold=0.5)
        print(f'K: {K}, size {ensemble_size}:', end=' ')
        print('%.2f' % sklearn.metrics.homogeneity_score(
                bc.target, preds.labels['majority_vote']))
```
- 8.5 oe_graph_closure.py 
```python
# --- SECTION 3 ---
# Create the ensembles and calculate the homogeneity score
for K in [2, 3, 4, 5, 6, 7]:
    for ensemble_size in [3, 4, 5]:
        ensemble = oe.cluster(cluster_data)
        for i in range(ensemble_size):
            name = f'kmeans_{ensemble_size}_{i}'
            ensemble.cluster('parent', 'kmeans', name, K)

        preds = ensemble.finish_graph_closure(threshold=0.5)
        print(f'K: {K}, size {ensemble_size}:', end=' ')
        print('%.2f' % sklearn.metrics.homogeneity_score(
                bc.target, preds.labels['graph_closure']))
```


- 8.6 oe_co_occurence.py 
```python
# --- SECTION 3 ---
# Create the ensembles and calculate the homogeneity score
for K in [2, 3, 4, 5, 6, 7]:
    for ensemble_size in [3, 4, 5]:
        ensemble = oe.cluster(cluster_data)
        for i in range(ensemble_size):
            name = f'kmeans_{ensemble_size}_{i}'
            ensemble.cluster('parent', 'kmeans', name, K)

        preds = ensemble.finish_co_occ_linkage(threshold=0.5)
        print(f'K: {K}, size {ensemble_size}:', end=' ')
        print('%.2f' % sklearn.metrics.homogeneity_score(
                bc.target, preds.labels['co_occ_linkage']))
```

