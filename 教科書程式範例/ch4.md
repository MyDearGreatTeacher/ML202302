#
```python
# 基本套件和模組
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
plt.rcParams['font.sans-serif'] = ['DFKai-sb'] 
plt.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore')

# 資料模組
from sklearn.datasets import load_boston
boston = load_boston()
```

```python
boston.keys()
```


```python
print('\n'.join(boston['DESCR'].split('\n')[:26]))
```


```python
print(boston['feature_names'])
```


```python
df = pd.DataFrame(boston['data'], columns = boston['feature_names'])
df.head()
```


```python
df['target'] = boston['target']
df.head()
```


```python
df.info()
```


```python
df['target'].plot(kind='hist', bins=30, alpha=0.5)
```


```python
corr = df.corr().round(2)
corr['target'].sort_values(ascending=False)
```


```python
plt.figure(figsize=(8, 6))
corr[np.abs(corr) < 0.6] = 0
sns.heatmap(corr, annot=True, cmap='coolwarm');
```


```python

```


```python

```


```python

```
