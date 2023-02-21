## Seaborn
  - [seaborn: statistical data visualization](https://seaborn.pydata.org/) 
  - Colab 已有支援
  - [User guide and tutorial](https://seaborn.pydata.org/tutorial.html)

## 檢查Colab上版本
```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.__version__
```
## 測試1
```python
# 生成 100000 組標準常態分配（平均值為 0，標準差為 1 的常態分配）隨機變數
normal_samples = np.random.normal(size = 100000) 
sns.distplot(normal_samples)
```
