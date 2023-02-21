# 
- [Writing mathematical expressions](https://matplotlib.org/stable/tutorials/text/mathtext.html)
- [Plot Mathematical Expressions in Python using Matplotlib](https://www.geeksforgeeks.org/plot-mathematical-expressions-in-python-using-matplotlib/)

## Elements of a Figure
- [解說圖形組成](https://github.com/PacktPublishing/Matplotlib-3.0-Cookbook/blob/master/Chapter01/Chapter%201%20-%20Anatomy%20of%20Matplotlib.ipynb)

## 官方範例
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 5, 0.1)
y = np.sin(x)
fig, ax = plt.subplots()
ax.plot(x, y)
```
## 使用plot()畫折線圖
```pyhton
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 100)    # 建立含100個元素的陣列
y1 = np.sin(x)                       # sin函數
y2 = np.cos(x)                      # cos函數
```
- 基本線條
```python
plt.plot(x, y)                      
plt.show()
```
- 設定線條寬度
```python
plt.plot(x, y1, lw = 2)             # 線條寬度是 2
plt.plot(x, y2, linewidth = 5)      # 線條寬度是 5                
plt.show()
```
- 設定線條顏色
```python
plt.plot(x, y1, color='c')          # 設定青色cyan            
plt.plot(x, y2, color='r')          # 設定紅色red
plt.show()
```
- legend()
```python
plt.plot(x, y1, label='Sin')                    
plt.plot(x, y2, label='Cos')
plt.legend()
plt.grid()                          # 顯示格線
plt.show()
```
### 線條的樣式
```python
import matplotlib.pyplot as plt

d1 = [1, 2, 3, 4, 5, 6, 7, 8]           
d2 = [1, 3, 6, 10, 15, 21, 28, 36]     
d3 = [1, 4, 9, 16, 25, 36, 49, 64]     
d4 = [1, 7, 15, 26, 40, 57, 77, 100]  

plt.plot(d1, linestyle = 'solid')       # 預設實線
plt.plot(d2, linestyle = 'dotted')      # 虛點樣式
plt.plot(d3, ls = 'dashed')             # 虛線樣式
plt.plot(d4, ls = 'dashdot')            # 虛線點樣式
plt.show()
```
### 節點的樣式
```python
import matplotlib.pyplot as plt

d1 = [1, 2, 3, 4, 5, 6, 7, 8]           
d2 = [1, 3, 6, 10, 15, 21, 28, 36]      
d3 = [1, 4, 9, 16, 25, 36, 49, 64]      
d4 = [1, 7, 15, 26, 40, 57, 77, 100]    

seq = [1, 2, 3, 4, 5, 6, 7, 8]
plt.plot(seq,d1,'-',marker='x')
plt.plot(seq,d2,'-',marker='o')
plt.plot(seq,d3,'-',marker='^')
plt.plot(seq,d4,'-',marker='s') 
plt.show()
```
- [matplotlib.markers](https://matplotlib.org/stable/api/markers_api.html)

### 標題 | x 軸 | y 軸
```python
import matplotlib.pyplot as plt

temperature = [23, 22, 20, 24, 22, 22, 23, 20, 17, 18,
               20, 20, 16, 14, 14, 20, 20, 20, 15, 14,
               14, 14, 14, 16, 16, 16, 18, 21, 21, 20,
               16]
x = [x for x in range(1,len(temperature)+1)]        
plt.plot(x, temperature)
plt.title("Weather Report", fontsize=24)
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.show()
```
## [scatter](https://matplotlib.org/stable/plot_types/basic/scatter_plot.html#sphx-glr-plot-types-basic-scatter-plot-py) 
```
import matplotlib.pyplot as plt
import numpy as np

# make the data
np.random.seed(3)
x = 4 + np.random.normal(0, 2, 24)
y = 4 + np.random.normal(0, 2, len(x))

# size and color:
sizes = np.random.uniform(15, 80, len(x))
colors = np.random.uniform(15, 80, len(x))

# plot
fig, ax = plt.subplots()

ax.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()
```
## 產生滿足機率分布的直方圖
```python
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

# 生成 100000 組標準常態分配（平均值為 0，標準差為 1 的常態分配）隨機變數
normal_samples = np.random.normal(size = 100000) 

# 生成 100000 組介於 0 與 1 之間均勻分配隨機變數
uniform_samples = np.random.uniform(size = 100000) 

plt.hist(normal_samples)
plt.show()
plt.hist(uniform_samples)
plt.show()
```


## 多表並陳(subplot、subplots)
- 程式範例 [建立多個子圖表 ( subplot、subplots )](https://steam.oxxostudio.tw/category/python/example/matplotlib-subplot.html)

- [subplot(row, column, index)](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.subplot.html)
```python
import matplotlib.pyplot as plt
x = [1,2,3,4,5]
y = [5,4,3,2,1]
fig = plt.figure()
plt.subplot(221)
plt.plot(x)
plt.subplot(224)
plt.plot(y)
plt.show()
```
- [matplotlib.pyplot.subplots](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html#matplotlib.pyplot.subplots)
```python
import matplotlib.pyplot as plt
x = [1,2,3,4,5]
y = [5,4,3,2,1]
fig, ax = plt.subplots(2,2)
ax[0][0].plot(x)
ax[1][1].plot(y)
plt.show()
```
