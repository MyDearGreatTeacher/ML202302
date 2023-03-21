# Matplotlib: Visualization with Python
- [matplotlib](https://matplotlib.org/)
- [Documentation](https://matplotlib.org/stable/index.html)
- [Tutorials]()
- [Example gallery](https://matplotlib.org/stable/gallery/index.html)

## [Quick start guide](https://matplotlib.org/stable/tutorials/introductory/quick_start.html)
```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])  #
```

- [matplotlib.pyplot.subplots](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html)
  - 輸入參數 ==> 請參看網址說明
  - 回傳結果(Returns): 
    - fig Figure
    - ax Axes or array of Axes

## MAtplotlib
![matplotlib.png](./matplotlib.png)

## 
```python
fig = plt.figure()  # an empty figure with no Axes

fig, ax = plt.subplots()  # a figure with a single Axes

fig, axs = plt.subplots(2, 2)  # a figure with a 2x2 grid of Axes

# a figure with one axes on the left, and two on the right:
fig, axs = plt.subplot_mosaic([['left', 'right-top'],['left', 'right_bottom']])
```


##
```python
np.random.seed(19680801)  # seed the random number generator.
data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
ax.scatter('a', 'b', c='c', s='d', data=data)
ax.set_xlabel('entry a')
ax.set_ylabel('entry b')
```


##
```python


```


##
```python


```


##
```python


```


##
```python


```


##
```python


```


##
```python


```
