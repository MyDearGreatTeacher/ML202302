# scipy
- [scipy](https://scipy.org/)
- [SciPy documentation](https://docs.scipy.org/doc/scipy/)
- 超爛[SciPy Tutorial - W3Schools](https://www.w3schools.com/python/scipy/index.php)
- [SciPy Tutorial (2022): For Physicists, Engineers, and Mathematicians](https://www.youtube.com/watch?v=jmX4FOUEfgU)
- [SciPy Tutorial](https://www.tutorialspoint.com/scipy/index.htm)

## scipy範例學習
- 確認Google Colab已安裝 == > from scipy import *

##
```python
import scipy

print(scipy.__version__)

```




## scipy.integrate
- https://www.tutorialspoint.com/scipy/scipy_integrate.htm
```python
import scipy.integrate
from numpy import exp

f= lambda x:exp(-x**2)
i = scipy.integrate.quad(f, 0, 1)

print(i)
```


## 積分

$$\int_{0}^{1/2} dy \int_{0}^{\sqrt{1-4y^2}} 16xy \:dx$$

```python
import scipy.integrate
from numpy import exp
from math import sqrt

f = lambda x, y : 16*x*y
g = lambda x : 0
h = lambda y : sqrt(1-4*y**2)
i = scipy.integrate.dblquad(f, 0, 0.5, g, h)

print(i)

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


##
```python


```
