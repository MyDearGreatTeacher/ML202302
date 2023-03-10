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


## Image Processing


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


## SciPy:線性代數之解聯立方程式
- linear algebra


```python
#importing the scipy and numpy packages
from scipy import linalg
import numpy as np

#Declaring the numpy arrays
a = np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]])
b = np.array([2, 4, -1])

#Passing the values to the solve function
x = linalg.solve(a, b)

#printing the result array
print(x)
```


## SciPy:線性代數之Finding a Determinant
```python
#importing the scipy and numpy packages
from scipy import linalg
import numpy as np

#Declaring the numpy array
A = np.array([[1,2],[3,4]])

#Passing the values to the det function
x = linalg.det(A)

#printing the result
print(x)
```


## SciPy:線性代數之Eigenvalues and Eigenvectors
```python
#importing the scipy and numpy packages
from scipy import linalg
import numpy as np

#Declaring the numpy array
A = np.array([[1,2],[3,4]])

#Passing the values to the eig function
l, v = linalg.eig(A)

#printing the result for eigen values
print(l)

#printing the result for eigen vectors
print(v)

```


## [Singular Value Decomposition(SVD)]()
- https://www.tutorialspoint.com/scipy/scipy_linalg.htm
```python
#importing the scipy and numpy packages
from scipy import linalg
import numpy as np

#Declaring the numpy array
a = np.random.randn(3, 2) + 1.j*np.random.randn(3, 2)

#Passing the values to the eig function
U, s, Vh = linalg.svd(a)

# printing the result
print(U, Vh, s)

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
