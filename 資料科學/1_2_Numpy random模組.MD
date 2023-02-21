## [Numpy random模組](https://numpy.org/doc/1.16/reference/routines.random.html)
- Numpy random模組有許多函數
- [Random Numbers in NumPy](https://www.w3schools.com/python/numpy/numpy_random.asp)

## 產生亂數樣本
```python
import numpy as np
np.random.seed(0) 

x1 = np.random.randint(10, size=6) 
x2 = np.random.randint(10, size=(3, 4)) 
x3 = np.random.randint(10, size=(3, 4, 5)) 
x4 = np.random.rand(3,2)
x5 = np.random.randint(5,10,size=(3, 4))
x5
```

```
1. rand(d0,d1,.....,dn)產生[0,1]的浮點隨機數,括號裡面的引數可以指定產生陣列的形狀
   例如：np.random.rand(3,2)則產生 3×2的陣列，裡面的數是0～1的浮點隨機數
 
2.randn(d0,d1,...,dn)產生標準正則分佈隨機數(N)，引數含義與rand相同
 
3.randint(low,high,size)產生指定範圍的隨機數位於半開區間[low,high)，最後一個引數是元組，他確定陣列的形狀
```

## 產生特殊機率分布的樣本
- [Binomial distribution]()
- [範例說明](https://www.w3schools.com/python/numpy/numpy_random_binomial.asp)
```python
from numpy import random

x = random.binomial(n=10, p=0.5, size=10)
x2 = random.binomial(n=10, p=0.5, size=(2,3))
print(x)
```
- 比較不同機率分布
```python
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(random.normal(loc=50, scale=5, size=1000), hist=False, label='normal')
sns.distplot(random.binomial(n=100, p=0.5, size=1000), hist=False, label='binomial')

plt.show()
```

### (1)Simple random data
```
rand(d0, d1, …, dn)	Random values in a given shape.
randn(d0, d1, …, dn)	Return a sample (or samples) from the “standard normal” distribution.
randint(low[, high, size, dtype])	Return random integers from low (inclusive) to high (exclusive).
random_integers(low[, high, size])	Random integers of type np.int between low and high, inclusive.
random_sample([size])	Return random floats in the half-open interval [0.0, 1.0).
random([size])	Return random floats in the half-open interval [0.0, 1.0).
ranf([size])	Return random floats in the half-open interval [0.0, 1.0).
sample([size])	Return random floats in the half-open interval [0.0, 1.0).
choice(a[, size, replace, p])	Generates a random sample from a given 1-D array
bytes(length)	Return random bytes.
```
## (2)Permutations
```
shuffle(x)	Modify a sequence in-place by shuffling its contents.
permutation(x)	Randomly permute a sequence, or return a permuted range.
```
### (3)Distributions各式各樣的機率分布
```
beta(a, b[, size])	Draw samples from a Beta distribution.
binomial(n, p[, size])	Draw samples from a binomial distribution.
chisquare(df[, size])	Draw samples from a chi-square distribution.
dirichlet(alpha[, size])	Draw samples from the Dirichlet distribution.
exponential([scale, size])	Draw samples from an exponential distribution.
f(dfnum, dfden[, size])	Draw samples from an F distribution.
gamma(shape[, scale, size])	Draw samples from a Gamma distribution.
geometric(p[, size])	Draw samples from the geometric distribution.
gumbel([loc, scale, size])	Draw samples from a Gumbel distribution.
hypergeometric(ngood, nbad, nsample[, size])	Draw samples from a Hypergeometric distribution.
laplace([loc, scale, size])	Draw samples from the Laplace or double exponential distribution with specified location (or mean) and scale (decay).
logistic([loc, scale, size])	Draw samples from a logistic distribution.
lognormal([mean, sigma, size])	Draw samples from a log-normal distribution.
logseries(p[, size])	Draw samples from a logarithmic series distribution.
multinomial(n, pvals[, size])	Draw samples from a multinomial distribution.
multivariate_normal(mean, cov[, size, …)	Draw random samples from a multivariate normal distribution.
negative_binomial(n, p[, size])	Draw samples from a negative binomial distribution.
noncentral_chisquare(df, nonc[, size])	Draw samples from a noncentral chi-square distribution.
noncentral_f(dfnum, dfden, nonc[, size])	Draw samples from the noncentral F distribution.
normal([loc, scale, size])	Draw random samples from a normal (Gaussian) distribution.
pareto(a[, size])	Draw samples from a Pareto II or Lomax distribution with specified shape.
poisson([lam, size])	Draw samples from a Poisson distribution.
power(a[, size])	Draws samples in [0, 1] from a power distribution with positive exponent a - 1.
rayleigh([scale, size])	Draw samples from a Rayleigh distribution.
standard_cauchy([size])	Draw samples from a standard Cauchy distribution with mode = 0.
standard_exponential([size])	Draw samples from the standard exponential distribution.
standard_gamma(shape[, size])	Draw samples from a standard Gamma distribution.
standard_normal([size])	Draw samples from a standard Normal distribution (mean=0, stdev=1).
standard_t(df[, size])	Draw samples from a standard Student’s t distribution with df degrees of freedom.
triangular(left, mode, right[, size])	Draw samples from the triangular distribution over the interval [left, right].
uniform([low, high, size])	Draw samples from a uniform distribution.
vonmises(mu, kappa[, size])	Draw samples from a von Mises distribution.
wald(mean, scale[, size])	Draw samples from a Wald, or inverse Gaussian, distribution.
weibull(a[, size])	Draw samples from a Weibull distribution.
zipf(a[, size])	Draw samples from a Zipf distribution.
```
### (4)Random generator亂數產生器
```
RandomState([seed])	Container for the Mersenne Twister pseudo-random number generator.
seed([seed])	Seed the generator.
get_state()	Return a tuple representing the internal state of the generator.
set_state(state)	Set the internal state of the generator from a tuple.
```
