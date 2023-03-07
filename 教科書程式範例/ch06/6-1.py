import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.rcParams['font.sans-serif'] = ['DFKai-sb'] 
plt.rcParams['axes.unicode_minus'] = False
%config InlineBackend.figure_format = 'retina'
import warnings
warnings.filterwarnings('ignore')

x = np.linspace(-10,10,1000)
# sigmoid function
y = 1/(1+np.exp(-x))
plt.plot(x,y)
plt.axhline(0.5, c='k', ls='--')
plt.axvline(0, c='k', ls='--')
plt.annotate('切割點(0, 0.5)', xy=(0,0.5), fontsize=14, 
             xytext=(20,10), textcoords='offset points', 
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))
plt.ylim(0,1);