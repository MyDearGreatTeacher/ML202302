## `資料科學(Data Science)`

- [中文Wiki說明]([https://en.wikipedia.org/wiki/Data_science](https://zh.wikipedia.org/wiki/%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6))
  - 資料科學（英語：data science）又稱數據科學，是一門利用資料（數據）學習知識的學科
  - 目標是通過從資料中提取出有價值的部分來生產資料產品
  - 學科範圍涵蓋了：資料取得、資料處理、資料分析等過程，舉凡與資料有關的科學均屬資料科學。
  - 資料科學結合了諸多領域中的理論和技術，包括應用數學、統計、圖型識別、機器學習、資料視覺化、資料倉儲以及高效能計算。
  - 資料科學通過運用各種相關的資料來幫助非專業人士理解問題。 
  - 資料科學技術可以幫助我們如何正確的處理資料並協助我們在生物學、社會科學、人類學等領域進行研究調研。資料科學也對商業競爭有極大的幫助。
  - 美國國家標準技術研究所於2015年發表七卷巨量資料參考框架（NIST Big Data Reference Architecture，NBDRA），
    - 於第一卷定義篇中將資料科學定為在理論科學、實驗科學和計算科學之後的第四科學範式
- [Wiki說明](https://en.wikipedia.org/wiki/Data_science)
  - Data science is an interdisciplinary field focused on extracting knowledge from typically large data sets and applying the knowledge 
  - and insights from that data to solve problems in a wide range of application domains


## `資料科學(Data Science)_常用套件`

- 資料科學  Data Science
  - numpy
  - pandas
  - scipy
  - statsmodels(統計分析)

- 科學計算  scientific calculation
  - sympy
  - scipy
- 電腦視覺
  - opencv
- 資料視覺化  Data Visulization
  - matplotlib
  - seaborn
  - plotly
  - .....

- 機器學習
  - scikit-learn (sklearn)  

- 機器學習與人工智慧
  - Tensorflow
  - Pytorch (torch)


## 作業 1: 確認開發環境的套件版本
- Google Colab ==> !pip list
- anaconda
  - [Anaconda介紹及安裝教學](https://medium.com/python4u/anaconda%E4%BB%8B%E7%B4%B9%E5%8F%8A%E5%AE%89%E8%A3%9D%E6%95%99%E5%AD%B8-f7dae6454ab6) 

## 作業 2:如何載入套件

- import pandas as pd
- import numpy as np 
- import tensorflow as tf 

## 作業 3:如何學習或是使用套件
```
import numpy as np

b = np.linspace(0, 2, 4)
b

# array([0.        , 0.66666667, 1.33333333, 2.        ])
```
```
c = np.linspace(0, 2, 4, endpoint=False) 
c

# array([0. , 0.5, 1. , 1.5])
```
- 學習函式(function) linspace()的使用
  - [官方說明 numpy.linspace](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html)
  - [Python numpy linspace用法及代碼示例](https://vimsky.com/zh-tw/examples/usage/python-numpy.linspace.html)
```
numpy.linspace(
start, 
stop, 
num=50, == >
endpoint=True, 
retstep=False, 
dtype=None, 
axis=0)
```
- 學習重點:
  - 函式(function)參數有哪些個?
    - 那些參數有預設值?
    - 那些參數一定要填
  - 函式(function)回傳的資料
    - 回傳的資料型態
    - 可以有幾種接收值
- 範例: numpy.linspace()
  - 函式(function)參數有哪些個?
    - 共有多少個參數? 7 個 ==> start, stop, num,endpoint, retstep, dtype, axis
    - 那些參數有預設值?
      - num=50,  預設值=>50
      - endpoint=True, 預設值=>True
      - retstep=False, 預設值=>False
      - dtype=None, 預設值=>None
      - axis=0   預設值=>0
    - 那些參數一定要填
      - start, stop,
  - 函式(function)回傳的資料
    - 回傳的資料型態 ==> ndarray
    - 可以有幾種接收值 ===> ndarray
