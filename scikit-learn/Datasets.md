# sklearn資料集分為幾種類型： 
- [【深度學習】深度學習常用資料集](https://blog.csdn.net/weixin_47876613/article/details/118806756)
- [7. Dataset loading utilities](https://scikit-learn.org/stable/datasets.html)
  - 7.1. Toy datasets
  - 7.2. Real world datasets
  - 7.3. Generated datasets
  - 7.4. Loading other datasets
- [sklearn資料集](https://www.modb.pro/db/89613)
  - 1、自帶的小資料集（packaged dataset）：sklearn.datasets.load_
  - 2、真實世界中的資料集（Downloaded Dataset）：sklearn.datasets.fetch_
  - 3、電腦生成的資料集（Generated Dataset）：sklearn.datasets.make_
  - 4、svmlight/libsvm格式的資料集:sklearn.datasets.load_svmlight_file(...)
  - 5、從data.org線上下載獲取的資料集:sklearn.datasets.fetch_mldata(...)

- [Iris Species@kaggle](https://www.kaggle.com/datasets/uciml/iris/versions/2?resource=download)
  - [Ensemble Learning Techniques Tutorial](https://www.kaggle.com/code/pavansanagapati/ensemble-learning-techniques-tutorial) 
- [sklearn.datasets.load_iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris) 
### 1、自帶的小資料集
-  loaders 可用來載入小的標準資料集
-  資料集還包含一些對DESCR描述
-  同時一部分也包含feature_names和target_names的特徵。
```
1.1、波士頓房價資料集load_boston
資料位址：https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
該資料集是一個回歸問題。
每個類的觀察值數量是均等的，共有 506 個觀察，13 個輸入變數和1個輸出變數。
每條資料包含房屋以及房屋周圍的詳細資訊。
其中包含城鎮犯罪率，一氧化氮濃度，住宅平均房間數，到中心區域的加權距離以及自住房平均房價等等


1.2、鳶尾花資料集load_iris
著名的 Iris資料集取自 Fisher 的論文。
資料集包含 3 個類，每個類 50 個實例，
每條記錄都有 4 項特徵：花萼長度、花萼寬度、花瓣長度、花瓣寬度
通過這4個特徵預測鳶尾花卉屬於（iris-setosa, iris-versicolour, iris-virginica）中的哪一品種。



1.3、糖尿病資料集load_diabetes
資料集一共442例糖尿病資料
10 個特徵變數：年齡、性別、體重指數、平均血壓和6 個血清測量值，
預測基線後一年疾病進展的定量測量值。
來源網址：https : //www4.stat.ncsu.edu/~boos/var.select/diabetes.html


1.4、手寫數位資料集的光學識別load_digits
資料集包含1797個手寫數位的圖像：10 個類，其中每個類指一個數字。
生成一個 8x8 的輸入矩陣。
來源網址：https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits


1.5、Linnerrud 資料集load_linnerud
Linnerud 資料集是一個多元回歸資料集。
它由20 名中年男性收集的三個運動資料：chins、仰臥起坐、跳遠和三個生理目標變數：體重、腰圍、脈搏組成。

範例教學:An exploration of sklearn's Linnerrud dataset  
https://www.youtube.com/watch?v=oBfIl8XzydY
https://github.com/TracyRenee61/Misc-Predictions/blob/main/Linnerrud_dataset.ipynb

1.6、葡萄酒識別資料集load_wine
資料一共有178行資料
特徵值為酒精、蘋果酸等葡萄酒的十三種檢測值，
預測屬於三種葡萄酒中的哪一種。
來源網址：
https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data


1.7、乳腺癌威斯康辛（診斷）資料集load_breast_cancer
資料集包含569個樣本，從乳房腫塊的細針抽吸（FNA）的數位化圖像計算特徵。
它們描述了圖像中存在的細胞核的30個特徵：半徑、紋理、周長等等。
預測癌症是良性還是惡性。
來源網址：https://goo.gl/U2Uwz2
```

### 2、真實世界中的資料集
```
2.1、Olivetti 人臉資料集
該資料集包含1992 年 4 月至 1994 年 4 月在 AT&T 劍橋實驗室拍攝的一組面部圖像
由40個人的400張圖片構成，即每個人的人臉圖片為10張。
每張圖片的灰度級為8位元，每個圖元的灰度大小位於0-255之間，每張圖片大小為64×64。。

2.2、20 個新聞群組文本資料集
20 newsgroups資料集18000多篇新聞文章，一共涉及到20種話題，所以稱作20newsgroups text dataset
分為兩部分：訓練集和測試集，通常用來做文本分類，均勻分為20個不同主題的新聞群組集合。
20newsgroups資料集是被用於文本分類、文本挖據和資訊檢索研究的國際標準資料集之一。


2.3、Wild 人臉識別資料集
本資料集是網上收集的名人JPEG圖片的集合
詳細資訊可在官網查看：http://vis-www.cs.umass.edu/lfw/
scikit-learn提供了兩個載入器，它們將自動下載、緩存、解析中繼資料檔、解碼 jpeg 並將有趣的切片轉換為 memapped numpy 陣列。
此資料集大小超過 200 MB。
第一次載入通常需要超過幾分鐘才能將 JPEG 檔的相關部分完全解碼為 numpy 陣列。。

2.4、森林覆蓋類型
資料集記錄的是美國 Colorado 植被覆蓋類型資料，也是唯一一個關心真實森林的資料。
每條記錄都包含很多指標描述每一塊土地。例如：高度、坡度、到水的距離、樹蔭下的面積、土壤的類型等等。
森林的覆蓋類型是需要根據其他54個特徵進行預測的特徵。
這是一個有趣的資料集，它包含分類和數值特徵。總共有581012條記錄。
每條記錄有55列，其中一列是土壤的類型，其他54列是輸入特徵。
有七種覆蓋類型，使其成為一個多類分類問題。
來源網址：https://archive.ics.uci.edu/ml/datasets/Covertype

2.5、RCV1 資料集
路透社語料庫第一卷 (RCV1) 是一個包含超過 800,000 個手動分類的新聞專線故事的檔案，由 Reuters, Ltd. 提供，用於研究目的。
來源網址：https://jmlr.csail.mit.edu/papers/volume5/lewis04a/

2.6、Kddcup 99 資料集
資料集是從一個類比的美國空軍局域網上採集來的9個星期的網路連接資料，分成具有標識的訓練資料和未加標識的測試資料。
測試資料和訓練資料有著不同的概率分佈,測試資料包含了一些未出現在訓練資料中的攻擊類型,這使得入侵偵測更具有現實性。
在訓練資料集中包含了1種正常的標識類型normal和22種訓練攻擊類型。

2.7、加州住房資料集
目標變數是加利福尼亞地區的房屋價值中位數。
該資料集源自 1990 年美國人口普查，每個人口普查區塊組使用一行。
計算了各個街區群形心點之間的距離（以緯度和經度衡量），並排除了針對引數和因變數未報告任何條目的所有街區群。
最終資料包含涉及 8個特徵和一個預測值，一共 20640 條觀察資料。
來源網址：http://lib.stat.cmu.edu/datasets/
```


### 3、電腦生成的資料集
- 3.1、單標籤
  - make_blobs 和 make_classification 通過分配每個類的一個或多個正態分佈的點的群集創建的多類資料集。
  - make_blobs 對於中心和各簇的標準差提供了更好的控制，可用於演示聚類。
  - make_classification 專門通過引入相關的，冗餘的和未知的噪音特徵；將高斯集群的每類複雜化；在特徵空間上進行線性變換。
  - make_gaussian_quantiles 將single Gaussian cluster （單高斯簇）分成近乎相等大小的同心超球面分離。
  - make_hastie_10_2 產生類似的二進位、10維問題。
  - make_circles和make_moon生成二維分類資料集時可以説明確定演算法（如質心聚類或線性分類），包括可以選擇性加入高斯雜訊。它們有利於視覺化。
    - make_circles生成高斯資料，帶有球面決策邊界以用於二進位分類
    - make_moon生成兩個交叉的半圓。
```
sklearn.datasets.make_blobs(n_samples=100, n_features=2, *, centers=None, cluster_std=1.0, center_box=- 10.0, 10.0, shuffle=True, random_state=None, return_centers=False)

sklearn.datasets.samples_generator.make_classification
(n_samples=100, n_features=20, n_informative=2, n_redundant=2,
 n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, 
 flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
  shuffle=True, random_state=None)

sklearn.datasets.samples_generator.make_gaussian_quantiles(mean=None, cov=1.0, n_samples=100, n_features=2, n_classes=3, 
shuffle=True, random_state=None)
```

- 3.2、多標籤
  - make_multilabel_classification 生成多個標籤的隨機樣本，
```python
sklearn.datasets.make_multilabel_classification(n_samples=100,
 n_features=20, *, n_classes=5, n_labels=2, length=50, 
 allow_unlabeled=True, sparse=False, return_indicator='dense', 
 return_distributions=False, random_state=None)

pick the number of labels: n ~ Poisson(n_labels)：選取標籤的數目
n times, choose a class c: c ~ Multinomial(theta) ：n次,選取類別C:多項式
pick the document length: k ~ Poisson(length) ：選取文檔長度
k times, choose a word: w ~ Multinomial(theta_c)：k次,選取一個單詞
```
- 3.3、雙聚類
```python
sklearn.datasets.make_biclusters(shape, n_clusters, *, noise=0.0, minval=10, maxval=100, shuffle=True, random_state=None) 

sklearn.datasets.make_checkerboard(shape, n_clusters, *, noise=0.0, minval=10, maxval=100, shuffle=True, random_state=None)
```


- 3.4、回歸生成器
  - make_regression 產生的回歸目標作為一個可選擇的稀疏線性組合的具有雜訊的隨機的特徵。
```python
sklearn.datasets.make_regression(n_samples=100, n_features=100,
 *, n_informative=10, n_targets=1, bias=0.0, effective_rank=None, 
 tail_strength=0.5, noise=0.0, shuffle=True, coef=False, random_state=None)
```


- 3.5、流形學習生成器
```python
sklearn.datasets.make_s_curve(n_samples=100, *, noise=0.0, random_state=None)

sklearn.datasets.make_swiss_roll(n_samples=100, *, noise=0.0, random_state=None)
```


- 3.6、生成器分解
```python

sklearn.datasets.make_low_rank_matrix(n_samples=100,n_features=100, *, effective_rank=10, tail_strength=0.5, random_state=None)

sklearn.datasets.make_sparse_coded_signal(n_samples, *, n_components, n_features, n_nonzero_coefs, random_state=None)

sklearn.datasets.make_spd_matrix(n_dim, *, random_state=None)

sklearn.datasets.make_sparse_spd_matrix(dim=1, *, alpha=0.95,norm_diag=False, smallest_coef=0.1, largest_coef=0.9, random_state=None)
```
### 4、載入其他資料集
```
4.1、樣本圖片
scikit 在通過圖片的作者共同授權下嵌入了幾個樣本 JPEG 圖片。
這些圖像為了方便使用者對 test algorithms （測試演算法）和 pipeline on 2D data （二維資料管道）進行測試。

sklearn.datasets.load_sample_images()
sklearn.datasets.load_sample_image(image_name)



4.2、svmlight或libsvm格式的資料集
scikit-learn 中有載入svmlight libsvm格式的資料集的功能函數。
svmlight libsvm 格式的公共資料集:
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets
更快的API相容的實現: https://github.com/mblondel/svmlight-loader


4.3、從openml.org下載資料集
openml.org是一個用於機器學習資料和實驗的公共存儲庫，它允許每個人上傳開放的資料集。
在sklearn.datasets包中，可以通過sklearn.datasets.fetch_openml函數來從openml.org下載資料集．


4.4、從外部資料集載入
scikit-learn使用任何存儲為numpy陣列或者scipy稀疏陣列的數值資料。
其他可以轉化成數值陣列的類型也可以接受，如pandas中的DataFrame。

以下推薦一些將標準縱列形式的資料轉換為scikit-learn可以使用的格式的方法:
pandas.io 提供了從常見格式（包括 CSV、Excel、JSON 和 SQL）讀取資料的工具。
DataFrames 也可以從元組或字典的清單中構建。
Pandas 可以流暢地處理異構資料，並提供用於操作和轉換為適合 scikit-learn 的數值陣列的工具。
scipy.io 專門研究科學計算環境中常用的二進位格式，例如 .mat 和 .arff
numpy/routines.io 用於將柱狀資料標準載入到 numpy 陣列中

datasets.load_svmlight_file用於 svmlight 或 libSVM 稀疏格式的scikit-learn

scikit-learndatasets.load_files用於文字檔的目錄，其中每個目錄的名稱是每個類別的名稱，每個目錄中的每個檔對應於該類別的一個樣本

對於圖片、視頻、音訊等一些雜資料，您不妨參考：
skimage.io或 Imageio 用於將圖像和視頻載入到 numpy 陣列中
scipy.io.wavfile.read 用於將 WAV 檔讀入 numpy 陣列
```

- [Bike Sharing Demand@Kaggle](https://www.kaggle.com/competitions/bike-sharing-demand/overview)
```python
from sklearn.datasets import fetch_openml

bikes = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True, parser="pandas")
# Make an explicit copy to avoid "SettingWithCopyWarning" from pandas
X, y = bikes.data.copy(), bikes.target
X
```

```python
from sklearn.datasets import load_boston

boston=load_boston()

boston.keys()
```

```python
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train')

from pprint import pprint
print(list(newsgroups_train.target_names))
```
