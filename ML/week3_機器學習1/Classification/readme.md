## 教學主題
- 分類演算法
  - [k-nearest neighbors algorithm (k-NN)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- 分類演算法效能評估指標
- 專案應用(Binary classification):Pima印地安人糖尿病預測模型
  - [Deep learning approach for diabetes prediction using PIMA Indian dataset(2020)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7270283/) 
  - [Pima Indians Diabetes - Prediction & KNN Visualization](https://towardsdatascience.com/pima-indians-diabetes-prediction-knn-visualization-5527c154afff)
- 專案應用(Nulti-class classification):IRIS
- 專案應用(Nulti-class classification):路透社新聞分類

## DATASET :Pima印地安人糖尿病
- [原始論文:Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus(1988)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2245318/)
  - The National Institute of Diabetes Digestive and Kidney Diseases、The Johns Hopkins University School of Medicine共同發表
  - Proc Annu Symp Comput Appl Med Care. 1988 Nov 9 : 261–265
- [Pima Indians Diabetes Database@Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  

- Pima印第安人(Phoenix, Arizona)是一批糖尿病高風險發病的族群
- 該族群居民自1965年起每隔兩年都會接受身體檢查。
- 如果”口服葡萄糖耐受測試”(oral glucose tolerance test)後的2小時血糖數值，高於200 mg/dl 即視為糖尿病患者。
- 更有768筆記錄
- 每筆紀錄有9個欄位：8 特徵(F) + 1 target(答案)
```
1.Number of times pregnant 懷孕次數
2.Plasma Glucose Concentration at 2 Hours in an Oral Tolerance Test (GTT) 口服葡萄糖耐受測試後2小時的血糖數據
3.Diastolic Blood pressure 血壓(舒張壓) mmHg
4.Triceps Skin Fold Thickness (mm) 肱三頭肌皮膚厚度
5.2-Hour Serum Insulin 2小時後血清胰島素數據
6.Body mass index BMI
7.Diabetes Pedigree Function族譜系數
8.Age (years)年齡
```
- [7.Diabetes Pedigree Function族譜系數 請參看說明](https://ithelp.ithome.com.tw/m/articles/10263714)
  - [程式碼](https://github.com/neoCaffe/SkLearnPractice) 


## 分析
- EDA
- Decision Tree Classification
  - [EDA+Decision Tree Classification on different data](https://www.kaggle.com/code/jasleensondhi/eda-decision-tree-classification-on-different-data) 
- svm
  - [Pima Indians Diabetes SVM](https://www.kaggle.com/code/baiazid/pima-indians-diabetes-svm/notebook) 
- Random Forest
  - [Diabetes Test with Random Forest](https://www.kaggle.com/code/abdallahhassan22/diabetes-test-with-random-forest) 
