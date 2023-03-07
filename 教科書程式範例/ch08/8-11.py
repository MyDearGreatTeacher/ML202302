from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
num_pl = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler()
)
#檢查數值管道器的運作  
print(f'數值型資料的欄位有：{X_col_num}')
num_pl.fit_transform(df[X_col_num])[:3]