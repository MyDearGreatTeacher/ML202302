# 第一行是為了能看欄位裡所有的值
pd.set_option('display.max_colwidth', -1) 
df_cv = pd.DataFrame(gs.cv_results_)[['params','mean_test_score']].\
sort_values(by = 'mean_test_score', ascending=False).head(12)
df_cv