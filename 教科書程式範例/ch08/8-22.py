# 先取到所有欄位名稱含獨熱編碼的欄位
cols = X_col_num + oh_cols.tolist()
selector = model_pl_svc.named_steps['selectkbest']
# 先將資料變成array的資料型態，再用布林值取出欄位名稱
np.array(cols)[selector.get_support()]