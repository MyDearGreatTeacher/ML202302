pd.DataFrame(cat_pl.fit_transform(df[X_col_cat]), 
             columns=oh_cols).head()