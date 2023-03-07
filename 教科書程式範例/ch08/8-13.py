oh = cat_pl.named_steps['onehotencoder']
oh_cols = oh.get_feature_names(X_col_cat)
oh_cols