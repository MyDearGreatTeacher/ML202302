corr = df.corr().round(2)
corr['target'].sort_values(ascending=False)