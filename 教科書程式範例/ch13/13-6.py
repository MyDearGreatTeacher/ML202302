size = df['left'].value_counts()
pct = df['left'].value_counts(normalize=True).round(2)
pd.DataFrame(zip(size, pct), columns=['次數', '百分比'])