df.groupby('Sex')['Survived'].value_counts().\
unstack(1).plot(kind='bar', figsize=(5,3));