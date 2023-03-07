df.groupby('Survived')['Age'].plot(kind='hist', alpha=0.6, 
                                   bins=30, legend=True);