sns.pairplot(df, vars=['mean radius','mean texture','mean perimeter','mean area'], 
             hue='target', size=2);