from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
df = df[['sepal width (cm)', 'petal length (cm)']]
df['target'] = iris['target']
df = df.iloc[50:]
X_cols = df.columns.drop('target')
y_col = 'target'
X = df[X_cols]
y = df[y_col]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                   test_size=0.33, random_state=42)