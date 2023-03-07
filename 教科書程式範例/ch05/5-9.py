from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
model_pl = make_pipeline(StandardScaler(), LinearRegression())
model_pl.fit(X_train, y_train)