import pandas as pd
import numpy as np

data = {
    'size': ['M','S',np.nan,'M','XL'],
    'color': ['green', 'blue', 'blue', np.nan, np.nan],
    'price': [200, np.nan, 200, 300, 300],
    'quantity': [np.nan, 35000, np.nan, 20000, 10000]
}
X = pd.DataFrame(data)
X_orig = X.copy()
X.style.highlight_null(null_color='yellow')