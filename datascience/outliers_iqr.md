# Filter outliers using iqr
```python
import numpy as np
import pandas as pd

np.random.seed(100)
df = (
    # A standard distribution
    pd.DataFrame({'nb': np.random.randint(0, 100, 20)})
        # Adding some outliers
        .append(pd.DataFrame({'nb': np.random.randint(100, 200, 2)}))
        # Reseting the index
        .reset_index(drop=True)
    )

q1 = df['nb'].quantile(0.25)
q3 = df['nb'].quantile(0.75)
iqr_nb = q3 - q1 # 61.0

filtered = df.query('(@q1 - 1.5 * @iqr_nb) <= nb <= (@q3 + 1.5 * @iqr_nb)')

df.join(filtered, rsuffix='_filtered').boxplot()

# NOTE:
import scipy.stats
from scipy.stats import iqr as IQR

iqr = IQR(df.nb.values) # 61.0
```
