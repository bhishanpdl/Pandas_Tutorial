# Groupby ranking methods
```python
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from functools import partial


df = pd.DataFrame({'A': [1, 1, 1, 2, 2],
                   'B': [1, 1, 2, 2, 1],
                   'C': [10, 20, 30, 40, 50],
                   'D': ['X', 'Y', 'X', 'Y', 'Y']})

def min_rank(x): return pd.Series.rank(x,method='min')
def min_rank_scipy(x): return rankdata(x,method='min')


df['rank_min_A'] = df.groupby('D')['A'].rank('min')
df['rank_min_A1'] = df.groupby('D')['A'].agg('rank',method='min')
df['rank_min_A2'] = df.groupby('D')['A'].apply(pd.Series.rank) # average
df['rank_min_A2a'] = df.groupby('D')['A'].apply(partial(pd.Series.rank,method='min'))
df['rank_min_A3'] = df.groupby('D')['A'].apply(min_rank)
df['rank_min_A4'] = df.groupby('D')['A'].transform(min_rank_scipy)
df['rank_min_A4a'] = df.groupby('D')['A'].apply(min_rank_scipy) # all NaNs

df['rank_min_A00'] = df.groupby('D').agg( {'A': 'rank' }) # average
# df['rank_min_A01'] = df.groupby('D').agg( {'A': pd.Series.rank }) # FAILS

print(df)

   A  B   C  D  rank_min_A  rank_min_A1  rank_min_A2  rank_min_A2a  \
0  1  1  10  X         1.0          1.0          1.5           1.0
1  1  1  20  Y         1.0          1.0          1.0           1.0
2  1  2  30  X         1.0          1.0          1.5           1.0
3  2  2  40  Y         2.0          2.0          2.5           2.0
4  2  1  50  Y         2.0          2.0          2.5           2.0

   rank_min_A3  rank_min_A4 rank_min_A4a  rank_min_A00
0          1.0            1          NaN           1.5
1          1.0            1          NaN           1.0
2          1.0            1          NaN           1.5
3          2.0            2          NaN           2.5
4          2.0            2          NaN           2.5

```
