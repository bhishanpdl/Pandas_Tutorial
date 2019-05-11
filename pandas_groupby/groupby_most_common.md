```python
import numpy as np
import pandas as pd
import scipy.stats
from collections import Counter

df = pd.DataFrame({'Country' : ['USA', 'USA', 'Russia','USA'],
                  'City' : ['New-York', 'New-York', 'Sankt-Petersburg', 'New-York'],
                  'Short name' : ['NY','New','Spb','NY']})
print(df)
  Country              City Short name
0     USA          New-York         NY
1     USA          New-York        New
2  Russia  Sankt-Petersburg        Spb
3     USA          New-York         NY

#==================================================================================
# fastest
def get_most_common(x):
  return max(Counter(x).items(), key = lambda i: i[1])[0]

df.groupby(['Country','City']).agg(get_most_common)

## easiest 2nd fastest
df.groupby(['Country','City']).agg(pd.Series.mode)

## third
df.groupby(['Country','City']).agg(lambda x: scipy.stats.mode(x)[0][0])

## last
df.groupby(['Country','City']).agg(lambda x:x.value_counts().index[0])
```
