Table of Contents
=================
   * [keep only subset of duplicates](#keep-only-subset-of-duplicates)
   * [drop duplicated based on two columns and disregard column names](#drop-duplicated-based-on-two-columns-and-disregard-column-names)

# keep only subset of duplicates
```python
import numpy as np
import pandas as pd

df = pd.DataFrame({'col1': ['a', 'b', 'c'],
          'col2': [1, 1, 3],
          'col3': [2, 2, 2],
          'col4': ['abc', 'abc', 'def']})
print(df)
  col1  col2  col3 col4
0    a     1     2  abc
1    b     1     2  abc
2    c     3     2  def

# keep only duplicates subset
#================================================================================
df1 = df[df.duplicated(subset = ['col2', 'col3', 'col4'], keep = False)]

**aliter slow
df[df.groupby(['col2','col3','col4']).col1.transform(len) > 1]

  col1  col2  col3 col4
0    a     1     2  abc
1    b     1     2  abc

# keep only non-duplicates
#================================================================================
df.drop_duplicates(subset=['col2','col3','col4'],keep=False)
  col1  col2  col3 col4
2    c     3     2  def

# keep first duplicate
#================================================================================
  col1  col2  col3 col4
0    a     1     2  abc
2    c     3     2  def
```

# drop duplicated based on two columns and disregard column names
```python
df = pd.DataFrame({'c1' : ['A', 'A', 'B', 'B', 'A'],
          'c2' : ['B', 'C', 'A', 'D', 'B'],
          'c3' : ['x', 'y', 'x', 'z', 'y']})
print(df)
# NOTE: we treat row0 and row2 same.
  c1 c2 c3
0  A  B  x
1  A  C  y
2  B  A  x
3  B  D  z
4  A  B  y
df[~pd.DataFrame(np.sort(df[['c1','c2']], axis=1)).duplicated()]
df[~pd.DataFrame(np.sort(df[['c1','c2']], axis=1)).duplicated().values] # adding .values makes code run faster.
df[~pd.DataFrame(np.sort(df[['c1','c2']], axis=1), index=df.index).duplicated()]


  c1 c2 c3
0  A  B  x
1  A  C  y
3  B  D  z

NOTE:
pd.DataFrame(np.sort(df[['c1','c2']], axis=1)) # gives two columns 5 rows AB AC AB BD AB
```

