# grouby group_keys example
```python
Note: group_keys=False  to get one column, no need to do .values or to do .droplevel(level=0)
Note: as_index=False    no need to do reset_index()


import numpy as np
import pandas as pd
df = pd.DataFrame({'x':['A','A','B','B'],'y':[10,20,30,40],'z':[1,3,3,5]})

# note: group A mean = 2 and groupB mean = 4
df['new'] = df.groupby('x', group_keys=False).apply(lambda g: g.y - g.z.mean())
df['new1'] = df.groupby('x').apply(lambda g: g.y - g.z.mean()).values # we need .values
df['new2'] = df.groupby('x').apply(lambda g: g.y - g.z.mean()).droplevel(level=0) # or, droplevel 0

print(df)

'''
   x   y  z   new  new1  new2
0  A  10  1   8.0   8.0   8.0
1  A  20  3  18.0  18.0  18.0
2  B  30  3  26.0  26.0  26.0
3  B  40  5  36.0  36.0  36.0
'''

```
