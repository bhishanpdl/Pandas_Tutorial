# Pandas transform support only one aggregation, but we can use list comp
```python
import numpy as np
import pandas as pd

df = pd.DataFrame({'a': list("aabbbb"),
                   'w': [1,2,3,4,5,6],
                   'x': [10,20,30,30,40,130],
                   'y': [40,60,70,60,70,80],
                   'z': [1,4,3,1,5,np.nan]})
# fast and easy way
df['w_sum'] = df.groupby('a')['w'].transform('sum')
df['w_count'] = df.groupby('a')['w'].transform('count')
df['w_mean'] = df.groupby('a')['w'].transform('mean')

# slow and complicated with warning of levels
# This gives warning of merging 1 levels on the left, 2 on the right
df1 = df[['a']].join(df.groupby('a').agg({'w': ['sum', 'count','mean']}), on='a')
df1 = df1.rename(columns = lambda x: '_'.join(x))
print(df1)

# slow and complicated but no warning of levels
df1 = df.groupby('a').agg({'w': ['sum', 'count','mean']})
df1.columns = [i[0]+'_'+i[1] if i[1] else i[0] for i in df1.columns.ravel()]
df2 = df[['a']].join(df1,on='a')

# piping using slow method
(df[['a']].join(

    # new dataframe from agg
    df.groupby('a')
   .agg({'w': ['sum', 'count','mean']})

   # make one level columns
   .pipe(lambda x: x.set_axis([i[0]+'_'+i[1] if i[1] else i[0]
                               for i in x.columns.ravel()],
                               axis=1, inplace=False))

    # join on column a
    ,on='a')
)
# NOTE: df.join gives all columns + new columns.


# Result
   a  w_sum  w_count  w_mean
0  a      3        2     1.5
1  a      3        2     1.5
2  b     18        4     4.5
3  b     18        4     4.5
4  b     18        4     4.5
5  b     18        4     4.5
```

# transform multiple-functions after groupby multiple-columns
```python
# fast and easy way
df['w_sum'] = df.groupby(['a','x'])['w'].transform('sum')
df['w_count'] = df.groupby(['a','x'])['w'].transform('count')
df['w_mean'] = df.groupby(['a','x'])['w'].transform('mean')

**aliter
df.assign(
    w_sum = lambda dff: dff.groupby(['a','x'])['w'].transform('sum'),
    w_count = lambda dff: dff.groupby(['a','x'])['w'].transform('count'),
    w_mean = lambda dff: dff.groupby(['a','x'])['w'].transform('mean')
          )

# slow and complicated way
dfagg = df.groupby(['a','x'],as_index=False).agg({'w': ['sum', 'count','mean']})
dfagg.columns = [i[0]+'_'+i[1] if i[1] else i[0] for i in dfagg.columns.ravel()]

dftrans = df[['a','x','w']].merge(dfagg, on=['a','x'],how='left')
print(dftrans)

   a    x  w  w_sum  w_count  w_mean
0  a   10  1      1        1     1.0
1  a   20  2      2        1     2.0
2  b   30  3      7        2     3.5
3  b   30  4      7        2     3.5
4  b   40  5      5        1     5.0
5  b  130  6      6        1     6.0
```
