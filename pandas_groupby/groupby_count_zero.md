# Count also zero in gropuby count
```python
import numpy as np
import pandas as pd

df = pd.DataFrame({'date': pd.date_range('2018-01-01', periods=6),
                   'a': range(6),
                   })

df.iloc[2,0] = df.iloc[1,0]
print(df)
        date  a
0 2018-01-01  0
1 2018-01-02  1
2 2018-01-02  2
3 2018-01-04  3
4 2018-01-05  4
5 2018-01-06  5


# Do not include zeros
df1 = (df.query("a > 0")
    .groupby(['date'])[['a']]
    .count()
    .add_suffix('_count')
    .reset_index()
     )

print(df1)
        date  a_count
0 2018-01-02        2
1 2018-01-04        1
2 2018-01-05        1
3 2018-01-06        1

# Include Zeros
df1 = (df.assign(
           to_sum = (df['a']> 0).astype(int)
           )
 .groupby('date')['to_sum']
 .sum()
 .rename('a_count')
 .to_frame()
 .reset_index()

)

print(df1)
        date  a_count
0 2018-01-01        0
1 2018-01-02        2
2 2018-01-04        1
3 2018-01-05        1
4 2018-01-06        1
```
