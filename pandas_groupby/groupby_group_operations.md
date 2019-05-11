# group operation: relative value for maximum rank
```python
import numpy as np
import pandas as pd

df = pd.DataFrame({'year': [1990,1990,1992,1992,1992],
                  'value': [100,200,300,400,np.nan],
                  'rank':  [2,1,2,1,3]})


# find relative value for maximum rank
df['value_relative']=df.value/df.groupby('year').value.transform('max') # best

df['value_relative2'] = df.value/df.loc[df.groupby('year')['rank'].transform('idxmin'),'value'].values

df['value_relative3'] = df.value/df.sort_values('rank').groupby('year').value.transform('first')

df['value_relative99']=df.groupby('year')['value'].transform(lambda x: x/x.max())


# relative value according to rank2
## note: lambda is slow, try not to use lambda operations
df['value_relative_rank2'] = df.value/df.year.map(df.loc[df['rank']==2].set_index('year')['value'])

df['value_relative_rank2A'] =df.groupby('year')['value'].transform(lambda x: x/x.nlargest(2).iloc[-1])

NOTE: df.loc[df['rank']==2].set_index('year')['value']
year
1990    100.0
1992    300.0
```
