# In pandas groupby agg, usage of dictionary is deprecated
https://github.com/pandas-dev/pandas/issues/16337
```python
df.groupby(df['City'])['isFraud'].agg({'Fraud':sum,
                                       'Non-Fraud': lambda x: len(x)-sum(x),
                                       'Squared': lambda x: (sum(x))**2})
FutureWarning: using a dict on a Series for aggregation
is deprecated and will be removed in a future version

**Solution
result = df.groupby(df['City'])['isFraud'].agg(['sum', 'size'])
result = pd.DataFrame({'Fraud': result['sum'],
                        'NonFraud': result['size']-result['sum'],
                        'Squared': result['sum']**2})

** Another option is create functions for all cases
def Fraud(group):
     return group.sum()

def NonFraud(group):
     return len(group)-sum(group)

def Squared(group):
     return (sum(group))**2

df.groupby(df['City'])['isFraud'].agg([Fraud, NonFraud, Squared])

```

# Deprecation example 2
```python
import numpy as np
import statsmodels.robust as smrb
from functools import partial

mydf = pd.DataFrame(
    {
        'cat': ['A', 'A', 'A', 'B', 'B', 'C'],
        'energy': [1.8, 1.95, 2.04, 1.25, 1.6, 1.01],
        'distance': [1.2, 1.5, 1.74, 0.82, 1.01, 0.6]
    },
    index=range(6)
)

# median absolute deviation as a partial function
# in order to demonstrate the issue with partial functions as aggregators
mad_c1 = partial(smrb.mad, c=1)

# renaming and specifying the aggregators at the same time
# note that I want to choose the resulting column names myself
# for example "total_xxxx" instead of just "sum"
mydf_agg = mydf.groupby('cat').agg({
    'energy': {
        'total_energy': 'sum',
        'energy_p98': lambda x: np.percentile(x, 98),  # lambda
        'energy_p17': lambda x: np.percentile(x, 17),  # lambda
    },
    'distance': {
        'total_distance': 'sum',
        'average_distance': 'mean',
        'distance_mad': smrb.mad,   # original function
        'distance_mad_c1': mad_c1,  # partial function wrapping the original function
    },
})

# get rid of the first MultiIndex level in a pretty straightforward way
mydf_agg.columns = mydf_agg.columns.droplevel(level=0)


print(mydf_agg)

# This gives WARNING but outputs in pandas 0.23
total_energy  energy_p98  energy_p17  total_distance  average_distance  \
cat
A            5.79      2.0364      1.8510            4.44             1.480
B            2.85      1.5930      1.3095            1.83             0.915
C            1.01      1.0100      1.0100            0.60             0.600

     distance_mad  distance_mad_c1
cat
A        0.355825            0.240
B        0.140847            0.095
C        0.000000            0.000



#==================================================================================
# New syntax
def my_agg(x):
    data = {'energy_sum': x.energy.sum(),
            'energy_p98': np.percentile(x.energy, 98),
            'energy_p17': np.percentile(x.energy, 17),
            'distance sum' : x.distance.sum(),
            'distance mean': x.distance.mean(),
            'distance MAD': smrb.mad(x.distance),
            'distance MAD C1': mad_c1(x.distance)}
    return pd.Series(data)

mydf.groupby('cat').apply(my_agg)

#==================================================================================
# Another way

# median absolute deviation as a partial function
# in order to demonstrate the issue with partial functions as aggregators
mad_c1 = partial(smrb.mad, c=1)

# Identical dictionary passed to `agg`
funcs = {
    'energy': {
        'total_energy': 'sum',
        'energy_p98': lambda x: np.percentile(x, 98),  # lambda 98th percentile
        'energy_p17': lambda x: np.percentile(x, 17),  # lambda 17th percentile
    },
    'distance': {
        'total_distance': 'sum',
        'average_distance': 'mean',
        'distance_mad': smrb.mad,   # original function
        'distance_mad_c1': mad_c1,  # partial function wrapping the original function
    },
}

# Write a proxy method to be passed to `pipe`
def agg_assign(grouped, fdict):
    data = { (column, name): grouped[column].agg(fn) 
                             for column, column_dict in fdict.items()
                             for name, fn in column_dict.items() }
    return pd.DataFrame(data)

# All the API we need already exists with `pipe`
mydf.groupby('cat').pipe(agg_assign, fdict=funcs)
```
