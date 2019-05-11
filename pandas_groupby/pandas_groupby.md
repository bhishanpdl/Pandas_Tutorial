Table of Contents
=================
   * [groupby month and day from date column](#groupby-month-and-day-from-date-column)
   * [groupby and take top 2](#groupby-and-take-top-2)
   * [groupby example](#groupby-example)
   * [grouping countries by continents](#grouping-countries-by-continents)


# groupby month and day from date column

```python
# get max value for a calendar-day of any year
df = pd.DataFrame({'id' : list('abcde'),
          'date' : ['3/22/2019', '3/22/2019', '3/23/2019', '3/22/2017', '3/22/2018'],
          'name' : list('vwxyz'),
          'value' : range(10,15)})

df['date'] = pd.to_datetime(df['date'],format='%m/%d/%Y')

df.groupby(df['date'].dt.strftime('%m-%d'))['value'].max()
df.loc[df.groupby(df['date'].dt.strftime('%m-%d'))['value'].idxmax()].sort_index()
```



# groupby and take top 2
```python
# Thinking process
df = pd.DataFrame({'id':[1,1,1,1,1,1,1,1,1,2,2,2,2,2], 
                    'value':[20,20,20,30,30,30,30,40, 40,10, 10, 40,40,40]})
print(df)
    id  value
0    1     20
1    1     20
2    1     20
3    1     30
4    1     30
5    1     30
6    1     30
7    1     40
8    1     40
9    2     10
10   2     10
11   2     40
12   2     40
13   2     40


grouped = df.groupby('id')['value']
for name, group in grouped:
  print(name)
  print(group)
  x = group
  break
  
Result:
======
name =  1
0    20
1    20
2    20
3    30
4    30
5    30
6    30
7    40
8    40

code:
x.value_counts()
30    4
20    3
40    2

x.value_counts().head(2)
30    4
20    3
** this is what we need!!

Solution:
===***====
df.groupby('id')['value']\
  .apply(lambda x: x.value_counts().head(2))\
  .reset_index(name='count')\
  .rename(columns={'level_1':'value'})
  
#-----------------------------------------------------------------
# another example
df = pd.DataFrame({'id':[9,4,1,1,1,2,2,2,2,3,4],
                   'value':[90,1,1,2,3,1,2,3,4,1,1]})

arr = [[n, g.nlargest(2).tolist()] for n, g in df.groupby('id')['value']]
dct = dict([n, g.nlargest(2).tolist()] for n, g in df.groupby('id')['value'])
pd.DataFrame(arr)

Result:
=========
0       1
1  [3, 2]
2  [4, 3]
3     [1]
4  [1, 1]
9    [90]

#-----------------------------------------------------------------
# Another method
# groupby id and take only top 2 values.
df = pd.DataFrame({'id':[1,1,1,1,1,1,1,1,1,2,2,2,2,2],
               'value':[20,20,20,30,30,30,30,40, 40,10, 10, 40,40,40]})

(df.groupby('id')['value']
.value_counts()
 .groupby(level=0)
 .nlargest(2)
 .to_frame()
 .rename(columns={'value':'count'})
 .reset_index([1,2]) # reset only index 1 and 2, do not reset index 0
 .reset_index(drop=True))

# detailed way
x = df.groupby('id')['value'].value_counts().groupby(level=0).nlargest(2).to_frame()
x.columns = ['count']
x.index = x.index.droplevel(0)
x = x.reset_index()
x

Result:
=======
   id  value  count
0   1     30      4
1   1     20      3
2   2     40      3
3   2     10      2


```

# groupby example
https://realpython.com/python-pandas-tricks/
```python
url = ('https://archive.ics.uci.edu/ml/'
       'machine-learning-databases/abalone/abalone.data')
cols = ['sex', 'length', 'diam', 'height', 'weight', 'rings']
abalone = pd.read_csv(url, usecols=[0, 1, 2, 3, 4, 8], names=cols)

abalone.head()

abalone['ring_quartile'] = pd.qcut(abalone.rings, q=4, labels=range(1, 5))
grouped = abalone.groupby('ring_quartile')

for idx, frame in grouped:
    print(f'Ring quartile: {idx}')
    print('-' * 16)
    print(frame.nlargest(3, 'weight'), end='\n\n')
    
grouped.groups.keys() # dict_keys([1, 2, 3, 4])
grouped.get_group(2).head() # second group NOT THIRD!!
grouped['height', 'weight'].agg(['mean', 'median'])

# perform query
agg_dict = {
'length': ['sum', 'mean', 'max', 'min', 'std', 'median', 'nunique', 'skew'],
'diam': ['min', 'max', 'mean']
}

grouped = (abalone.groupby(['ring_quartile', 'sex'])
           .agg(agg_dict)
           .sort_values(by=('length', 'median'), ascending=False)
           .query("sex == 'M'"))
```

# grouping countries by continents
https://realpython.com/python-pandas-tricks/
```python
countries = pd.Series([
    'United States',
    'Canada',
    'Mexico',
    'Belgium',
    'United Kingdom',
    'Thailand'
])

groups = {
    'North America': ('United States', 'Canada', 'Mexico', 'Greenland'),
    'Europe': ('France', 'Germany', 'United Kingdom', 'Belgium')
}


from typing import Any

def membership_map(s: pd.Series, groups: dict,
                   fillvalue: Any=-1) -> pd.Series:
    # Reverse & expand the dictionary key-value pairs
    groups = {x: k for k, v in groups.items() for x in v}
    return s.map(groups).fillna(fillvalue)
    
df = pd.DataFrame({'country': countries, 'group': mapper})
```
