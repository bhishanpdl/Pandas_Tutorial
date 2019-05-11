Table of Contents
=================
   * [Qn1: Find min of B after groupby A](#qn1-find-min-of-b-after-groupby-a)
   * [Qn2: Find number of non-delayed flights per airports](#qn2-find-number-of-non-delayed-flights-per-airports)
   * [Qn3: Find col1 value when col2 has two successive decrements](#qn3-find-col1-value-when-col2-has-two-successive-decrements)
   * [Qn4: Relaxed Functional Dependency (RFD) largest subset when wt_diff &lt;=2 then ht_diff&lt;=1](#qn4-relaxed-functional-dependency-rfd-largest-subset-when-wt_diff-2-then-ht_diff1)
   * [Search elements of df1.col2 in df2.col1](#search-elements-of-df1col2-in-df2col1)
   * [check two words in one column](#check-two-words-in-one-column)
   * [Mathematics](#mathematics)
   * [Multiple conditions](#multiple-conditions)
   * [First and last percentage increase in values](#first-and-last-percentage-increase-in-values)
   * [New column with numeric manipulations](#new-column-with-numeric-manipulations)
   * [Weighted average after groupby](#weighted-average-after-groupby)
   * [Wrapping aggregation functions](#wrapping-aggregation-functions)
   * [One column has list of lists](#one-column-has-list-of-lists)
   * [pandas and json to get currency conversion](#pandas-and-json-to-get-currency-conversion)
   * [Unnesting column with list](#unnesting-column-with-list)
   * [Mixed examples](#mixed-examples)
   * [New column based on element belongs to which list](#new-column-based-on-element-belongs-to-which-list)
   * [Normalized Frequency count of series elements](#normalized-frequency-count-of-series-elements)
   * [Comparing first two letters of columns of dataframes](#comparing-first-two-letters-of-columns-of-dataframes)
   * [Grouping by two columns and taking sum of same values](#grouping-by-two-columns-and-taking-sum-of-same-values)
   * [Rename only few columns](#rename-only-few-columns)
   * [Creating new column by comparing two dfs](#creating-new-column-by-comparing-two-dfs)
   * [Create dict from one column and another column last sring value](#create-dict-from-one-column-and-another-column-last-sring-value)
   * [efficient partial argsort using numpy and pandas](#efficient-partial-argsort-using-numpy-and-pandas)
   * [get index of top 3 values for each columns](#get-index-of-top-3-values-for-each-columns)
   * [Find nearest value from other column to a given column](#find-nearest-value-from-other-column-to-a-given-column)
   * [create new column nearest in value to another column](#create-new-column-nearest-in-value-to-another-column)
   * [triple double](#triple-double)
   * [check two words in one column](#check-two-words-in-one-column-1)
   * [Sort df by a value in json element of column](#sort-df-by-a-value-in-json-element-of-column)


# Qn1: Find min of B after groupby A
```python
df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
                          'foo', 'bar'],
                   'B' : [1, 2, 3, 4, 5, 6],
                   'C' : [2.0, 5., 8., 1., 2., 9.]})

Answer:
    A   B   C
0  foo  1  2.0
1  bar  2  5.0


# solution
import numpy as np
import pandas as pd

df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
                          'foo', 'bar'],
                   'B' : [1, 2, 3, 4, 5, 6],
                   'C' : [2.0, 5., 8., 1., 2., 9.]})

df

# first look at groupby, groupby gives multiindex by A
x = df.groupby('A').first()
     B    C
A          
bar  2  5.0
foo  1  2.0

# to find min value we must look at idxmin
x.loc[x.B.idxmin()]

# groupby apply
df.groupby('A').apply(lambda x: x.loc[x['B'].idxmin()])
       A  B    C
A               
bar  bar  2  5.0
foo  foo  1  2.0

# remove redundant column A before reset
print(df.groupby('A').apply(lambda x: x.loc[x['B'].idxmin()])[['B','C']].reset_index())
     A  B    C
0  bar  2  5.0
1  foo  1  2.0

when using .loc specify only B and C as columns, groupby is already grouped by A.
aliter: df.groupby('A').apply(lambda x: x.loc[x['B'].idxmin(), ['B','C']]).reset_index()

#------------------- using transform ---------------------------------------
transform keeps all rows, when operating on B, it keeps all rows of B
x == x.min() gives boolean,   and x[x== x.min()] gives actual series.

df[df.groupby('A')['B'].transform(lambda x: x == x.min())]

#----------------- using sort_values and drop_duplicates -------------------
values are sorted by B and all rows are available, then we keep duplicated 
first A values since sort values is smaller first and larger downward.

df.sort_values('B').drop_duplicates('A') # keep='first' is default

```

# Qn2: Find number of non-delayed flights per airports
```python
df = pd.DataFrame({'Origin': list('abcdefabefaf'),
                  'Cancelled': [0.,0.,1.,1.,1.,1.,0.,1.,0.,0.,1.,1.]})

df


# ***** solution****
** faster and removes zero count flights
df[df.Cancelled !=1.0].groupby('Origin')['Cancelled'].count()
Origin
a    2
b    1
e    1
f    1
713 µs ± 18.9 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)


**slower but shows all airports
df.groupby('Origin')['Cancelled'].apply(lambda x: len(x) - len(x.nonzero()))
754 µs ± 36.3 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
Origin
a    2
b    1
c    0
d    0
e    1
f    2
```

# Qn3: Find col1 value when col2 has two successive decrements
```python
import numpy as np
import pandas as pd

df = pd.DataFrame({'colA' : ['A', 'B', 'C', 'D', 'F'],
          'colB' : [10, 20, 5, 2, 30]})
print(df)

'''
Required:

col1 value when col2 have two successive decrements.

   colA colB
1   B   20


Solution:
df.loc[(df["colB"]>=df["colB"].shift(-1)) &
        (df["colB"].shift(-1)>=df["colB"].shift(-2) )]

''';
```

# Qn4: Relaxed Functional Dependency (RFD) largest subset when wt_diff <=2 then ht_diff<=1
https://stackoverflow.com/questions/54728545/how-to-get-dataframe-subsets-having-similar-values-on-some-columns
```python
import numpy as np
import pandas as pd
import networkx as nx

df = pd.DataFrame({'height' : [175, 175, 175, 176, 178, 169, 170],
          'weight' : [70, 75, 69, 71, 81, 73, 65],
          'shoe_size' : [40, 39, 40, 40, 41, 38, 39],
          'age' : [30, 41, 33, 35, 27, 49, 30]})
print(df)


'''
Relaxed Functional Dependency (RFD):

('weight': 2.0) ==> ('height': 1.0)

meaning that for each couple of rows having a difference <=2 on the weight
they would have a difference <=1 on the height too.

find the biggest subsets of rows on which this RFD holds.


From:
   height  weight  shoe_size  age
0     175      70         40   30  ** included  example, start from first row
1     175      75         39   41
2     175      69         40   33  ** included  RFD holds
3     176      71         40   35  ** included  RFD holds
4     178      81         41   27
5     169      73         38   49
6     170      65         39   30


Required:
   height  weight  shoe_size  age
0     175      70         40   30  # ht_diff = 0, wt_diff = 1
2     175      69         40   33  # ht_diff = 1, wt_diff = 1
3     176      71         40   35
''';

Solution:
values = df[['height', 'weight']].values # values.shape =  (7, 2)
dist = np.abs(values[:,None] - values) # dist.shape =  (7, 7, 2) # O(N^2)
im = (dist[:,:,0] <= 1) & (dist[:,:,1] <= 2) # im.shape = (7, 7)
G = nx.from_numpy_matrix(im) # type(G) = networkx.classes.graph.Graph

print(df.loc[sorted(nx.connected_components(G), key=len)[-1],:])

```

# Search elements of df1.col2 in df2.col1
```python
df1 = pd.DataFrame({'Text': ['text 1', 'text 2','monkey eats banana','text 4']}) # one column Text
df2 = pd.DataFrame({'Keyword': ['apple', 'banana', 'chicken'],
                    'Type':    ['fruit', 'fruit', 'meat']})  # print all text but new column Type get values from df1.Text
                    
import re
pat = rf"({'|'.join(map(re.escape, df2['Keyword']))})"  # '(apple|banana|chicken)'
df1['Type'] = (df1['Text'].str.extract(pat, expand=False)
               .map(df2.set_index('Keyword')['Type']))

```

# check two words in one column
https://stackoverflow.com/questions/54372893/pandas-filter-by-more-than-one-contains-for-not-one-cell-but-entire-column

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({'Data' : ['hello', 'world', 'hello', 'world', 'hello', 'world','hi'],
          'Data2': ['hello', 'hello', 'hello', 'hello', 'hello', 'hello','there']})
print(df)

# method 1
'hello' in df.Data.sum() and 'world' in df.Data.sum() # True

# method2
lst = ['hello','world']
any(x in df.Data.sum() for x in lst) # True

# method3
df.Data2.isin(['hello','world']).all() # False

# method4
import re 
bool(re.search(r'^(?=.*hello)(?=.*world)', df['Data'].sum()))
```


# Mathematics
#==============================================================================
```python
df = pd.DataFrame({ 'minutes': [55, 70,np.nan]})
df['hour'], df['min'] = np.divmod(df['minutes'].values,60)
```

# Multiple conditions
#==============================================================================
```python
## example 1
pd.cut(df['count'], 
       bins=[-np.inf, 1, 10, np.inf], 
       labels=['unique', 'infrequent', 'frequent'])

## example 2 *********************
### numpy where is fast
np.where(c0 > 400, 'high', 
         (np.where(c0 < 200, 'low', 'medium')))

### cut is best, memory efficient
pd.cut(df['c0'], [0, 200, 400, np.inf], labels=['low','medium','high'])

### general method  *********************
def conditions(x):
    if x > 400:
        return "High"
    elif x > 200:
        return "Medium"
    else:
        return "Low"

func = np.vectorize(conditions)
new_col = func(df["c0"])

### np.select is slowest and should not be used  *********************
col         = 'c0'
conditions  = [ df[col] >= 400, (df[col] < 400) & (df[col]> 200), df[col] <= 200 ]
choices     = [ "high", 'medium', 'low' ]
df["c0"] = np.select(conditions, choices, default=np.nan)
```


# First and last percentage increase in values
#==============================================================================
```python
a = df.drop_duplicates('name', keep='last').set_index('name')['value']
b = df.drop_duplicates('name').set_index('name')['value']
df = b.sub(a).div(a).mul(100).round(2).reset_index()

Aliter:
df = (df.groupby('name')['value']
       .agg([('mylast','last'),('value','first')])
       .pct_change(axis=1)['value']
       .mul(100)
       .reset_index())
```
 
# New column with numeric manipulations
#==============================================================================
```python
# check string is numeric (for decimal remove dot)
is_num = df["units"].astype(str).str.replace('.', '').str.isnumeric()
df.loc[is_num, 'new_col'] = df.loc[is_num, 'units']/1000 # divide by 1000 if isdigit
df.loc[is_num,'new_col']=( 
       df.loc[is_num, "price"].astype(float)* 
       df.loc[is_num, "units"]/1000)
```


# Weighted average after groupby
#==============================================================================
```python
df = pd.DataFrame({'val1': [10,20,30,40],'val2': [100,200,300,400],
                  'id': [1,1,2,2],'wt': [0.1,0.1,0.5,0.5]})
wtavg = lambda x: np.average(x.loc[:, ['val1','val2']], weights = x.wt, axis = 0)
dfwavg = df.groupby('id').apply(wtavg)
# verify
df1 = df1 = df[df.id == 1]
sum(df1.val1 * df1.wt) / df1.wt.sum() # 15.0
# verify
(10 * 0.1 + 20 * 0.1) / 0.2 # 15.0
```

# Wrapping aggregation functions
#==============================================================================
```python
def pct_between(s, low, high):
    return s.between(low, high).mean()


def make_agg_func(func, name, *args, **kwargs):
    def wrapper(x):
        return func(x, *args, **kwargs)
    wrapper.__name__ = name
    return wrapper

pct_1_3k   = make_agg_func(pct_between, 'pct_1_3k', low=1000, high=3000)
pct_10_30k = make_agg_func(pct_between, 'pct_10_30k', 10000, 30000)

# state abbreviation, religious affiliation, undergraduate students
college.groupby(['STABBR', 'RELAFFIL'])['UGDS'].agg(['mean', pct_1_3k, pct_10_30k]).head()
```

[Go to Contents :arrow_heading_up:](https://github.com/bhishanpdl/Tutorial-pandas#contents) 
# One column has list of lists
```python
df = pd.DataFrame({'team': ['A', 'B', 'C'],
                  'data': [['x',1,30], ['y',2,31], ['z',3,32]]})
(pd.DataFrame(df.data.tolist())
 .rename({0: 'name',1:'year',2:'age'},axis=1)
 .join(df.team))
```


# pandas and json to get currency conversion
```python
import numpy as np
import pandas as pd
import seaborn as sns

import json
from urllib.request import urlopen

tips = sns.load_dataset('tips')
grouped = tips.groupby(['sex', 'smoker'])

conv = 'USD_EUR'
url = f'http://free.currencyconverterapi.com/api/v5/convert?q={conv}&compact=y'

# DO NOT USE: converter = json.loads(urlopen(url).read())
with urlopen(url) as u:
    raw = u.read()
    converter = json.loads(raw)

print(converter) # {'USD_EUR': {'val': 0.87945}}
g = grouped['total_bill', 'tip'].mean().apply(lambda x: x*converter[conv]['val'])
print(g)
```


# Unnesting column with list
#==============================================================================
```python
df=pd.DataFrame({'A':[1,2],'B':[[1,2],[1,2]]})
def unnest(df):
    vals = np.array(df.B.values.tolist())
    a = np.repeat(df.A, vals.shape[1])
    return pd.DataFrame(np.column_stack((a, vals.ravel())), columns=df.columns)
## testing 
unnest(df)
```


# Mixed examples
#==============================================================================
```python
## item totals
df['item_totals'] = df.groupby('names')['values'].transform('sum')

## three columns all greater than 3 respective thresholds
df[df.gt(thr).all(axis=1)] # eg. for thr = [1,2,2], row [8,8,3] is selected.

## choose rows only if a has two or more repeated values
note: transform gives matrix of one less dimension after groupby, all row values are same for b and c
df[df.groupby("a").transform(len)["b"] >= 2]

## 2-digit year to 4-digit year
yr = df['year']
df['year'] = np.where(yr <= 20, 2000 + yr, 1900 + yr)

## change all df values greater to 1
df = df.clip_upper(1)
df = pd.DataFrame(np.clip(df.values, a_min=0, a_max=1),index=df.index,columns=df.columns)

## swapping values based on condition
df['a'], df['b'] = np.where(df['a'] > df['b'] , [df['b'], df['a']], [df['a'], df['b']])

## assign values
df.loc[df['c0'].isin(group),'c1'] = 'good' # group = ['a','b'], if c0 has 'a' c1 value will be changed.
df.loc[ (df['A']=='blue') & (df['C']=='square'), 'D'] = 'Yes'

## Read json
s = pd.read_json('countries.json',typ='series')
df = pd.DataFrame(s)

## drop duplicated values in one column and set its value to 'unknown' in another column
df.loc[df['col1'].duplicated(keep = False), 'col2'] = 'unknown'
df = df.drop_duplicates()

## train test split
train=df.sample(frac=0.8,random_state=200)
test=df.drop(train.index)

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2)

# copy data not view
msk = np.random.rand(len(df)) < 0.8
train, test = df[msk].copy(deep = True), df[~msk].copy(deep = True)
```

# New column based on element belongs to which list
https://stackoverflow.com/questions/55165380/reading-columns-in-pandas-with-lists-to-create-new-categorical-columns
```python
df = pd.DataFrame()
df['col_1'] =  df['col_1'] =  ['Spiderman', 'Abe Lincoln', 'Superman', 'Ghandi', 
                'Jane Austin', 'Robert de Niro', 'Elon Musk', 'George Bush',
                'Bill Gates', 'Barak Obama', 'Anne Frank']

l1 = [ 'Abe Lincoln', 'George Bush', 'Barak Obama']
l2 = ['Spiderman', 'Superman']
l3 = ['AnneFrank', 'Ghandi']
d = {'l1': l1, 'l2': l2,'l3': l3} 

# Required:
             col_1  new
0        Spiderman   l2
1      Abe Lincoln   l1
2         Superman   l2
3           Ghandi   l3
4      Jane Austin  NaN
5   Robert de Niro  NaN
6        Elon Musk  NaN
7      George Bush   l1
8       Bill Gates  NaN
9      Barak Obama   l1
10      Anne Frank  NaN

## solution1:
for k, v in d.items():
    df.loc[df['col_1'].isin(v), 'new'] = k


## solution2
map_d = {i: k for k, v in d.items() for i in v}
df['col_2'] = df['col_1'].map(map_d).str.extract(r'(\d+$)',expand=False)  # gives 1,2 instead of l1, l2

map_d = 
{'Abe Lincoln': 'l1',
 'AnneFrank': 'l3',
 'Barak Obama': 'l1',
 'George Bush': 'l1',
 'Ghandi': 'l3',
 'Spiderman': 'l2',
 'Superman': 'l2'}


## solution3
from functools import reduce
map_d = reduce(lambda a, b: dict(a, **b), [dict.fromkeys(y,x) for x , y in d.items()])
df['col_2']=df['col_1'].map(map_d)
```

# Normalized Frequency count of series elements
https://stackoverflow.com/questions/55197717/is-there-a-more-efficient-way-to-aggregate-a-dataset-and-calculate-frequency-in
```python
# req:                      0.25 0.5 0.25
df = pd.DataFrame({'value':[0, 1, 1, 2]})
df['value'].value_counts(normalize=True,sort=False)

## numpy method
u,c=np.unique(df.value,return_counts=True)
pd.Series(c/c.sum(),index=u) # gives series
pd.Series(c/c.sum(),index=u).to_frame('freq') # gives dataframe
pd.DataFrame(c/c.sum(), index=u,columns=['freq']) # dataframe

## using collections.Counter  (same as just value_counts)
from collections import Counter
df = pd.DataFrame({'value':[0, 1, 1, 2]})
pd.DataFrame.from_dict(Counter(df['value']), orient='index').reset_index().rename(columns={0: 'freq'})
```

# Comparing first two letters of columns of dataframes
https://stackoverflow.com/questions/54383285/conditionally-align-two-dataframes-in-order-to-derive-a-column-passed-in-as-a-co
```python
from pandas import DataFrame
import numpy as np

Names1 = {'First_name': ['Jon','Bill','Billing','Maria','Martha','Emma']}
df = DataFrame(Names1,columns=['First_name'])
print(df)

names2 = {'name': ['Jo', 'Bi', 'Ma']}
df_2 = DataFrame(names2,columns=['name'])
print(df_2)

Required:
First_name  like_flg
0   Jon     true
1   Bill    true
2   Billing true
3   Maria   true
4   Martha  true
5   Emma    Emma

Solution:
df['like_flg'] = np.where(df.First_name.str[:2].isin(df2.name), df.First_name.str[:2], df.First_name)
```


# Grouping by two columns and taking sum of same values
https://stackoverflow.com/questions/53385348/find-maximum-value-in-col-c-in-pandas-dataframe-while-group-by-both-col-a-and-b

```python
I have a pandas dataframe like this:

df = pd.DataFrame({"RT":[9,10,10,11,11,11,11],"Quality":[70,60,50,60,80,70,80],'Name' :['a','a','b','c','b','c','b'],'Similarity':[0.98,0.97,0.97,0.95,0.95,0.95,0.95]})

    RT  Quality Name    Similarity
0   9   70      a       0.98
1   10  60      a       0.97
2   10  50      b       0.97
3   11  60      c       0.95
4   11  80      b       0.95
5   11  70      c       0.95
6   11  80      b       0.95

the value in col Similarity is equal goubp by col RT

and I want to group col RT and find the maximum col Quality value group by col Name. 
for example: in col RT value 11,which have col Name value c and b ,
sum each of the col Quality values, then get c = 130, b =160, and sort the maximum  160,b then get

RT  Quality Name    Similarity
0   9   70  a       0.98
1   10  60  a       0.97
2   10  50  b       0.97
3   11  160 b       0.95
4   11  130 c       0.95


Solution:
df.groupby(['RT','Name']).agg({'Quality':'sum', 'Similarity':lambda x:x.unique()})
```

# Rename only few columns
https://stackoverflow.com/questions/53380310/how-to-add-suffix-to-column-names-except-some-columns
```python
keep_same = {'Id', 'Name'}
df.columns = ['{}{}'.format(c, '' if c in keep_same else '_old')
              for c in df.columns]
```

# Creating new column by comparing two dfs
https://stackoverflow.com/questions/53381590/quickest-way-to-map-lookup-table-to-pandas-column
```python
Col1    Col2
A       1
A       5
B       2
C       3
C       4

ColX    ColY
Mon     2  
Tues    3
Weds    5
Thurs   4
Fri     1


ColX    ColY    ColZ
Mon     2       B
Tues    3       C
Weds    5       A
Thurs   4       C
Fri     1       A

df2['Colz']=df2.ColY.map(df1.set_index('Col2').Col1)
```

# Create dict from one column and another column last sring value
https://stackoverflow.com/questions/53392451/get-string-slices-in-a-groupby-statement-python
```python
From:
          GG ID
0  L3S_0097A  Q
1  L3S_0097B  Q
2  L3S_0097C  Q
3  L3S_0097a  R
4  L3S_0097b  R
5  L3S_0097c  R

To:
{'Q': ['A', 'B', 'C'], 'R': ['a', 'b', 'c']}



df1 = pd.DataFrame({
         'ID': list('QQQRRR'),
         'GG':['L3S_0097A','L3S_0097B','L3S_0097C','L3S_0097a','L3S_0097b','L3S_0097c']

})

print (df1)

# method1
mm = df1['GG'].str[-1].groupby(df1['ID']).apply(list).to_dict()

# method2
mm = df1.assign(GG = df1['GG'].str[-1]).groupby('ID')['GG'].apply(list).to_dict()

# method3
from collections import defaultdict

mm = defaultdict(list)
#https://stackoverflow.com/a/10532492
for i, j in zip(df1.ID,df1.GG):
    mm[i].append(j[-1])

print (mm)
```

# efficient partial argsort using numpy and pandas
https://stackoverflow.com/questions/53651197/how-to-efficiently-partial-argsort-pandas-dataframe-across-columns

```python
From:
   p1  p2  p3  p4
0   0   9   1   4
1   0   2   3   4
2   1   3  10   7
3   1   5   3   1
4   2   3   7  10

To:
  Top1 Top2 Top3
0   p2   p4   p3
1   p4   p3   p2
2   p3   p4   p2
3   p2   p3   p1
4   p4   p3   p2

note:  for index 3, Top3 can be 'p1' or 'p4'.

Code:

import pandas as pd, numpy as np

df = pd.DataFrame({'p1': [0, 0, 1, 1, 2],
                   'p2': [9, 2, 3, 5, 3],
                   'p3': [1, 3, 10, 3, 7],
                   'p4': [4, 4, 7, 1, 10]})

def full_sort(df):
    return pd.DataFrame(df.columns[df.values.argsort(1)]).iloc[:, len(df.index): 0: -1]

def partial_sort(df):
    n = 3
    parts = np.argpartition(-df.values, n, axis=1)[:, :-1]
    args = (-df.values[np.arange(df.shape[0])[:, None], parts]).argsort(1)
    return pd.DataFrame(df.columns[parts[np.arange(df.shape[0])[:, None], args]])

df = pd.concat([df]*10**5)
%timeit full_sort(df)     # low ms per loop
%timeit partial_sort(df)  # high ms per loop

# method 2
def topN_perrow_colsindexed(df, N):
    # Extract array data
    a = df.values

    # Get top N indices per row with not necessarily sorted order
    idxtopNpart = np.argpartition(a,-N,axis=1)[:,-1:-N-1:-1]

    # Index into input data with those and use argsort to force sorted order
    sidx = np.take_along_axis(a,idxtopNpart,axis=1).argsort(1)
    idxtopN = np.take_along_axis(idxtopNpart,sidx[:,::-1],axis=1)    

    # Index into column values with those for final output
    c = df.columns.values
    return pd.DataFrame(c[idxtopN], columns=[['Top'+str(i+1) for i in range(N)]])
    
topN_perrow_colsindexed(df, N=3)
```

# get index of top 3 values for each columns
https://stackoverflow.com/questions/53649374/find-3-largest-values-in-every-column-in-data-frame-and-get-the-index-number-pyt

```python
From:
          A         B         C         D
0  0.037949  0.021150  0.127416  0.040137
1  0.025174  0.007935  0.011774  0.003491
2  0.022339  0.019022  0.024849  0.018062
3  0.017205  0.051902  0.033246  0.018605
4  0.044075  0.044006  0.065896  0.021264

To:
   A  B  C  D
0  4  3  0  0
1  0  4  4  4
2  1  0  3  3


Codes:
df.apply(lambda s: pd.Series(s.nlargest(3).index))

# example 2
pd.DataFrame([df[i].nlargest(3).index.tolist() for i in df.columns]).T

# example 3
pd.DataFrame(df.values.argsort(0), columns=df.columns)\
        .iloc[len(df.index): -4: -1]
```

# Find nearest value from other column to a given column
https://stackoverflow.com/questions/53969800/find-nearest-value-from-multiple-columns-and-add-to-a-new-column-in-python
```python
import pandas as pd
import numpy as np
data = {
    "index": [1, 2, 3, 4, 5],
    "A": [11, 17, 5, 9, 10],
    "B": [8, 6, 16, 17, 9],
    "C": [10, 17, 12, 13, 15],
    "target": [12, 13, 8, 6, 12]
}
df = pd.DataFrame.from_dict(data)
print(df)

# not having nans
idx = df.drop(['index', 'target'], 1).sub(df.target, axis=0).abs().idxmin(1)
df['result'] = df.lookup(df.index, idx)
df

# having nans
df2 = df.select_dtypes(include=[np.number])
idx = df2.drop(['index', 'target'], 1).sub(df2.target, axis=0).abs().idxmin(1)
df['result'] = df2.lookup(df2.index, idx.fillna('v1'))

# using numpy
idx = df.drop(['index', 'target'], 1).sub(df.target, axis=0).abs().idxmin(1)
# df['result'] = df.values[np.arange(len(df)), df.columns.get_indexer(idx)]
df['result'] = df.values[df.index, df.columns.get_indexer(idx)]

df
```

# create new column nearest in value to another column
https://stackoverflow.com/questions/30793178/alternatives-to-nested-numpy-where-for-multiconditional-pandas-operations
```python
Given: mydict = {'foo': [1, 2], 'bar': [2, 3]}

From column A and B get the column error:
    A    B   error
1 'foo' 1.2   0
2 'bar' 1.3  -0.7
3 'foo' 2.2   0.2

# nested np where
getmin = lambda x: mydict[x][0]
getmax = lambda x: mydict[x][1] 
df['error'] = np.where(df.B < df.A.map(getmin),
                       df.B - df.A.map(getmin),
                       np.where(df.B > df.A.map(getmax),
                                df.B - df.A.map(getmax),
                                0
                                )
                       )

# general method
df['min'] = df.A.apply(lambda x: min(mydict[x]))
df['max'] = df.A.apply(lambda x: max(mydict[x]))
df['error'] = 0.
df.loc[df.B.gt(df['max']), 'error'] = df.B - df['max']
df.loc[df.B.lt(df['min']), 'error'] = df.B - df['min']
df.drop(['min', 'max'], axis=1, inplace=True)

```

# triple double
https://stackoverflow.com/questions/54381858/create-a-triple-double-column-in-pandas-with-nba-stats
```python
stats = ['points', 'rebounds', 'assists', 'blocks', 'steals']
james_harden['trip_dub'] = (james_harden[stats] >= 10).sum(1) >= 3
```

# check two words in one column
https://stackoverflow.com/questions/54372893/pandas-filter-by-more-than-one-contains-for-not-one-cell-but-entire-column

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({'Data' : ['hello', 'world', 'hello', 'world', 'hello', 'world','hi'],
          'Data2': ['hello', 'hello', 'hello', 'hello', 'hello', 'hello','there']})
print(df)

# method 1
'hello' in df.Data.sum() and 'world' in df.Data.sum() # True

# method2
lst = ['hello','world']
any(x in df.Data.sum() for x in lst) # True

# method3
df.Data2.isin(['hello','world']).all() # False

# method4
import re 
bool(re.search(r'^(?=.*hello)(?=.*world)', df['Data'].sum()))
```

# Sort df by a value in json element of column
https://stackoverflow.com/questions/55587549/how-to-sort-pandas-dataframe-on-json-field/55658511#55658511
```python
   id     import_id              investor_id     loan_id      meta
   35736  unremit_loss_100312         Q05         0051765139  {u'total_paid': u'75', u'total_expense': u'75'}
   35737  unremit_loss_100313         Q06         0051765140  {u'total_paid': u'77', u'total_expense': u'78'}
   35739  unremit_loss_100314         Q06         0051765141  {u'total_paid': u'80', u'total_expense': u'65'}
   
# fast
if isinstance(df.at[0, 'meta'], str):
    df['meta'] = df['meta'].map(eval) # ast.literal_eval can also be used.
    
df.iloc[np.argsort( [ float(x.get('total_expense', '-1')) for x in df.meta.values ])]

# handles nans
import ast
if isinstance(df.at[0, 'meta'], str):
    df['meta'] = df['meta'].map(ast.literal_eval)
    
u = [  
  float(x.get('total_expense', '-1')) if isinstance(x, dict) else -1 
  for x in df['meta']
]
df.iloc[np.argsort(u)]

## another example
df['meta'] = df['meta'].apply(ast.literal_eval)
df = df.iloc[df['meta'].str['total_expense'].astype(int).argsort()]
df = df.iloc[df['meta'].str.get('total_expense').fillna(-1).astype(int).argsort()]


# Regex are always slow (use list comp as above)
df = pd.read_clipboard(r'\s\s+',engine='python')
df['total_expense'] = df.meta.str.extract(r"""u'total_expense': u'([0-9.]+)'""",expand=False)
df.sort_values('total_expense').drop('total_expense')

# Using apply is also slow
df['total_expense'] = df.meta.apply(eval).apply(
                        lambda x: x.get('total_expense', -1))
df.sort_values('total_expense')
```

Link: https://stackoverflow.com/questions/55441956/nice-way-to-make-the-following-wide-to-long-format-conversion-of-data-frame

From: date	type1_source1	type1_source2	type2_source1	type2_source2  
To:   date	source	type1	type2

```python
import pandas as pd
import numpy as np
import datetime as dt

n = 10
date = [dt.datetime.strftime(dt.datetime.now() + dt.timedelta(days=x), '%Y-%m-%d') for x in range(n)]
rn1 = np.random.randint(0, 50, n)
rn2 = np.random.randint(-50, 1, n)

data = {'date': date, 'type1_source1': rn1, 'type2_source1': rn1*100, 'type1_source2': rn2, 'type2_source2': rn2*100}
df = pd.DataFrame(data)
df
         date  type1_source1  type1_source2  type2_source1  type2_source2
0  2019-03-31             21            -24           2100          -2400
1  2019-04-01             21              0           2100              0
2  2019-04-02             28            -17           2800          -1700
3  2019-04-03              3            -43            300          -4300
4  2019-04-04             24            -39           2400          -3900
5  2019-04-05             36            -43           3600          -4300
6  2019-04-06             15             -4           1500           -400
7  2019-04-07             29            -14           2900          -1400
8  2019-04-08             14            -11           1400          -1100
9  2019-04-09             49            -42           4900          -4200
```

**Solution**  
```python
df = df.set_index('date')
df.columns = df.columns.str.split('_', expand=True)

df = (df.stack()
        .sort_index(level=1)
        .reset_index()
        .rename(columns={'level_1':'source'}))

df

          date   source  type1  type2
0   2019-03-31  source1     17   1700
1   2019-04-01  source1     36   3600
2   2019-04-02  source1     45   4500
```



