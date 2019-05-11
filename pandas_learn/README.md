Table of Contents
=================
   * [Pandas ABC-CAVEATS](#pandas-abc-caveats)
   * [Pandas Basics](#pandas-basics)
   * [pandas <strong>apply</strong>](#pandas-apply)
   * [pandas <strong>axis</strong>](#pandas-axis)
   * [pandas <strong>combine_first</strong>](#pandas-combine_first)
   * [pandas <strong>crosstab</strong>](#pandas-crosstab)
   * [pandas <strong>cut</strong>](#pandas-cut)
   * [pandas <strong>DataFrame</strong>](#pandas-dataframe)
   * [pandas <strong>datetime</strong>](#pandas-datetime)
   * [pandas Efficiency](#pandas-efficiency)
   * [pandas <strong>eval</strong>](#pandas-eval)
   * [pandas <strong>filter</strong>](#pandas-filter)
   * [pandas <strong>from_dict</strong>](#pandas-from_dict)
   * [pandas <strong>groupby</strong>](#pandas-groupby)
   * [Pandas <strong>iterrows</strong>](#pandas-iterrows)
   * [pandas <strong>pd.io.json.json_normalize</strong>](#pandas-pdiojsonjson_normalize)
   * [Pandas Manipulations](#pandas-manipulations)
   * [pandas <strong>map</strong>](#pandas-map)
   * [pandas <strong>melt</strong>](#pandas-melt)
   * [pandas <strong>MultiIndex</strong>](#pandas-multiindex)
   * [pandas <strong>pipe</strong>](#pandas-pipe)
   * [pandas <strong>pivot_table</strong>](#pandas-pivot_table)
   * [Pandas <strong>query</strong>](#pandas-query)
   * [pandas <strong>qcut</strong>](#pandas-qcut)
   * [pandas <strong>read_csv</strong>](#pandas-read_csv)
   * [pandas <strong>rename</strong>](#pandas-rename)
   * [pandas <strong>stack</strong>](#pandas-stack)
   * [pandas <strong>string</strong>](#pandas-string)
   * [pandas <strong>unstack</strong>](#pandas-unstack)
   * [pandas <strong>wide_to_long</strong>](#pandas-wide_to_long)
   * [pandas <strong>where</strong>](#pandas-where)


# Pandas ABC-CAVEATS
- Pandas operations are slow, but numpy operations are fast.
- There is no integer representation of NaN in numpy and Pandas unlike in R.
- Pandas ser.var() uses N-1 values to calculate unbiased variance, however, numpy uses N values and gives biased variance.
- Pandas is highly memory inefficient, it takes about 10 times RAM that of loaded data.
- To parallize pandas operation we can use modin.pandas or dask or use vaex or PySpark etc.
- Pandas agg supports list and dict but transform does not support it. `df.groupby('a')['b'].transform(['sum','count']` fails on pandas 0.24.2.
- Do not make datatype `np.float16`. `pd.Series([253]).astype(np.float16)` gives 252.

```python
** LOC and ILOC gives different results!!
# .loc is end inclusive, but .iloc and all numpy indices are not
## Reason: .loc is label based indexer.
df = pd.DataFrame([[0,0,0],[10,100,1000],[20,200,2000],[30,300,3000],[40,400,5000]])
df.loc[2] ==> row named 2
df[2] ==> column named 2 as series 
df[[2]] ==> column named 2 as dataframe
df[:2] == df.iloc[:2,] == df.iloc[:2,:] use iloc, df[:2] is confusing.
df.loc[1:3, :]    # gives dataframe with 3 rows with 10,20,30 as first column
df.iloc[1:3] ==> iloc is end exclusive but loc is inclusive
df.iloc[1:3, :]   # gives dataframe with 2 rows with 10,20 as first column
df.values[1:3, :] # gives numpy array with 2 rows
df1.loc['a':'f']  # gives all rows with row index between 'a' and 'f' even if 'z' is between, f is also included.
df1.loc[1:3]      # gives all rows with row index between 1 and 3 even if 5 is between.
                  # range indices just look at given unique index names, not incremental values.

# chained-assignment is BAD
df.loc[0, 'A'] = 11 # GOOD Here, the part 0,A of given dataframe is directly selected.
df['A'][0] = 111    # BAD Here, df['A'] dataframe is first extracted, pandas does not know whether it is copy or view.
                    # Extra note: Also, dfc.copy()['A'][0] is bad, same problem occurs.


# Libraries dependencies
pd.read_excel ==> needs: xlrd (to write excel we need openpyxl)
pd.read_hdf ==> needs: pytables (conda install pytables, dont use pip)
pd.read_parquet ==> needs: pyarrow (conda install -n viz -c conda-forge pyarrow)
NOTE: pd.read_parquet failed me with fastparquet libaray, use pyarrow.
      fastparquet needs python-snappy and even after installing it, it gave me errors.

# For aggregation function must return single value
# For apply it works with return series
df.groupby('A')['B'].agg(np.sqrt).head()   # ValueError: because np.sqrt(arr) gives array not a single value
df.groupby('A')['B'].apply(np.sqrt).head() # Works fine because apply works with series return

# dataframe has method applymap, but not series
df.applymap(myfunc)  # works fine
ser.applymap(myfunc) # fails

# pd.wide_to_long fails with datetime column, (we need to make it str)
https://github.com/bhishanpdl/Data_Cleaning/blob/master/data_cleaning_examples/pandas_Tap4_Fe_example.ipynb
df.columns = 'Date', 'Tap4.Fe', 'Tap4.Mn', 'Tap4.Fe', 'Tap5.Mn' # here Date is datetime category
pd.wide_to_long(df,'Tap','Date','numCompound').head() # ValueError: can only call with other PeriodIndex-ed objects
df.melt(id_vars='Date') # This does not fail and gives 3 columns: Date, variable, value

# Python caveats
## caveat: a = b = c
a = b = c = [1,2,3]
b.append(4)
print(a) gives [1,2,3,4]

## caveat: numpy dtype is fixed
a = np.array([1,2,3])
a[0] = 10.5
print(a) gives [10,2,3]  dtypes are not changed

## caveat: a+= b
a += b is not same as a = a + b
a = [1,2,3]
b = 'hello'
a = a + b # TypeError: can only concatenate list (not "str") to list
a += b
print(a) # [1, 2, 3, 'h', 'e', 'l', 'l', 'o']
```

# Pandas Basics
```python

## Remember
- axis=0:: default is along each columns. eg. np.mean(arr) is mean for each columns.
- axis=1:: .dropna('c2',1) .rename_axis(None,1) .set_index(['c0','c1'], axis=1) .filter(regex='e$', axis=1)
- axis=1:: .concat([df1,df2], axis=1)  .apply(func, axis=1)
- groupby:: grouped = df.groupby('id')['value']; for name, group in grouped: print(name,group)
- groupby:: df.groupby('c0')['c1'].apply(list)
- groupby:: df.groupby(['c0','c1'],as_index=False)['c2'].mean()
- groupby/add_suffix:: df.groupby('A')[['B']].sum().add_suffix('_sum').reset_index()
- groupby/apply:: acts on the whole dataframe of given group
- groupby/apply:: to get nlargest values, first sortby column, then groupby/tail
- groupby/agg:: acts on column and gives one value per group
- groupby/level:: level=0 groups by first index column, level=1 by 2nd index column NOT column (better level='index')
- groupby/transform:: acts on column and gives values for all rows
- gropby/agg/set_axis:: df.groupby('D')['A'].agg(['mean','count']).set_axis(['A_mean','n'],axis=1,inplace=False)
- groupby/manycolumns:: UNSTACK df.groupby(['a','b']).count()['c'].unstack()
- groupby/unstack:: is faster than pivot_tables(index,colums,values,aggfunc)
- groupby/nlargest:: df.groupby(['a'])['b'].nlargest(10).droplevel(-1).reset_index()
- groupby/multiindex:: [ i[0] + '_' + i[1] if i[1] else i[0] for i in df1.columns.ravel() ]
- get_dummies:: Pandas one hot encoding is called get_dummies
- rename:: we can use .rename(columns={'old': 'new'})  We can also use set_index(['c0','c1'], axis=1, inplace=False)
- reset_index:: we can rename before reset and also use reset_index([1,2]) to not reset first multiindex level
- reset_index:: reset_index(name='newname') is not in the documentation,so should not be used.
- stack:: stack => rename_axis => reset_index   (note: unstack(level=-1, fill_value=0))
- apply:: df[['year', 'month', 'day']].apply( lambda row: '{:4d}-{:02d}-{:02d}'.format(*row),axis=1)
- concat:: pd.concat([s1, df1], axis='columns', ignore_index=True) # keys = ['s1', 'df1'] gives hierarchial frame
- series :: s.values.tolist()  # if s has elements as list
- series.str.contains:: case=True and regex=False are defaults.
- string.split::  df['a'] == df['b'].str.split('.').str[0] can compare two string parts

## Efficiency tips
1. When you have json/dict as column elements, use ast.literal_eval (or simply eval)
   to get keys/values and do the manipulations such as sorting using list comprehension.
   Do not use regex to parse json as string and then do manipulations.

## Warnings
1. Be careful when using `np.where` about the NaNs. Instead use `pd.cut`. (Look below topic df.where)
2. Be careful while comparing two columns 'hello' and 'hello ' are not equal, remember to add .strip().

## Some Series methods
NO parenthesis:: is_unique, shape, size, dtype/dtypes ndim name
Paren: count() unique()
Math:: pct_change, ptp, quantile, divmod, diff, shift round
Math:: clip clip_lower clip_upper between ravel
Statistics:: mad kurt mode std rank
Cumulative:: expanding().sum() rolling(window=2.sum() cumsum cumprod cummax 
Rename:: reindex, reindex_axis, reindex_like, rename, rename_axis, set_axis, reset_index, swapaxes swaplevel
Nans:: isnull isna notnull notna hasnans ffill bfill fillna dropna 
Duplicated:: duplicated drop_duplicates
Columns:: rename set_axis sort_index reset_index add_suffix add_prefix
Arg:: argmax argmin argsort
Combining:: combine combine_first merge join
Deleting:: drop  truncate(before=indexname,after) mask filter
Useful:: nunique,   value_counts sort_values nlargest 
         sample(n=5 or frac=0.2, weights=None, random_state=None), nonzero, 
Esoteric:: last_valid_index, memory_usage()
Functions:: gropuby transform unstack  map

## Remember some methods
filter: df.filter(regex='e$', axis=1) # all columns ending with letter e
query: df.query('a==1 and  b==4 and not c == 8')
stack: pd.DataFrame({'c0' : [2,3],'c1' : ['A', 'B']}).stack() # gives only one series with multi-index
get_dummies: pd.get_dummies(pd.Series(list('abcaa')), drop_first=True) # only two columns of b and c with 0 and 1s.
cut: pd.cut(df['A'], bins=bins, labels=labels,include_lowest=True,right=False)

## crosstab (pivot, pivot_table, crosstab are slower than groupby/unstack)
elements of df.make will be index, and elements of df.num_doors will be new columns
crosstab: pd.crosstab(df.make, df.num_doors, margins=True, margins_name="Total",aggfunc='count')

## melt
** Columns B and C will be melted and disappeared from columns.
** New column 'variable' will have names only B and C.
** New column 'value' will have values from old columns B and C.
** Column A has only its elements.
melt: pd.melt(df, id_vars=['A'], value_vars=['B','C'],var_name='variable', value_name='value')

** columns A2019, B2020 have stubnames A and B. New column 'Year' will be created with values from stubnamed cols.
wide_to_long: pd.wide_to_long(df, stubnames=['A', 'B'], i=['Untouched', 'Columns', j='Year').reset_index()

** elements of columns A,B will be multiple-index, elements of column C will be column names,
** the columns will have elements values from values of D column.
pivot_table: pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'], aggfunc=np.sum)

## data slicing
df[0]  # column CALLED 0 if column 0 exists, else gives ERROR.
df[:1] # first ROW  similarly df[1:3] gives 2nd and 3rd row whatever be their index names.
df['col_0'] # df.col_0 works but df.0 does not work, so, using bracket notation is better.
df[['col1','col2']] # two columns
df.loc['a'] # row named 'a'
df.loc['a'] # row named 'a'
df.xs('a')  # row named 'a'
df.iloc[:, [1,3]]
df.loc['row_a':'row_p', 'col_a':'col_c']
df.iloc[:, np.r_[0,3,5:8]]
df.iloc[np.r_[0:5, -5:0]]
df.head().append(df.tail())
df['a': 'c'] # FAILS  (use: df.loc[:, 'a': 'c'])
df.loc[:, df.columns.isin(list('aed'))] # columns are automatically sorted
df.loc[:, df.columns.str.contains('a',case=False))]
df.loc[lambda x: x['a'] == 5.0] # df.pipe(lambda x: x['a'] == 5.0) also works for chaining.

# Insert column
loc = df.columns.get_loc("colName")
df.insert(loc+1, column, value) # This is inplace operation.

df[df.columns[2:4]]
df = df.sort_index()  # its good to sort the index, if they are not sorted

## only some indices
no_outliers = df.loc[~df.index.isin(idx_outliers)]

## data slicing more advanced
## NOTE: Use .loc instead of query for data <15k rows.
## https://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-query
df.loc[df['IQ'] > 90] # df.query('IQ > 90')
df.loc[df['country'].isin(['Nepal','Japan']) # df.query(' ["Nepal", "Japan"] in country')
df.loc[df['country'] == 'Nepal'] # df.query(' country == "Nepal" ')
df.loc[df['country'].str.contains('United')] # United States, United Kingdom etc.
df[(df.a == 1) & (df.b == 4) & ~(df.c == 8)]  # both ~ and not are same in query.
df.query(' (a==1) and  (b==4) and (not c == 8)') # both & and and are same in query.
df.query('a==1 and  b==4 and not c == 8')
df.query(" c0 < c1 < c2 and c3 != 500 ")
df[df.columns.difference(['exclude','me'])

# loc versus query
df = pd.DataFrame({'airport1': ['CMH','ORD','PHX','OAK','LAS','CMH','ORD'],
                   'airport2': ['ORD','PHX','OAK','LAS','CMH','DFW','ATL']})
df.query(' (airport1 == "CMH" and airport2 == "ORD") or\
           (airport1 == "ORD" and airport2 == "PHX")')
df.loc[ ((df.airport1 == 'CMH') & (df.airport2 == 'ORD')) |
        ((df.airport1 == 'ORD') & (df.airport2 == 'PHX')) ]

## select rows for column col1 when string length is 2
df.loc[df['col1'].str.len() == 2]

## use loc to assign values, df[mask] = something gives warning.
df[df['Affiliation'].str.contains('Harvard')]['Affiliation'] = 'Harvard University' # BAD
df.loc[ df['Affiliation'].str.contains('Harvard',case=False), 'Affiliation'] = 'Harvard University' # GOOD

## replace values
df[['a','b']] = df[['a','b']].replace([10], [0]) # change only column a b value 10 to 0, but keep other columns

## categorical values
df = pd.DataFrame({'a': [1.0, 0.0]})
df['a2'] = df.a.astype('category').cat.rename_categories(['M','F'])

## mapping new values
df.loc[df.sex == 'male', 'sex'] = 1
df.loc[df.sex == 'female', 'sex'] = 0
df['sex'] = df['sex'].replace(['male', 'female'], [1, 0]) # does not change nans
df['sex'] = df['sex'].map({'male': 1, 'female': 0}) 
df['sex'] = df['sex'].apply({'male': 1, 'female': 0}.get) 

## multiple conditions be aware of NaN values.
df['young_male'] = ((df.sex == 'male') & (df.age < 30)).map({True: 1, False:0}) # NOTE: True not 'True'
df.loc[1,'sex'] = np.nan # just for checking
df.loc[df.sex.isnull() | df.age.isnull(), 'young_male'] = np.nan

## iat/at are faster than iloc/loc
df.at['index_name']  # faster than df.loc['index_name']
df.iat[index_number] # df.iloc[index_number]
df.iat[0,2] # faster than df.iloc[0,2]

## faster sub-selection
row_num = df.index.get_loc('my_index_name')
col_num = df.columns.get_loc('my_column_name')
df.iat[row_num,col_num]

## delete some rows (note: inplace operations are always faster)
df.drop(df[df.colA == 'hello'].index, inplace=True)
df = df[df.ColA != 'hello'] # Gives copy warning

## select using dtypes or substrings
df.select_dtypes(include=['int']).head()
df.select_dtypes(include=['number']).head()
df.filter(like='facebook').head()  # all columns having substring facebook
df.filter(regex='\d').head() # column names with digits on them

## select using lookup
df.lookup([3.0,4.0,'hello'], ['c0','c1','c2'])

## dropping range of columns
df = df.drop(df.columns.to_series()["1960":"1999"], axis=1) # inclusive last value

## mapping values
s = pd.Series([1,2,3], index=['one', 'two', 'three'])
map1 = {1: 'A', 2: 'B', 3: 'C'}
s.map(map1) # this will change values from 1,2,3 to A,B,C

## number of notnull elements in a column
num_col2_notnull = pd.notnull(df['col2']).sum()

## Change Y to 1 and N to 0
df['yesno'] = df['yesno'].eq('Y').astype(int)

## Number of notnull numbers after 3rd column
df.iloc[:,3:].notnull().sum(1)

## filter examples
note: select num_items, num_sold but not price
df['mean_row'] = df.filter(like='num_').mean(axis=1,skipna=True)
df1 = df.filter(items=['col0','col1'])
df1 = df.filter(regex='e$', axis=1) # col_one, col_three but not col_two

## some statistics
df.describe(include=[np.number]).T
df.describe(include=[np.number],percentiles=[.01, .05, .10, .25, .5, .75, .9, .95, .99]).T
df.describe(include=[np.object, pd.Categorical]).T
df['col_5'].dropna().gt(120).mean()

## string
s.str.slice(0,3) # same as s.str[0:3] which gives first three letters.
pd.Series(['16 Jul 1950 - 15:00']).str.split('-').str[0] # don't forget last .str

df[['c0','c1']] = df['mycol'].str.split('_', expand=True) # eg. Nepal_Kathmandu gives two new columns
df['c0'], df['c1'] = zip(*df['mycol'].str.split('_')) # zip and star two columns separately

## some methods
df.nlargest(10, 'col_5').head() # gives all columns, but sorts by col_5
df.nlargest(100, 'imdb_score').nsmallest(5, 'budget')

df = df.set_index('A').sort_index() # always set index and sort them.
df.loc['myrow'] # faster
df[df['A'] == 'myrow'] # slower

### we can also create index from combining multiple columns
**For numbers, df.A.astype(str) + df.B.astype(str)
df.index = df['A'] + ', ' + college['B'] # NO .str plus works.
df = df.sort_index()
df.loc['firstname, lastname']

## Copy paste trick
df=pd.read_clipboard(sep='\s\s+') # will also read one space separated words.

## Series general methods
s = df['c0'] = pd.Series([1,2,3,4])
s.expanding().sum() # [ 1.,  3.,  6., 10.]
s.rolling(window=2,center=False).sum() # [nan,  3.,  5.,  7.]  1+2=3 and 2+3=5 and so on.
s.pct_change(periods=1, fill_method='pad', axis=0) # [nan, 1. , 0.5, 0.33333333]  (2-1)/1 = 1 and (3-2)/2 = .5 and so on.
```

# pandas **apply**
https://stackoverflow.com/questions/54432583/when-should-i-ever-want-to-use-pandas-apply-in-my-code
```python
import pandas as pd
import numpy as np
%load_ext memory_profiler

#------------------ df.max() is fast ----------------------------
df = pd.DataFrame({"A": [9, 4, 2, 1], "B": [12, 7, 5, 4]})
df.apply(np.sum) # very slow
df.sum() # fast

df.apply(lambda x: x.max() - x.min()) # very slow
df.max() - df.min() # fast

#---------------- use list comps --------------------------------
# find rows where name appears in title
df = pd.DataFrame({
    'Name': ['mickey', 'donald', 'minnie'],
    'Title': ['wonderland', "welcome to donald's castle", 'Minnie mouse clubhouse'],
    'Value': [20, 10, 86]})
df[df.apply(lambda x: x['Name'].lower() in x['Title'].lower(), axis=1)] # easy but slow
df[[n.lower() in t.lower() for n, t in zip(df['Name'], df['Title'])]]   # fast


#---------------- listcomp is faster than apply --------------------------------
df = pd.DataFrame({'col1' : ['1', '2;3;5', '3', '4;2', '5;1'],
                   'col2' : ['NaN', 'NaN', 'foo', 'bar', 'NaN']})
dict1 = {1:'aaa', 2:'bbb', 3:'foo', 4:'bar', 5:'ccc'}
df['col3'] = [';'.join([dict1[int(y)] for y in x.split(';')]) for x in df.col1]
df['col3'] = df.col1.str.split(';').apply(lambda x: ';'.join([ dict1[int(i)] for i in x]))

#-----------------df from series with lists ---------------------
s = pd.Series([[1, 2]] * 3)
s.apply(pd.Series) # very slow
pd.DataFrame(s.tolist()) # fast

#------------------- groupby+groupby vs groupby+apply -----------
# both apply and groupby are slow, so we may use apply
# find lagged cumsum
# ['a', 'a', 'b', 'c',  'c',  'c', 'd', 'd', 'e',  'e'] # A
# [12,  7,    5,   4,    5,   4,    3,   2,  1,    10]  # B
# [12,  19,   5,   4,    9,   13,   3,   5,  1,    11]  # cumsum
# [nan, 12.,  nan, nan,  4.,  9.,   nan, 3., nan,  1.]  # lagged cumsum
df = pd.DataFrame({"A": list('aabcccddee'), "B": [12, 7, 5, 4, 5, 4, 3, 2, 1, 10]})
df.groupby('A')['B'].cumsum().groupby(df['A']).shift()
df.groupby('A')['B'].apply(lambda x: x.cumsum().shift())  # here we may use apply

#------------------- dtype conversion ----------------------------
** df['col'].apply(str) may slightly outperform df['col'].astype(str).
** df.apply(pd.to_datetime) working on strings doesn't scale well with rows versus a regular for loop.

#------------------- apply is faster with many columns -----------
np.random.seed(0)
df = pd.DataFrame(np.random.random((10**7, 3)))     # Scenario_1, many rows
df = pd.DataFrame(np.random.random((10**4, 10**3))) # Scenario_2, many columns

                                               # Scenario_1  | Scenario_2
%timeit df.sum()                               # 800 ms      | 109 ms
%timeit df.apply(pd.Series.sum)                # 568 ms      | 325 ms

%timeit df.max() - df.min()                    # 1.63 s      | 314 ms
%timeit df.apply(lambda x: x.max() - x.min())  # 838 ms      | 473 ms

%timeit df.mean()                              # 108 ms      | 94.4 ms
%timeit df.apply(pd.Series.mean)               # 276 ms      | 233 ms



#------------------- FOR LOOP is faster than Apply !!! -----------
df = pd.DataFrame(
         pd.date_range('2018-12-31','2019-01-31', freq='2D').date.astype(str).reshape(-1, 2), 
         columns=['date1', 'date2'])

%timeit df.apply(pd.to_datetime, errors='coerce') # slowest
%timeit pd.to_datetime(df.stack(), errors='coerce').unstack() # second slowest
%timeit pd.concat([pd.to_datetime(df[c], errors='coerce') for c in df], axis=1) # second fastest
%timeit for c in df.columns: df[c] = pd.to_datetime(df[c], errors='coerce') # fastest


#----------------------------------------------------------------
def complex_computation(a):
    # Pretend that there is no way to vectorize this operation.
    return a[0]-a[1], a[0]+a[1], a[0]*a[1]

def func(row):
    v1, v2, v3 = complex_computation(row.values)
    return pd.Series({'NewColumn1': v1,
                      'NewColumn2': v2,
                      'NewColumn3': v3})

def run_apply(df):
    df_result = df.apply(func, axis=1)
    return df_result

def run_loopy(df):
    v1s, v2s, v3s = [], [], []
    for _, row in df.iterrows():
        v1, v2, v3 = complex_computation(row.values)
        v1s.append(v1)
        v2s.append(v2)
        v3s.append(v3)
    df_result = pd.DataFrame({'NewColumn1': v1s,
                              'NewColumn2': v2s,
                              'NewColumn3': v3s})
    return df_result

def make_dataset(N):
    np.random.seed(0)
    df = pd.DataFrame({
            'a': np.random.randint(0, 100, N),
            'b': np.random.randint(0, 100, N)
         })
    return df

def test():
    from pandas.util.testing import assert_frame_equal
    df = make_dataset(100)
    df_res1 = run_loopy(df)
    df_res2 = run_apply(df)
    assert_frame_equal(df_res1, df_res2)
    print 'OK'

# Testing
df = make_dataset(1000000)
test() # OK
%memit run_loopy(df)  # peak memory: 321.32 MiB, increment: 148.74 MiB
%memit run_apply(df)  # peak memory: 3085.00 MiB, increment: 2833.09 MiB  (10 times more memory)
%timeit run_loopy(df) # 1 loop, best of 3: 41.2 s per loop
%timeit run_apply(df) # 1 loop, best of 3: 4min 12s per loop

#  (apply is too slow!)
df = pd.DataFrame({'A': list('abc')*1000000, 'B': [10, 20,200]*1000000,
                  'C': [0.1,0.2,0.3]*1000000})

# fastest (100ms)
for c in df.select_dtypes(include = [np.number]).columns:
    df[c] = np.log10(df[c].values)

# 3 times slower 300ms
log10_df = pd.concat([df.select_dtypes(exclude=np.number),
                      df.select_dtypes(include=np.number).apply(np.log10)],
                      axis=1)
# 6 times slower
log10_df = df.apply(lambda x: np.log10(x) if np.issubdtype(x.dtype, np.number) else x)

## apply is slow, however, we can we it for aggregations
df = pd.DataFrame({"User": ["a", "b", "b", "c", "b", "a", "c"],
                  "Amount": [10.0, 5.0, 8.0, 10.5, 7.5, 8.0, 9],
                  'Score': [9, 1, 8, 7, 7, 6, 9]})
def my_agg(x):
    mydict = {
        'Amount_mean': x['Amount'].mean(),
        'Amount_std':  x['Amount'].std(),
        'Amount_range': x['Amount'].max() - x['Amount'].min(),
        'Score_Max':  x['Score'].max(),
        'Score_Sum': x['Score'].sum(),
        'Amount_Score_Sum': (x['Amount'] * x['Score']).sum()}

    return pd.Series(mydict, list(mydict.keys()))

df.groupby('User').apply(my_agg) # has columns 'Amount_mean', 'Amount_std', ...

# apply with multiple arguments
np.random.seed(100)
s = pd.Series(np.random.randint(0,10, 10))

def two_args(x, low, high):
    return x+low+high

s.apply(two_args, args=(3,6))
s.apply(two_args, low=3, high=6)
s.apply(two_args, args=(3,), high=6)

#------------------ use list-comp not apply ---------------
import unidecode  # we need to install this, pip install unidecode

## using pandas apply is slow
s = pd.Series(['mañana','Ceñía'])
s.apply(unidecode.unidecode) # manana, Cenia

## using list-comp is fast
decoded = [unidecode.unidecode(x) for x in s]
decoded = list(map(unidecode.unidecode, s))

## dataframe
df = pd.DataFrame()
df['s'] = s
df['s2'] = list(map(unidecode.unidecode, df['s'])) # faster
df['s3'] = df['s'].apply(unidecode.unidecode) # slower

#---------------------------------------------------------
df = pd.DataFrame({'item': ['apple','banana'],
                  'price': [10,20]})

# NEVER use this method, this is just for illustration
# better use-case of apply: https://programminghistorian.org/en/lessons/visualizing-with-bokeh
# latitude, longitude 'epsg:4326' gps coordinate to mercaptor 'epsg:3857' coordinate
# note: apply acts on rows and takes the whole dataframe, so its very slow.
df['a2'], df['a3'] = zip(*df.apply(lambda x: myfunc(x['price']), axis=1))

```

# pandas **axis**
```python
# rename_axis
new_df = df.groupby(['color', 'car', 'country']).value.mean().unstack().reset_index()
new_df.columns.name = None
==> new_df = df.groupby(['color', 'car', 'country']).value.mean().unstack().reset_index().rename_axis(None,1)

gdp = gdp.groupby('year').apply(lambda x: x.nlargest(10,columns='value'))
gdp.index.names = ['','']
==> gdp.groupby('year').apply(lambda x: x.nlargest(10,'value')).rename_axis([None,None],0)

## Reindex with extra indices
df = pd.DataFrame({'col': list('abbbaa')})
lst = list('abc')

df.col.value_counts().reindex(lst,fill_value=0)
pd.Categorical(df.col,lst).value_counts()
```

# pandas **combine_first**
```python
df = pd.DataFrame({'addr':  ['a',   'b',     np.nan],
                   'addr2': ['aa',   np.nan, 'c'],
                   'addr3': [np.nan,'bb',     np.nan]})

# oldest not nan values
from functools import reduce
dfs = [df.addr, df.addr2, df.addr3]
df['oldest'] = reduce(lambda l,r: l.combine_first(r), dfs)

# latest not nans values
df['latest'] = df.iloc[:,:3].ffill(axis=1,inplace=False).iloc[:, -1]
print(df)
  addr addr2 addr3 oldest latest
0    a    aa   NaN      a     aa
1    b   NaN    bb      b     bb
2  NaN     c   NaN      c      c

#-------------------------------------------
# combine example
df1 = pd.DataFrame({'A': [0, 0], 'B': [4, 4]})
df2 = pd.DataFrame({'A': [1, 1], 'B': [3, 3]})
take_smaller = lambda s1, s2: s1 if s1.sum() < s2.sum() else s2
df1.combine(df2, take_smaller).pipe(print)

   A  B
0  0  3  # 0+0 < 1+1
1  0  3  # 3+3 < 4+4
```


# pandas **crosstab**
https://pbpython.com/pandas-crosstab.html
```python
** pivot, pivot_table and crosstab are slower than groupby/unstack
# automobile example
pd.crosstab(df.make, df.body_style, normalize=True)
pd.crosstab(df.make, df.body_style, normalize='columns')
pd.crosstab(df.make, df.body_style, normalize='index')
pd.crosstab(df.make, df.num_doors, margins=True, margins_name="Total")
pd.crosstab(df.make, df.body_style, values=df.curb_weight, aggfunc='mean').round(0) # aggfunc requires values

# example 2
df = pd.DataFrame({'topic': list('AAABBB'),
                  'type': ['car','bike','car'] + ['bike']*3})
pd.crosstab(df['type'], df['topic'])
''' this gives multi-index dataframe with type as rows and topic as columns.
topic  A  B
type
bike   1  3
car    2  0
'''
# .to_dict() gives {'A': {'bike': 1, 'car': 2}, 'B': {'bike': 3, 'car': 0}}
```

# pandas **cut**
```python
# Note: pd.cut creates categorical dtype
df = pd.DataFrame({'A': [-np.nan,0,40,50,70,80,81,np.nan]})

bins = [0, 40, 60, 80, np.inf] #[0,40) is category A,  80 and 81 are D.
labels=['A','B','C','D']

df['A_cat'] = pd.cut(df.A, bins=bins, labels=labels,
                      include_lowest=True,right=False
                     ).values.add_categories('missing').fillna('missing')

** using numpy where
df['A_cat2'] = pd.cut(df.A, bins=bins, labels=labels,include_lowest=True,right=False)
df['A_cat2'] = np.where(df.A_cat.isnull(),'missing',df.A_cat)

# plotting example using pd.cut and groupby
df = pd.DataFrame({'A': [40,50,70,80],'B': [400,500,700,800]})
bins = [0, 40, 60, 80, np.inf]
labels=['Fail', 'Third', 'Second', 'First']
cuts = pd.cut(df['A'], bins=bins, labels=labels,include_lowest=True,right=False)
df.groupby(cuts)['B'].count().plot.bar()
```

# pandas **DataFrame**
```python
# Create df from numpy record arrays
import statsmodels as sm
sm_data = sm.datasets.longley.load() # we may need to use load(as_pandas=False)
rec = sm_data.data
df = pd.DataFrame.from_records(rec)

## another example
d = [{'a': 1, 'b': 2}, {'a': 2, 'b': 3}]
df = pd.DataFrame.from_records(d,index='a') # only one column b, a is index.
df = pd.DataFrame(d).set_index('a') # gives same result
```

# pandas **datetime**
```python
a = Sun b = Sep
Ymd 2018 09 30
HMS 18 59 58
Ip 06 pm

# simple example
s = pd.Series(['Jun 1 2005  1:33PM']) # Note: datetime.datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')
s = pd.to_datetime(s,format='%b %d %Y %I:%M%p') # Always give the format when dealing with datetime

#----------------------------------------------------------------------------------
# 100 to time 1:00 o'colock
df = pd.DataFrame({'a': [0,5,10,100,105,2000,2355]})
df['a_date'] = pd.to_datetime(df.a.astype(str).str.rjust(4,'0'),format='%H%M').dt.time
df['a_date2'] = pd.to_datetime(df['a'].astype(str).str.zfill(4), format = '%H%M').dt.time

#----------------------------------------------------------------------------------
# create new month
df = pd.DataFrame({'year': [2010,2011], 'month': [2,3], 'day': ['d1','d2']})

## best method ************************
df['day'] = df['day'].str[1:]
df['date'] = pd.to_datetime(df[['year','month','day']])


## another method *********************
df['day'] = df['date'].str[1:].astype(int)
df['date'] = df[['year', 'month', 'day']].apply(
    lambda row: '{:4d}-{:02d}-{:02d}'.format(*row),
    axis=1)
df.head()

## another method *********************
dfx['date'] = df['year'].apply(lambda x: '{:4d}'.format(x)) + '-' +\
              df['month'].apply(lambda x: '{:02d}'.format(x)) + '-' +\
              df['day'].str[1:].astype(int).apply(lambda x: '{:02d}'.format(x))

#----------------------------------------------------------------------------------
df = pd.DataFrame({'yr': ['1990-01-01','1990-01-02'],'hrmn': [1540.0, np.nan] })
df['yr'] = pd.to_datetime(df['yr']) # lets assume yr is already datetime.
df['date'] = compute_date_timestamp(df,'yr','hrmn')
df

# using function (works for nans)
** aliter: hours = df[hr_min] // 100  also works for nans
**         minutes = df.hr_min % 100
def compute_date_timestamp(df,year,hr_min):
    '''
    column year   = 1990-01-01  dtype = datetime
    column hr_min = 1540.       dtype = float
    '''
    hours, minutes = np.divmod(df[hr_min].values, 100)
    return df[year] + pd.to_timedelta(hours, unit='h') + pd.to_timedelta(minutes, unit='m')

**does not work for nans
df['date'] = pd.to_datetime([str(d) + ' ' + str(h) + ':'+str(m)
                               for d,h,m in zip(df.yr.dt.date.values,
                               *np.divmod(df.hrmn.values.astype(int),100))])

** does not work for nans
df['date2'] = pd.to_datetime(df['yr'].astype(str) + ' ' +
                             df['hrmn'].astype(str).str.slice(0,2) + ':' +
                             df['hrmn'].astype(str).str.slice(2,4))
```

# Pandas Efficiency
```python
# Tips
- Operations at/iat are faster than loc/iloc.
- When using pd.to_datetime, always use format option.
- When melting a dataframe, rename columns first, then melt.
- Numpy operations are faster than pandas operations. (eg. df['c0'].values faster than df['c0'])
- Regex operations are slow, try alternatives, e.g. new column using loc.
- When you see categorical data, always make the column dtype categorical.
- When dealing with timeseries, make the index datetime and sort the index.
- Use loc operation than apply operation. e.g. (df.loc[df['B'] >= 1000, 'B'] -= 1000, make small large values)
- Groupby/transform lambda are slower than two separate column operations.


#------------------------------------------------------------------------
# Pandas duplicated is faster than drop_duplicates
def duplicated(df): # fastest
    return df[~   df["A"].duplicated(keep="first")  ].reset_index(drop=True)

def drop_duplicates(df): # second
    return df.drop_duplicates(subset="A", keep="first").reset_index(drop=True)

def group_by_drop(df): # last
    return df.groupby(df["A"], as_index=False, sort=False).first()

# Pandas apply is slower than transformed
def f_apply(df):
    return df.groupby('name')['value2'].apply(lambda x: (x-x.mean())/x.std())

def f_unwrap(df):
    g = df.groupby('name')['value2']
    v = df['value2']
    return (v-g.transform(np.mean))/g.transform(np.std)

# map uses very less memory than apply
df = pd.DataFrame({'A': [1, 1, 2,2,1, 5]})
df['B'] = df.apply(lambda row: 1 if row['A'] == 1 else 0, axis=1)
df['C'] = df['A'].map({1:1, 2:0}).fillna(value=0).astype(int)


# Numpy is faster than pandas
pandas: ((df['col1'] * df['col2']).sum()) # slower
numpy: (np.sum((df['col1'].values * df['col2'].values)))  # fastest
numpy: (np.nansum((df['col1'].values * df['col2'].values))) # use this if you have nans

# drop_duplicates is faster than groupby
# 1.15 ms
df.drop_duplicates('Group',keep='last').\
           assign(Flag=lambda x : x['string'].str.contains('Search',case=False))

# groupby 1.60 ms
df.groupby("Group")["string"] \
  .apply(lambda x: int("search" in x.values[-1].lower())) \
  .reset_index()

#------------------------------------------------------------------------
# using numba
import numba
@numba.vectorize
def double_every_value_withnumba(x):
    return x*2

%timeit df['col1_doubled'] = df.a*2  # 233 us
%timeit df['col1_doubled'] = double_every_value_withnumba(df.a.values) # 145 us

#------------------------------------------------------------------------
# multiprocessing is sometimes slower
import numpy as np
import pandas as pd
import multiprocessing

np.random.seed(100)
s = pd.Series(np.random.randint(0,10, 10))

def myfunc(x):
    return x+100

def parallelize(data, func):
    ncores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(ncores)

    data_split = np.array_split(data, ncores)

    data = np.concatenate(pool.map(myfunc,data_split))
    data = pd.Series(data)
    pool.close()
    pool.join()
    return data

result = parallelize(s.values, myfunc)

print(s.values)
print(result.values)
[8 8 3 7 7 0 4 2 5 2]
[108 108 103 107 107 100 104 102 105 102]

s = pd.Series(np.random.randint(0,10, 1000_000))
%timeit parallelize(s.values, myfunc)
1 loop, best of 3: 689 ms per loop

%timeit s.values + 100
10 loops, best of 3: 37 ms per loop  # multiprocessing is slow sometimes

#------------------------------------------------------------------------
# multiprocessing with multiple args
import numpy as np
import pandas as pd
import multiprocessing
import numba

np.random.seed(100)
s = pd.Series(np.random.randint(0,10, 10))

@numba.vectorize
def two_args(x, low,high):
    return x+low+high

def multi_run_wrapper(args):
    return two_args(*args)

def parallelize(data, func,low,high):
    ncores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(ncores)

    data_split = np.array_split(data, ncores)
    data_split_lst = [(d,low,high) for d in data_split]

    data = np.concatenate(pool.map(multi_run_wrapper,data_split_lst))
    data = pd.Series(data)
    pool.close()
    pool.join()
    return data

result = parallelize(s.values, multi_run_wrapper,2,3)

print(s.values)
print(result.values)

#------------------------------------------------------------------------
# miscellaneous
x = x + 2y is slow
x += y; x +=y  if fast,  x+y+y = x+2y

#------------------------------------------------------------------------
# pipe is faster than apply
df.groupby('a').pipe(lambda grp: grp.size() / grp.size().sum()) # faster
df.groupby('a').apply(lambda grp: grp.count() / df.shape[0]) # slower

#------------------------------------------------------------------------
# groupby transform lambda is slow
df = pd.DataFrame({'year': [1990,1990,1992,1992,1992],
                  'value': [100,200,300,400,np.nan],
                  'rank': [2,1,2,1,3]})
df['value_relative']=df.value/df.groupby('year').value.transform('max') # fast
df['value_relative99']=df.groupby('year')['value'].transform(lambda x: x/x.max()) # slow

df['value_relative_rank2'] = df.value/df.year.map(df.loc[df['rank']==2].set_index('year')['value']) # fast
df['value_relative_rank2A'] =df.groupby('year')['value'].transform(lambda x: x/x.nlargest(2).iloc[-1] # slow

#------------------------------------------------------------------------
# groupby/apply nlargest is slower than sort_values/groupby/tail
df.sort_values('value').groupby('year').tail(2) # fast
df.sort_values('value').groupby('year',as_index=False).nth([-2,-1]) # flexible -1 is last, -2 is second last
df.groupby('year')['value'].apply(lambda x: x.nlargest(2)).reset_index() # slow

**also nlargest(1) is slower than idxmax
df.loc[df.groupby('id')['date'].idxmax()] # fast
df.groupby('id')['date'].nlargest(1) # slow
```

# pandas **eval**
https://pandas.pydata.org/pandas-docs/stable/enhancingperf.html#enhancingperf-eval
```python
## WARNING: If dataset is smaller than 15k rows, eval is several order slower than normal methods.
##          DO NOT USE eval and query if you have less than 10k rows.
## pandas eval uses very less memory
## pd.eval used numexpr module, and operations are fast and memory efficient.
pd.eval('df1 + df2 + df3 + df4') # eval is better than plain df1 + df2 + df3 + df4
pd.eval('df.A[0] + df2.iloc[1]')
df.eval('(A + B) / (C - 1)')
df.eval('A + @column_mean') # column_mean = np.mean(df.A)
df.eval('D = (A + B) / C', inplace=True) # creates new column
pd.eval('(df1 < 0.5) and (df2 < 0.5) or (df3 < df4)')
df.eval('tip * sin(30 * @pi/180)') # pi = np.pi  # sin/cos/log are supported but not pi.

# Note that there is no comma at the end of lines and line-continuation is \ character.
df.eval("""c = a + b
           d = a + b + c
           a = 1
           long_line_of_calculation = (column1 + column2) /\
                                      (column3 + column4)
           """, inplace=False)

## Use plain ol' python for dataframe with rows less than 15k
## df is seaborn tips data.
%timeit df.tip + df.tip + df.tip + df.tip # 195 µs
%timeit pd.eval('df.size + df.size + df.size + df.size', engine='python') # 555 µs (3 times slower)
%timeit pd.eval('df.tip + df.tip + df.tip + df.tip') # 1.08 ms (5.5 times slower)

## example2
cols = ['A-B/A+B','A-C/A+C','B-C/B+C']
x = pd.DataFrame([df.eval(col).values for col in cols], columns=cols)
df.assign(**x)
```

# pandas **filter**
```python
df.filter(items=None, like=None, regex=None, axis=None) # default axis=0 for series, 1 for dataframe
df = pd.DataFrame({'id': [1,2,3], 'num_1': [10,20,30], 'num_2': [20,30,40]})
df.filter(items=['id','num_1'])
df.filter(like='num') # all columns having "num"
df.filter(regex='_\d$')
df.filter(regex='^((?!num).)*$') # df.loc[:,~df.columns.str.contains('num')]

#============ groupby/filter======================
# groupby g and delete the whole group if any of the element has value 0
df = pd.DataFrame({'g': [1,1,1,2,2,3,3], 'val': [0,2,3,4,5,6,0]})
df[df.val !=0].groupby('g')['val'].mean()
df.groupby('g').filter(lambda x : all(x['val']!=0)) # exclude group if zero and keep all rows for others
df.groupby('g').filter(lambda grp: len(grp) > 2) # only group1 with 3 rows is given.

# same thing using index and loc
# TIPS: Always before using groupby first think is there loc method,
#       may be two or three lines more but  runs faster??
idx = df[df.val == 0].index
zero_g = df.loc[idx]['g']
df[~df.g.isin(zero_g)]

%timeit df.groupby('g').filter(lambda x : all(x['val']!=0))
100 loops, best of 3: 2.5 ms per loop # groupby is slow

%timeit df[~df.g.isin(df.loc[df[df.val == 0].index]['g'])]
1000 loops, best of 3: 1.48 ms per loop # index and loc is fast
```

# pandas **from_dict**
```python
## example 1
data = {'col_1': [3, 2, 1, 0], 'col_2': ['a', 'b', 'c', 'd']}
df1 = pd.DataFrame.from_dict(data) # same as pd.DataFrame(data) # default orient is columns
df2 = pd.DataFrame.from_dict(data,orient='index') # gives two rows

## example 2  create dataframe of counts
from collections import Counter
df1 = pd.DataFrame({'value':[0, 1, 1, 2]})
df2 = pd.DataFrame.from_dict(Counter(df1['value']), orient='index') # df1.value.value_counts().to_frame().sort_index()

## example 3
data = {'aaa': {'x1': 879,'x2': 861,'x3': 876,'x4': 873},
        'bbb': {'y1': 700,'y2': 801,'y3': 900}}
pd.DataFrame.from_dict(data, orient='index') # columns: x1 x2 x3 x4 y1 y2 y3, rows: aaa bbb , also we have nans

***make it better  (NOTE: this method is slower than stack)
df = pd.DataFrame.from_dict(data, orient='index').stack().reset_index()
df = df.rename(columns={'level_0': 'col1', 'level_1': 'col2', 0: 'col3'})

**using stack  (FASTEST)
df = pd.DataFrame(data)
df = df.stack().reset_index().set_axis(['col1', 'col2', 'col3'], axis=1,inplace=False)

**using melt (slowest, also do not hesitate to dropna after doing melt even though it is slow)
df = pd.DataFrame(data)
df = df.rename_axis(['col1']).reset_index().melt('col1', var_name='col2',value_name='col3').dropna()
```


# pandas **groupby**
```python
# groupby attributes
print([attr for attr in dir(pd.core.groupby.groupby.DataFrameGroupBy) if not attr.startswith('_') ])
print([attr for attr in dir(pd.core.groupby.groupby.DataFrameGroupBy) if attr[0].islower() ])

['agg', 'aggregate', 'all', 'any', 'apply', 'backfill', 'bfill', 'boxplot', 'corr', 'corrwith', 'count', 'cov', 'cumcount', 'cummax', 'cummin', 'cumprod', 'cumsum', 'describe', 'diff', 'dtypes', 'expanding', 'ffill', 'fillna', 'filter', 'first', 'get_group', 'groups', 'head', 'hist', 'idxmax', 'idxmin', 'indices', 'last', 'mad', 'max', 'mean', 'median', 'min', 'ndim', 'ngroup', 'ngroups', 'nth', 'nunique', 'ohlc', 'pad', 'pct_change', 'pipe', 'plot', 'prod', 'quantile', 'rank', 'resample', 'rolling', 'sem', 'shift', 'size', 'skew', 'std', 'sum', 'tail', 'take', 'transform', 'tshift', 'var']

## example
#******************************************************************************
df = pd.DataFrame({'A': [1, 1, 1, 2, 2],
                   'B': [1, 1, 2, 2, 1],
                   'C': [10, 20, 30, 40, 50],
                   'D': ['X', 'Y', 'X', 'Y', 'Y']})

    A  B   C  D    note: groupby('A') gives multi-index df with A being index, BCD being columns
0  1  1  10  X   think x as dataframe with index name A and columns BCD (we can use x['B'].mean()>3)
1  1  1  20  Y   **apply gives only two rows since there are two groups for A, 
2  1  2  30  X   **transform gives all rows with same values for one group, we can make new column of this.
#-----------------------------------------------------------------------------
3  2  2  40  Y     groupby('A') has two parts above and this one (REMEMBER THIS!!!)
4  2  1  50  Y

##!!! GROUPBY WARNING!!!
** If you want to do many columns operations after groupby, you cant do it before.
** find A/B sum for each persons in D column.
## good
df1 = df.groupby('D')['A','B'].sum().reset_index()
df1['ab'] = df1.A / df1.B

## wrong answer
df['ab'] = df.A / df.B
df.groupby('D')['ab'].sum() # gives two rows but wrong answer

## good aliter for data with >15k rows
df1 = df.groupby('D')[['A', 'B']].sum().eval('ab = A / B').reset_index() # gives two rows ie. X and Y

# sanity check
a = df[df.D=='X']['A'].sum()
b = df[df.D=='X']['B'].sum()
a,b,a/b

# mean, sum, size, count, std, var, describe, first, last, nth, min, max
# agg function: sem (standard error of mean of groups)
## count ignores nans, size reads nans, nunique gives unique
df.groupby(‘A’)[‘B’].count() # gives count of non-NaNs (use .size() to count NaNs)
df.groupby(‘A’)[‘B’].sum()   # gives series
df.groupby(‘A’)[[‘B’]].sum() # gives dataframe
df.groupby(‘A’, as_index=False)[‘B’].sum() # as_index ==> no need of .reset_index()
df.groupby(‘A’, as_index=False).agg({‘B’: ‘sum’}) # gives columns A and B
df.groupby(‘A’)[‘B’].agg(lambda x: (x - x.mean()) / x.std() ) # zscore
df.groupby(‘D’).get_group(‘X’)
df.groupby(‘A’).filter(lambda x: x > 1)
df.groupby(‘A’).describe()
df.groupby(‘A’).apply(lambda x: x *2)
df.groupby(‘A’).expanding().sum()
df.groupby('A')['B'].sum() # two rows with sum 6 and 9 (note cumsum() gives 5 rows)
df.groupby(['A'])['B'].nlargest(2).droplevel(-1).reset_index()
df.groupby('A')['B'].transform('sum') # 5 rows, transform keeps same dimension.
df.transform({'A': np.sum, 'B': [lambda x : x+1, 'sqrt']})
df['new'] = df.groupby('D', group_keys=False).apply(lambda x: x.A + x.C.sum()) # group_keys ==> no need of .values
df.groupby('A')['C'].rank(method='dense',ascending=False,na_option='bottom').astype(int) # 321 21
df.groupby('D')['A'].agg(['mean','count']).set_axis(['A_mean','n'],axis=1,inplace=False)

## multiple columns operations after groupby
#******************************************************************************
df['E'] = (df.groupby('D').apply(
    lambda grp: (grp.A + grp.B) / (grp.C.sum()) )
          ).values

## groupby group_keys=False example
#******************************************************************************
df = pd.DataFrame({'A':['A','A','B','B'],'B':[10,20,30,40],'C':[1,3,3,5]})
df = df.assign(D=df.groupby('A', group_keys=False).apply(lambda x: x.B - x.C.mean()))
# this gives D = 8.0 for first row, 10 - ((1+3)/2)


## groupby count but also count nans to 0 (unstack,fill0 then stack)
#******************************************************************************
df = df.groupby(['A','B'])['C'].count().unstack(fill_value=0).stack()

## multiple aggregation
df.groupby(['A','B','D'])['C'].agg([np.mean,np.median]).add_suffix('_D').reset_index()

## multiple aggregation use of dict of rename columns is deprecated
df.groupby('A')['B'].agg(['mean','count']).reset_index() # good
df.groupby('A')['B'].agg({'new1': 'mean', 'new2': 'count'}).reset_index() # deprecated
df.groupby('A')['B'].agg(['mean','count']).reset_index()\  # use this for renaming
  .rename(columns = {'mean': 'new1', 'count': 'new2'})

#******************************************************************************
## add prefix
df.groupby('A').mean().add_prefix('mean_') # gives mean_B and mean_C, D is ignored.
df.groupby('A').max().add_suffix('_max')   # note: prefix, suffix accept ONLY ONE string.


(df.groupby('A')
    .agg({'B': 'sum', 'C': 'min'})
    .rename(columns={'B': 'B_sum', 'C': 'C_min'})
    .reset_index()
)

#******************************************************************************
## using custom function, function name becomes column name
def max_min(x): return x.max() - x.min()
max_min.__name__ = 'max_minus_min'
df1 = df.groupby('D').agg({'B':['sum', 'max'], 'C': max_min})

#******************************************************************************
# transform does not support multiple-aggregation but agg supports

**transform multiple groupby
# fast and also easy way
df['B_sum'] = df.groupby(['D','A'])['B'].transform('sum')
df['B_count'] = df.groupby(['D','A'])['B'].transform('count')
df['B_mean'] = df.groupby(['D','A'])['B'].transform('mean')

**aliter
df.assign(
    B_sum = lambda dff: dff.groupby(['D','A'])['B'].transform('sum'),
    B_count = lambda dff: dff.groupby(['D','A'])['B'].transform('count'),
    B_mean = lambda dff: dff.groupby(['D','A'])['B'].transform('mean')
          )

## slow and complicated way
## We can replicate multiple-transform using join df[['group']] with agg
df1 = df.groupby('D').agg({'A': ['sum', 'count','mean']})
df1.columns = [i[0]+'_'+i[1] if i[1] else i[0] for i in df1.columns.ravel()]
df2 = df[['A']].join(df1,on='A')

# slow and complicated way
dfagg = df.groupby(['D','A'],as_index=False).agg({'B': ['sum', 'count','mean']})
dfagg.columns = [i[0]+'_'+i[1] if i[1] else i[0] for i in dfagg.columns.ravel()]
dftrans = df[['D','A','B']].merge(dfagg, on=['D','A'],how='left')


## Rank uses method='average' by defalut
#******************************************************************************
from scipy.stats import rankdata
from functools import partial
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


## agg with custom function (mean of only positive values)
#******************************************************************************
** Warning: using a dict to rename column is deprecated, but not using dict for multiple columns.
def mean_pos(x): return x[x>0].mean()
df = (df.groupby('D')['A'].agg(['count',mean_pos]
df = (df.groupby('D').agg( {'A': ['count',mean_pos]}) # not deprecated  but ['A'].agg({'new': 'count'}) deprecated.

**method 2
df['A2'] = df['A']
df.loc[df.A <= 0,'A'] = np.nan
df.groupby(['D'])[['A','A2']].mean().reset_index()

#******************************************************************************
### using reset to drop one level of multi-index
g = (df.groupby('A') # agg gives here two rows of columns -- C,B and mean,count min,max and index A
    .agg({'B': ['min', 'max'], 'C': ['mean','count']})
    .reset_index()) # now index A, becomes first column with column name A, and its second level column name is empty.

### using as_index false to drop one level of multi-index (same as .reset_index())
** This does not give deprecation warning.
g = (df.groupby('A',as_index=False)
    .agg({'B': ['min', 'max'], 'C': ['mean','count']}))

#### rename columns
### In above multiple aggregation examples, we get two levels of columns. (after reset or as_index = False)
### To make only one column name, we can use list comprehension.
### note: When we reset and make the index A, as column, it does not have second level column name.
g.columns = ['_'.join(x).strip() if x[1] else x[0] for x in g.columns.ravel()]

***Rename columns of multi-index
g = (df.groupby(['sex','time','smoker'])
     .agg({'tip': ['count','sum'],
           'total_bill': ['count','mean']})
     .reset_index()
     .pipe(lambda x: x.set_axis([f'{a}_{b}' if b == '' else f'{a}'
                                 for a,b in x.columns],
                                axis=1, inplace=False)))


## Groupby with custom function
#******************************************************************************
## example: getting list of all members
data = {'mother':["Penny", "Penny", "Anya", "Sam", "Sam", "Sam"],
        'child':["Violet", "Prudence", "Erika", "Jake", "Wolf", "Red"]}
df = pd.DataFrame(data)
df.groupby('mother').agg(','.join).reset_index() # gives mother and her children
**aliter: df.groupby('mother')['child'].apply(list).reset_index()

## example2
df = pd.DataFrame({'Name': list('ABCAB'),'Score': [20,40,80,70,90]})
def get_stats(group):
    return {'min': group.min(), 'max': group.max(), 'count': group.count(), 'mean': group.mean()}
bins = [0, 25, 50, 75, 100]
group_names = ['Low', 'Okay', 'Good', 'Great']
df['categories'] = pd.cut(df['Score'], bins, labels=group_names)
df['Score'].groupby(df['categories']).apply(get_stats).unstack()

## Groupby with Date column
#******************************************************************************
# get max value for a calendar-day of any year
df.groupby(df['date'].dt.strftime('%m-%d'))['value'].max()

## Groupby with pd.Grouper
#******************************************************************************
df.groupby([pd.Grouper(freq='1M',key='Date'),'Buyer']).sum()

## time series
#******************************************************************************
df = pd.DataFrame({'date': pd.date_range(start='2016-01-01',periods=4,freq='W'),
                   'group': [1, 1, 2, 2],
                   'val': [5, 6, 7, 8]}).set_index('date') # only 4 rows
df.groupby('group').resample('1D').ffill() # 16 rows
## groupby with categorical data
#******************************************************************************

pd.Series([1, 1, 1]).groupby(pd.Categorical(['a', 'a', 'a'],
                        categories=['a', 'b']), observed=False).count()

## pipe and apply
#******************************************************************************
(df.groupby(['A', 'B'])
    .pipe(lambda grp: grp.C.max()) # .apply() gives same result.
    .unstack().round(2))

## using functions
def subtract_and_divide(x, sub, divide=1):
    return (x - sub) / divide

df.iloc[:,:-1].apply(subtract_and_divide, args=(5,), divide=3)

## groupby pipe (pipe is encourased to be used)
f(g(h(df), arg1=1), arg2=2, arg3=3)
(df.pipe(h)
       .pipe(g, arg1=1)
       .pipe(f, arg2=2, arg3=3)
    )

## groupby level=0
df = pd.DataFrame( dict( c0 = list('abac'), c1=[10,100,1000,10000]),index = list('pqpr'))
df.groupby(level=0).mean()

example2
df = pd.DataFrame( dict(c0=list('abac'), c1=list('wxyz'), c2=[10,100,1000,10000], c3=[20,200,2000,20000]))
df1 = df.set_index(['c0','c1'])

df1.groupby(level=0).mean()
df1.groupby(level='c1').mean()

df1.groupby(level=1).mean()
df1.groupby(level='c1').mean()
```

# pandas **iterrows**
```python
# Perform (1 + col1) * col2_previous_value
df = pd.DataFrame({'col1' : [0.1, 0.2, 0.3, 0.4],
                   'col2' : [20.0, 0, 0, 0]})

for idx, row in df.iterrows():
    if idx > 0: # Skip first row
        df.loc[idx, 'col2'] = (1 + row['col1']) * df.loc[idx-1, 'col2']

# here, we only need first value of col2, all other values are created.
# this means col2 can not be multiplied, only first value is used.
# but we will cumprod col1 to compensate that.
#
# to do 1 + col1 we also need to shift and fillwith 1.
# shift(-1) makes last row Nan, and fillwith 1.
(( 1 + df['col1'].shift(-1).fillna(1) )).cumprod().shift().fillna(1) * df['col2'].iloc[0]
# to use only first value of col2, we do cumprod of col1 and shift   (first value = 20.0)

```


# pandas **pd.io.json.json_normalize**
```python
## json vs python
python: dict    list,tuple  str      int,float  True False None
json  : object  array       string   number     true false null

import json

data = {'people':[{'name': 'Scott', 'website': 'stackabuse.com',
                   'married': True,'age':30,'salary':100000.0,
                  "languages": ["English", "French"]}]}

print(json.dumps(data,indent=4,sort_keys=True))

## to write json, we need file object.
with open('myjson.json','w') as fo:
    json.dump(data,fo)

## Read json file in terminal
!cat myjson.json | python -m json.tool

## read json file (can read True/False/None)
with open('myjson.json') as fi:
    data = json.load(fi)

    for key,value in data['people'][0].items():
        print(key,'==> ',value )

## read json from string
sdata = str(data)
sdata2 = sdata.replace("'",'"') # json needs double quotes
sdata2 = sdata2.replace("None",'null').replace('True','true').replace('False','fasle')
j = json.loads(sdata2)

## pandas dataframe
df = pd.DataFrame(data['people'][0]) # since we have two languages, it gives two rows.

## read json to pandas dataframe from file
df = pd.read_json('myjson.json') # gives one row
with open('myjson.json', 'r') as fi:
    data = json.load(fi)
df = pd.DataFrame(data['people'][0]) # gives nice dataframe.

##=============== ijson is better than json====================
import ijson
data = {
  "earth": {
    "europe": [
      {"name": "Paris", "type": "city", "info": {'a': 1, 'b':2.0 }},
      {"name": "Thames", "type": "river", "info": {'a': 10, 'b': 20.0 }},
    ],
    "america": [
      {"name": "Texas", "type": "state", "info": {'a': 100, 'b': 200.0 }},
    ]
  }
}

with open('myjson.json','w') as fo:
    json.dump(data,fo)

f = open('myjson.json', 'rb') # File needs to be open to the end until print(city)
objects_europe = ijson.items(f, 'earth.europe.item')
cities = (o for o in objects_europe if o['type'] == 'city')
for city in cities:
    print(city)
##=========== ijson parser=================
data = {"AssetCount": 2,"Server": "xy","Assets": [{"Identifier": "21979c09fc4e6574"},{"Identifier": "e6235cce58ec8b9c"}]}

## write json to a file
with open('sample.json','w') as fo:
    json.dump(data,fo,indent=4)

## parse the json using ijson
f = open('sample.json', 'rb')
for idn in ijson.items(f, 'Assets.item.Identifier'):
    print(idn)

## parse the json using ijson
file_name="sample.json"
with open(file_name) as file:
    parser = ijson.parse(file)
    for prefix, event, value in parser:
        if prefix=="AssetCount":
            print (value)
        if prefix=="Server":
            print (value)
        if prefix=="Assets.item.Identifier":
            print (value)
## To see the list parsed by parser
ijson_lst = list(ijson.parse(open('sample.json')))

#============= pd.io.json.json_normalize example==================
data = [{'state': 'Florida',
              'shortname': 'FL',
              'info': {
                   'governor': 'Rick Scott'
               },
              'counties': [{'name': 'Dade', 'population': 12345},
                          {'name': 'Broward', 'population': 40000},
                          {'name': 'Palm Beach', 'population': 60000}]},
             {'state': 'Ohio',
              'shortname': 'OH',
              'info': {
                   'governor': 'John Kasich'
              },
              'counties': [{'name': 'Summit', 'population': 1234},
                           {'name': 'Cuyahoga', 'population': 1337}]
        }]

pd.io.json.json_normalize(data=data,record_path='counties', meta=['state', 'shortname', ['info', 'governor']])

# column having json or dict elements
column ==>  'meta': [ "{u'total_paid': u'75', u'total_expense': u'75.6'}"]
## make it dictionary
if isinstance(df.at[0, 'meta'], str):
    df['meta'] = df['meta'].map(eval)

## then extract the keys
df['total_expense'] = [float(x.get('total_expense', '-1')) for x in df['meta']]
```

# Pandas Manipulations
```python
# Two columns have similar string
   countryCode   Name   number      myprice prices
           DZ  name1  number1         US.p    nan
           DZ  name1  number1  AU.currency    45
           DZ  name1  number1  DZ.currency    55
           DZ  name1  number1         DZ.p    62
           DZ  name1  number1         AU.p     73
           DZ  name1  number1  US.currency    nan
           AU  name1  number1         US.p    nan
           AU  name1  number1  AU.currency    77
# select rows when country code is in myprice
cond = (df['countryCode'] == df['myprice'].str.split(".").str[0])
cond = df.countryCode.eq(df.myprice.str.split(".").str[0])
cond = df.apply(lambda x: x['countryCode'] in x['myprice'], axis=1)
cond = [cc in pr for cc,pr in zip(df.countryCode, df.myprice)]
cond = np.core.char.find(df['myprice'].values.astype(str),df['countryCode'].values.astype(str))!=-1
df[cond]
```

# pandas **map**
```python
## We can use both series and dict for mapping
df = pd.DataFrame({'Color': ['Red', 'Red', 'Blue'], 'Value': [100, 150, 50]})
df['counts'] = df['Color'].map(df.Color.value_counts()) # df.Color.value_counts().to_dict() not needed
df['counts'] = df.groupby('Color')['Color'].transform('count') # count excludes nans but 'size' and len includes nans.
df['counts'] = df.Color.groupby(df.Color).transform('count')
df['coutns'] = df.Color.map(Counter(df.Color)) # from collections import Counter


## mapping gender based on first letter
df = pd.DataFrame({'gender' : ['male', 'M.','Male','F','female','Fem','Fem.']}) # check: df.gender.unique()
df['gender'] = df['gender'].str[0].str.lower().map({'m' : 1, 'f' : 0})

## mapping age group
s = pd.Series([14,1524,2534,3544,65])
age_map = {14: '0-14',1524: '15-24',2534: '25-34',3544: '35-44', 4554: '45-54',5564: '55-64',65: '65+'}
s.map(age_map)

# using regex
s = s.astype(str).str.replace(r'14', r'0-14',regex=True)
                 .str.replace(r'65', r'65+',regex=True)
                 .str.replace(r'(\d\d)(\d\d)', r'\1-\2',regex=True))
```

# pandas **melt**
```python
df = pd.DataFrame({'State' : ['Texas', 'Arizona', 'Florida'],
          'Apple'  : [12, 9, 0],
          'Orange' : [10, 7, 14],
          'Banana' : [40, 12, 190]})

df.melt(id_vars=['State'],
        value_vars=['Apple','Orange','Banana'],
        var_name='Fruit',
        value_name='Weight')

## example 2 *********************
  Weight Category  M35 35-39  M40 40-44  M45 45-49  M50 50-54  M80 80+
0              56        137        130        125        115        102
1              62        152        145        137        127        112

(df.melt(id_vars='Weight Category', var_name='sex_age', value_name='Qual Total')
   .assign(Sex      = lambda x: x['sex_age'].str.extract('([MF])', expand=True))
   .assign(AgeGroup = lambda x: x['sex_age'].str.extract('(\d{2}[-+](?:\d{2})?)', expand=True))
   .drop('sex_age', axis=1)
   ).head(2)
```

# pandas **MultiIndex**
```python
##------------------------------
## removing multi-index column names
df.columns # this gives column names just use df.columns = [whatever]
df.columns = df.columns.droplevel()
df.columns=df.columns.get_level_values(1)

## create multi-index series
s = pd.Series([10, 20,30,40], index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]])
s.index.names = ['year','category']

aliter:
index = pd.MultiIndex.from_product([[1999, 2000], ['A', 'B']], names=['year', 'category'])
s = pd.Series([5.2, 5.1,3.7,6.1], index=index)


##------------------------------
## create multi-index dataframe
index_matrix = [['a','b'],['c','d'],['e','f']]
data_c0 = [10,20]
data_c1 = [100,200]
index = pd.MultiIndex.from_arrays(index_matrix, names=['index_c0', 'index_c1', 'index_c2'])
df = pd.DataFrame({'data_c0': data_c0,'data_c1': data_c1}, index=index)
print(df)
                            data_c0  data_c1
index_c0 index_c1 index_c2
a        c        e              10      100
b        d        f              20      200

****reset index****
print(df.reset_index())
  index_c0 index_c1 index_c2  data_c0  data_c1
0        a        c        e       10      100
1        b        d        f       20      200

##----------------------------------------------------------------
## create multi-index dataframe using groupby
df = pd.DataFrame({"User": ["a", "b", "b", "c", "b", "a", "c"],
                  "Amount": [10.0, 5.0, 8.0, 10.5, 7.5, 8.0, 9],
                  'Score': [9, 1, 8, 7, 7, 6, 9]})
x = df.groupby('User').agg({"Score": ["mean",  "std"], "Amount": "mean"}).reset_index(drop=True)
      Score              Amount
       mean       std      mean
0  7.500000  2.121320  9.000000
1  5.333333  3.785939  6.833333
2  8.000000  1.414214  9.750000

****remove multi-index****
x.columns = [ i[0] + '_' + i[1] if i[1] else i[0] for i in x.columns.ravel() ]
   Score_mean  Score_std  Amount_mean
0    7.500000   2.121320     9.000000
1    5.333333   3.785939     6.833333
2    8.000000   1.414214     9.750000

##----------------------------------------------------------------
## multi-index slicing
column_names  c0  c1  c2
index_name1 index_name2
Whse_A    2011             NaN           NaN    108.000000           NaN
          2012             NaN           NaN     70.685714           NaN
Whse_C    2011       17.909091           NaN           NaN           NaN
          2012       36.653374           NaN           NaN           NaN
          2013       29.292553           NaN           NaN           NaN

**select all Whse_A
df.loc[['Whse_A']]
df.loc[('Whse_A',2011):('Whse_A',2013)] # this also works
```

# pandas **pipe**
```python
df = sns.load_dataset('iris')
df.pipe( (sns.catplot,'data'), x='species',y='sepal_length',kind='bar') # it gives mean plot
df.groupby('species',as_index=True)['sepal_length'].mean().plot.bar()   # equivalent in pandas
df[['species','sepal_length']].groupby('species').count().plot.bar() # pandas, faster, but all blue
df[['species','sepal_length']].groupby('species').count().plot.bar(colors=[sns.color_palette('Set1')])  # can use ('Set1',3)

# notes: we can also use sns.barplot
df.pipe( (sns.countplot,'data'), x='species') # all are 50 counts (y='species' gives horizontal plot)

# pipe is faster than apply
df.groupby('a').pipe(lambda grp: grp.size() / grp.size().sum()) # faster
df.groupby('a').apply(lambda grp: grp.count() / df.shape[0]) # slower
```

# pandas **pivot_table**
```python
** pivot, pivot_table and crosstab are very similar and all are slower than groupby/unstack.
## pivot_table
df = pd.DataFrame({'country': ['usa','canada','usa','canada','mexico','usa'],
                   'color':   ['silver','brown','brown','black','silver','black'],
                   'car':     ['honda','honda','nissan','toyota','honda','toyota'],
                   'value': range(60,66)})

      car   color country  value
0   honda  silver     usa     60
1   honda   brown  canada     61
2  nissan   brown     usa     62
3  toyota   black  canada     63
4   honda  silver  mexico     64
5  toyota   black     usa     65

df.groupby(['color', 'car', 'country'])['value'].mean().unstack().reset_index().rename_axis(None, axis=1)

aliter:
df.pivot_table(index=['color','car'], columns='country', values='value',aggfunc='mean')\
  .rename_axis(None, axis=1).reset_index()

aliter more lengthy:
pd.crosstab(index=[df.color,df.car], columns=df.country, values=df.value,aggfunc='mean')\
  .rename_axis(None, axis=1).reset_index()

    color     car  canada  mexico   usa
0   black  toyota    63.0     NaN  65.0
1   brown   honda    61.0     NaN   NaN
2   brown  nissan     NaN     NaN  62.0
3  silver   honda     NaN    64.0  60.0

##============ gropuby is faster than pivot_table===============
df.groupby(["months", "industry"]).agg({"orders": 'sum', "client": 'nunique'}).unstack(level="months").fillna(0)
df.pivot_table(index="industry", columns = "months",values = ["orders", "client"],
               aggfunc ={"orders": 'sum', "client": 'nunique'}).fillna(0)


##------------------- pivoting when a column has duplicate entries-----------------
df = pd.DataFrame({'name': ['Adam', 'Adam', 'Bob', 'Bob', 'Craig','Bob'],
              'number': [111, 222, 333, 444, 555,666],
              'type': ['phone', 'fax', 'phone', 'phone', 'fax','phone']})
# first create new column so that we can have unique pair of indices, then use 'sum' as aggfunc
df['key']=df.groupby(['name','type']).cumcount()
df.groupby(['key','name','type'])['number'].sum().unstack('type').reset_index().rename_axis(None,1) # faster
df.pivot_table(index=['key','name'], columns = 'type', values='number',aggfunc='sum').reset_index() # slow
```


# pandas **query**
```python
## NOTE: pd.query uses pd.eval and which uses numexpr library.
##       From official documentation eval is several order magnitude slower if df has <15k rows.
## NOTE: To use query, always rename columns with spaces to underscores
df.columns = df.columns.map(lambda x: x.replace(' ', '_') if isinstance(x, str) else x)

df = pd.DataFrame({'a': [1,2,3],'b': [4,5,6], 'c': [7,8,9]})
df.query('a==1 and  b==4 and not c == 8')
df.query('a==1 &  b==4 &  ~c == 8')
df.query('a != a.min()')
df.query('a not in b')  # df[df.a.isin(df.b)]
df.query('c != [1, 2]') # df.query('[1, 2] not in c')
df.query('a < b < c and (not mask) or d > 2')
df.query('a == @myvalue')
df.query('a == @mydict['mykey']) 

## exclude all rows if any row value is minimum of that column
df.columns = df.columns.map(lambda x: x.replace(' ', '_') if isinstance(x, str) else x)
q = ' and '.join([f'{i} != {i}.min()' for i in df.columns])
df.query(q)
```

# pandas **qcut**
```python
ser = pd.Series(np.random.randn(100))
factor = pd.qcut(ser, [0, .25, .5, .75, 1.])
ser.groupby(factor).mean()

# Pandas way for R's ntile
# y = c(3,2,2,NA,30,4)
# ntile(y, n=2) # 1  1  1 NA  2  2

y = pd.Series((3,2,2,np.nan,30,4))
cuts = 2
pd.qcut(y,q=cuts, labels=range(1, cuts+1))
```

# pandas **read_csv**
```python
## Always set category dtype if youw have caeggory data
   Auditor ID       Date  Price  Store ID        UPC
0         234 2017-10-18  24.95     66999  268588472
1         234 2017-10-27  49.71     66999  475245085
2         234 2017-10-20  25.75     66999  126967843

prices = pd.read_csv('prices.csv', parse_dates=['Date'],
                      dtype={'Auditor ID': 'category',
                            'Store ID': 'category',
                            'UPC': 'category',
                            'Price': np.float32})
prices.memory_usage(deep=True)


## usecols
usecols_func = lambda x: 'likes_' in x or x == 'A'
df = pd.read_csv('data.csv', index_col='A', usecols=usecols_func)

# read date columns (Best option is use pd.to_datetieme after reading file and using format)
# still we can read simple formats easily
file: a.csv has date "01/12/2019"
df = pd.read_csv('a.csv', parse_dates=[0], infer_datetime_format=True, dayfirst=False,header=None)
'''
parse_date = [0,1]  two date columns
parse_date = [[0,1]] one date from two columns
parse_date = {'mydate': [1,3]} one date called mydate using columns 1 and 3.

Always use dayfirst parameter if you use parse_dates parameter and also use infer
infer date will make operations 10 times faster.

# Best option
df = pd.read_csv('a.csv', header=None)
df[0] = pd.to_datetime(df[0], format='%m/%d/%Y', errors = 'coerce')

ignore: invalid will be string type
coerce: invalid will be NaT

# another example
a = """10 02 2018 1000
12 03 2018 2000
12 04 2019 3000
"""

pd.read_csv(io.StringIO(a),sep='\s+',  header=None)

date_cols = [0,1,2]   # 10 02 2018 gives ==> 2018-10-02
# date_cols = [1,0,2] # 02 10 2018 gives ==> 2018-02-10

df = pd.read_csv(io.StringIO(a),sep='\s+',  header=None, parse_dates={'date': date_cols},
         infer_datetime_format=True, dayfirst=False, index_col=0,keep_date_col=True)


print(df.loc['2018'])
             0   1     2     3
date
2018-10-02  10  02  2018  1000
2018-12-03  12  03  2018  2000
'''
```

# Pandas **read_json**
```python
pd.read_json('[{"A": 1, "B": 2}, {"A": 3, "B": 4}]')
   A  B
0  1  2
1  3  4
```

# pandas **rename**
```python
## rename columns
df1 = df.rename(columns=str.lower) # use inplace=True for inplace operation.
df1 = df.rename(columns=lambda x: x+'n' if x not in ['Height','Width'] else x)
df1 = df.rename(columns={'old': 'new'})
df1 = df.assign(new1=5, new2=df2.c1)
df1 = df.set_axis(list('abc'),axis=1)
df1 = df.reset_index(drop=True).rename(index=lambda x: list('abc')[x]) # rename index labels

## rename using regex
import re

Columns: 'movie_title', 'actor_1,2,3_name', 'actor_1,2,3_facebook_likes'

actor_1_name ==> actor_1  and actor_1_facebook_likes ==> actor_facebook_likes_1
df.rename(columns=lambda x: x.replace('_name','') if '_name' in x 
               else re.sub(r'(actor_)(\d)_(facebook_likes)', r'\1\3_\2',x) if 'facebook' in x 
               else x)

# now we can tidy up the dataframe from wide to long
stubs = ['actor', 'actor_facebook_likes']
df = pd.wide_to_long(df,
                       stubnames=stubs, # columns will become rows
                       i=['movie_title'], # keep this untouched
                       j='actor_num',  # new name
                       sep='_').reset_index()
```

# pandas **stack**
```python
## example1
pd.DataFrame({'c0' : [2,3],'c1' : ['A', 'B']}).stack() # gives only one series with multi-index

## example2
Data:
         Apple  Orange  Banana
Texas       12      10      40
Arizona      9       7      12
Florida      0      14     190

This data is not TIDY, there must be variable names and values.
(df.stack()
  .rename_axis(['state','fruit'])
  .reset_index())

## example3
## data has 5 columns with income ranges like $10-20k
## we will make these 5 columns as a single column called 'income'
## and their values will be called 'frequency' column
religion  <$10k  $10-20k  $20-30k  $30-40k  $40-50k  $50-75k
Agnostic     27       34       60       81       76      137

## using stack *********************
(df.set_index('religion')
  .stack()
  .rename_axis(['religion','income'])
  .reset_index()
  .rename(columns={0:'frequency')
  )

## using melt *********************
(pd.melt(df, id_vars=['religion'], value_vars=df.columns.values[1:],
             var_name='income', value_name='frequency')
  .sort_values(by='religion')
  .head())
```

# pandas **string**
```python
## String split
s.str[0:3] # s.str.slice(0,3)  gives first three letters of series
pd.Series(['16 Jul 1950 - 15:00']).str.split('-').str[0] # don't forget last .str

## extract letters between two underscores
df = pd.DataFrame.from_dict({'c0': ['T2 0uM_A1_A01.fcs'], 'MFI': [6995], 'Count': [8505]})
df['new'] = df['c0'].str.extract(r'_(.*?)_')  # e.g. A1 is extracted

## slice dataframe column values
# pd.Series.str.slice(start,stop,step)
s = pd.Series('GDP-2013')
s.str.slice(0,3) # GDP   (letter 0 to 3) same as s.str[0:3] or s.str[0:3:1]
s.str.slice(4).astype(int) # 2013 (WANRNING: fails if there are nans) (letter 4 to end)
s.str.slice(4).astype(float) # 2013 (works even if there are nans)
s.str[4:].astype(float) # 2013 (works even if there are nans)

## Remove non-ascii characters
import unidecode  # pip install unidecode
s = pd.Series(['mañana','Ceñía'])
s1 = s.apply(unidecode.unidecode) # manana, Cenia  (slow method)
s1 = pd.Series(list(map(unidecode.unidecode,s)))

## string split
df = pd.DataFrame({'mycol': ['Nepal_Kathmandu', 'Japan_Tokyo']})
df['c0'], df['c1'] = zip(*df['mycol'].str.split('_'))
df[['c0','c1']] = df['mycol'].str.split('_', expand=True) # eg. Nepal_Kathmandu gives two new columns

## regex with apply
df = pd.DataFrame({'a': ['123 45']}) # combine two numbers
df['a'].apply(lambda x: re.sub(r'(\d+)\s+(\d+)',r'\1\2',x)).astype(int) # gives 12345
df['a'].str.replace(r'(\d+)\s+(\d+)', lambda x: "{}{}".format(x.group(1), x.group(2)) ).astype(int)

##------------------------------------------------------------------------------------------
## extract string parts
df = pd.DataFrame({'a': ["number 1.23 has 1.2 ",
                         "number 12.2 has 12 "]})

# method 1 extract only first number found
pat1 = r"""(\d+\.\d+)"""
df['extract'] = df['a'].str.extract(pat1)

# method 2 extract first number using multi-line
pat2 = r"""(
\d+ # decimal
\.  # decimal point
\d+ # digits after dot
)"""

df['extract_reX'] = df['a'].str.extract(pat2,flags=re.X)

# method3 findall gives list of all matches
df['findall'] = df['a'].str.findall(pat2,flags=re.X)

# method 4  use findall and get string from list
df['findall_apply'] = df['a'].str.findall(r"(\d+\.\d+)").apply(lambda x: ", ".join(x))

# method 5 get two columns from method4
df[['x','y']] = pd.DataFrame(df['a'].str.findall(r"\d+\.\d+").tolist())

# method 5 here we caputre both 1.2 and 12 making part after decimal optional.
df['extractall_unstack'] = df['a'].str.extractall(r'(\d+(?:\.\d+)?)').unstack().apply(lambda x: ", ".join(x)).droplevel(0)

# method 6 get two columns from method5
df[['xx','yy']] = df['a'].str.extractall(r'(\d+(?:\.\d+)?)').unstack()[0] # or, unstack().iloc[:,0]


print(df)
                      a extract extract_reX      findall findall_apply  extractall_unstack x     y    xx   yy
0  number 1.23 has 1.2     1.23        1.23  [1.23, 1.2]     1.23, 1.2   1.23, 12.2        1.23 1.2   1.23 1.2
1   number 12.2 has 12     12.2        12.2       [12.2]          12.2   1.2, 12           12.2 None  12.2 12
#----------------------------------------------------------------------------------------------------------------

## Make second group titlecase
data = ['one two three', 'foo bar baz']
pat = r"(?P<one>\w+) (?P<two>\w+) (?P<three>\w+)"
repl = lambda m: m.group('two').title()  # lower(), upper(), swapcase() etc
pd.Series(data).str.replace(pat, repl)

## Extract from strings
addr = pd.Series([
    'Washington, D.C. 20003',
    'Brooklyn, NY 11211-1755',
    'Omaha, NE 68154',
    'Pittsburgh, PA 15211'
])

regex = (r'(?P<city>[A-Za-z ]+), '      # One or more letters
         r'(?P<state>[A-Z]{2}) '        # 2 capital letters
         r'(?P<zip>\d{5}(?:-\d{4})?)')  # Optional 4-digit extension

addr.str.replace('.', '').str.extract(regex) # this gives 3 columns
** using dataframe method **
df = pd.DataFrame([
    'Washington, D.C. 20003',
    'Brooklyn, NY 11211-1755',
    'Omaha, NE 68154',
    'Pittsburgh, PA 15211'
],columns=['addr'])

regex =  (r'([A-Za-z ]+), '     # One or more letters
         r'([A-Z]{2}) '         # 2 capital letters
         r'(\d{5}(?:-\d{4})?)') # Optional 4-digit extension (?:-\d{4})?) 
                                # Note:  ?: is non-capturing group
df[['city','state','zipcode']] = df['addr'].str.replace('.', '').str.extract(regex,expand=True)

# Extract from strings
s = pd.Series(['Auburn (Auburn University)[1]\n'])
regex = (r'(?P<City>.+)'
         r' \((?P<University>.+)\)'
         r'.*')  
s.str.extract(regex)

##-------------------------------------------------------------------

s = pd.Series(['M35 35-39', 'M40 40-44', 'M45 45-49', 'M50 50-54', 'M55 55-59',
       'M60 60-64', 'M65 65-69', 'M70 70-74', 'M75 75-79', 'M80 80+'])
str.extract('(?P<Sex>[MF])\d\d\s(?P<AgeGroup>\d{2}[-+](?:\d{2})?)', expand=True)

# series split expand
df = pd.DataFrame({'c0': [' a b c ', 'd e f', 'g h i ']})
df[['c1','c2','c3']] = df['c0'].str.split(expand=True) # extra whitespaces are removed
df = df.drop('c0',1)

##----------------------------------------------------------------
# another example
df = pd.DataFrame({'BP': ['100/80'],'Sex': ['M']})

# using str split
df[['sys','dias']] = df['BP'].str.split(pat='/', expand=True)

# using str extract
df[['sys','dias']] = df['BP'].str.extract(r'(\d+)/(\d+)',expand=True)

# using one-liner
df.drop('BP', 1).join(
    df['BP'].str.split('/', expand=True)
            .set_axis(['BPS', 'BPD'], axis=1, inplace=False)
            .astype(float))

# using assign **pop extract
df2 = (df.drop('BP',axis=1)
       .assign(BPS =  lambda x: df.BP.str.extract('(?P<BPS>\d+)/').astype(float))
       .assign(BPD =  lambda x: df.BP.str.extract('/(?P<BPD>\d+)').astype(float)))
# another method
df = df.assign(**df.pop('BP').str.extract(r'(?P<BPS>\d+)/(?P<BPD>\d+)',expand=True).astype(float))
```

# pandas **unstack**
```python
## Unstack is very useful after groupby multiple columns
columns = date, type, amount
df.groupby(['date','type']).sum()['amount'].unstack().plot() # x= date, y = daily total spent per type of items

## example
employee.groupby(['RACE', 'GENDER'])['BASE_SALARY'].mean().astype(int).unstack('GENDER')

## example
flights.groupby(['AIRLINE', 'ORG_AIR'])['CANCELLED'].sum().unstack('ORG_AIR', fill_value=0)
flights.pivot_table(index='AIRLINE', columns='ORG_AIR', values='CANCELLED', aggfunc='sum',fill_value=0).round(2)
```

# pandas **wide_to_long**
```python
df = pd.DataFrame({"A.2018.5" : ['Atlanta','Austin'],  # These value will be in column A
                   "A.2019.5" : ['Arlington','Albuquerque'], # These values will be in column A, total 4 values
                   "B.2018.5" : ['Boston','Baltimore'], # These value will be in column B
                   "B.2019.5" : ['Bakersfield', 'Buffalo'], # These value will be in column B
                   "apple_price" : [1.5,2.0], # These values are kept same and may be repeated
                   "id" : [1,4]}) # Hint: i = intact

## make it long  sep="" and suffix="\d+" is default.
pd.wide_to_long(df, stubnames=['A','B'], i=['apple_price','id'] , j='year',sep='.',suffix='.+').reset_index()

   apple_price  id    year            A            B
0          1.5   1  2018.5      Atlanta       Boston
1          1.5   1  2019.5    Arlington  Bakersfield
2          2.0   4  2018.5       Austin    Baltimore
3          2.0   4  2019.5  Albuquerque      Buffalo

## example 2 *********************
Data:
year           artist          track   time   date.entered  wk1  wk2
2000           Justin          Baby    4:22   2000-02-26    87   82
2000           Adele           Hello   3:15   2000-09-02    91   87

# solution:
# wk1 wk2 means week is 1 and week 2 and they have different ranks for the given song.
# df = pd.read_clipboard()
df = pd.DataFrame({'year' : [2000, 2000],
          'artist' : ['Justin', 'Adele'],
          'track' : ['Baby', 'Hello'],
          'time' : ['4:22', '3:15'],
          'date.entered' : ['2000-02-26', '2000-09-02'],
          'wk1' : [87, 91],
          'wk2' : [82, 87]})

# reorder columns and make wk1 wk2 end columns
df = df[['year', 'artist', 'date.entered', 'time', 'track', 'wk1', 'wk2']]
df1 = (pd.wide_to_long(df, 'wk', i=df.columns.values[:-2], j='week')
         .reset_index()
         .rename(columns={'date.entered': 'date', 'wk': 'rank'})
         .assign(date = lambda x: pd.to_datetime(x['date']) + 
                                  pd.to_timedelta((x['week'].astype(int) - 1) * 7, 'd'))
         .sort_values(by=['track', 'date'])
)
print(df1)
   year  artist       date  time  track week  rank
0  2000  Justin 2000-02-26  4:22   Baby    1    87
1  2000  Justin 2000-03-04  4:22   Baby    2    82
2  2000   Adele 2000-09-02  3:15  Hello    1    91
3  2000   Adele 2000-09-09  3:15  Hello    2    87

# Example from documentation
df = pd.DataFrame({'A(quarterly)-2010': [1,2,3],'A(quarterly)-2011': [4,5,6],
                   'B(quarterly)-2010': [7,8,9],'B(quarterly)-2011': [10,11,12],
                   'X' : [2,5,7]})
stubnames = sorted(
     set([match[0] for match in df.columns.str.findall(
         r'[A-B]\(.*\)').values if match != [] ]))
df['id'] = df.index # we need this
pd.wide_to_long(df, stubnames, i='id', j='year', sep='-') 
# Note:  change 2010 ==> one then we need to use suffix:   sep='_', suffix='\w' (default is \d+)
```

# pandas **where**
```python
df.where(mask, -df) # mask = df % 3 == 0  # divisibles of 3 are positive (eg. 0, 3, 6, 9)
df.where(lambda x: x > 4, lambda x: x + 10) # if value is larger than 4, add 10 to it.

## numpy where
df['young_male'] = np.where( (df.sex=='male') &(df.age<30) ,1,0) # fastest
df['young_male'] = ((df.sex == 'male') & (df.age<30)).astype(int) # nearly same fast
df['young_male'] = ((df.sex == 'male') & (df.age<30)).map({True:1, False:0}) # slow

## multiple conditions
**WARNING: Be careful about NaNs in np.where
df['age_cat'] = np.where(df.age > 60, 'old', (np.where(df.age <20, 'child', 'medium'))) # BAD!!
df.loc[df.age.isnull(), 'age_cat'] = np.nan # THIS IS ALWAYS NEEDED

## nested where is complicated but fast
df['age_cat'] = np.where(df.age > 60, 'old',
                         (np.where(df.age <20, 'child',
                                   np.where(df.age.isnull(), np.nan, 'medium'))))

## pd.cut is easiest and nearly fast
df['age_cat1'] = pd.cut(df.age, [0, 20, 60, np.inf], labels=['child','medium','old'])
df['age_cat1'] = pd.cut(df.age, [0, 20, 60, np.inf], include_lowest=False, right=False, labels=['child','medium','old'])

## other examples
df['year'] = np.where(yr <= 20, 2000 + yr, 1900 + yr) # yr = df['year']
df.a, df.b = np.where(df.a > df.b, [df.b, df.a], [df.a, df.b])  # swap values e.g. when last value is nan for persons name
df['d'] = np.where(cond1,  np.where(cond2,cond2_true, cond2_false), cond1_false)
```