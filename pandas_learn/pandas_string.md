Table of Contents
=================
   * [capitalize words](#capitalize-words)
   * [Split at comma or whitespace](#split-at-comma-or-whitespace)
   * [Sort df by a value in json element of column](#sort-df-by-a-value-in-json-element-of-column)

# capitalize words
```python
ser = pd.Series(['how', 'to', 'kick', 'ass?'])
pd.Series([i.title() for i in ser]) # best

ser.apply(lambda x: x.title()) # map also works
ser.apply(lambda x: x[0].upper() + x[1:])

ser.map(len) # gives length
```

# Split at comma or whitespace
```python
df = pd.DataFrame(["STD, City    State",
"33, Kolkata    West Bengal",
"44, Chennai    Tamil Nadu",
"40, Hyderabad    Telengana",
"80, Bangalore    Karnataka"], columns=['row'])

                          row
0          STD, City    State
1  33, Kolkata    West Bengal
2   44, Chennai    Tamil Nadu
3  40, Hyderabad    Telengana
4  80, Bangalore    Karnataka

# fastest method (600 us)
df_out = df.row.str.split(',|\s\s+', expand=True)
df_out = df_out.iloc[1:]
columns = df.iloc[0,0].replace(",", "").split()
df_out.columns = columns

**slow: columns = df.iloc[0].str.split(',|\s\s+', expand=True).values.squeeze()
**slow: df_out.columns = columns

# one-liner from fastest method is also slow, needs to do same operation muliple times
(df.row.str.split(',|\s\s+', expand=True)
.iloc[1:]
.rename(columns= 
        lambda x: df.iloc[0,0].replace(",", "").split()[x]))

# slow one-liner (900 us)
(df["row"].iloc[1:]
.str.replace(",", "")
.str.split(expand=True, n=2)
.rename(columns= 
        lambda x: df.iloc[0,0].replace(",", "").split()[0]))

# slower, but one-liner
(df["row"].iloc[1:]
.str.replace(",", "")
.str.split(expand=True, n=2) # n=2 gives 3 splits at spaces
.rename(columns= dict(zip(range(3), 
                 df.iloc[0,0].replace(",", "").split()))))
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
    
df.iloc[np.argsort([float(x.get('total_expense', '-1')) for x in df['meta']])]

# handles nans
import ast
if isinstance(df.at[0, 'meta'], str):
    df['meta'] = df['meta'].map(ast.literal_eval)
    
u = [  
  float(x.get('total_expense', '-1')) if isinstance(x, dict) else -1 
  for x in df['meta']
]
df.iloc[np.argsort(u)]


# Regex are always slow (use list comp as above)
df = pd.read_clipboard(r'\s\s+',engine='python')
df['total_expense'] = df.meta.str.extract(r"""u'total_expense': u'([0-9.]+)'""",expand=False)
df.sort_values('total_expense').drop('total_expense')
```

# pandas string methods
![](images/df_str_methods1.png)
![](images/df_str_methods2.png)

