Table of Contents
=================
   * [multi index selections](#multi-index-selections)
   * [Multi-index examples](#multi-index-examples)

# multi index selections
https://stackoverflow.com/questions/55381058/filtering-rows-on-dataframe-based-on-data-in-a-series-failing-with-dataframe-ran
```python
df = pd.DataFrame({'year' : [1999]*4 + [2000]*4,
                   'category': list('AABBAABB'),
                   'grade': [3.5,7.2,0.2,6.4,1.4,2.5,3.3,8.4]})

s = pd.Series([5.2, 5.1,3.7,6.1], index=[[1999, 1999, 2000 , 2000], ['A', 'B', 'A', 'B']])
s.index.names = ['year','category']
year  category
1999  A           5.2
      B           5.1
2000  A           3.7
      B           6.1

# select passed students
df[df.set_index(['year','category']).grade.gt(s).values]
  category  grade  year
1        A    7.2  1999
3        B    6.4  1999
7        B    8.4  2000

# passed students multi-index
df1=df.set_index(['year','category'])
df1[df1.grade.gt(s.reindex(df1.index)).values]

               grade
year category
1999 A           7.2
     B           6.4
2000 B           8.4
```

# Multi-index examples
```python
import numpy as np
import pandas as pd
import seaborn as sns

tips = sns.load_dataset("tips")
tips.head()

   total_bill   tip     sex smoker  day    time  size
0       16.99  1.01  Female     No  Sun  Dinner     2
1       10.34  1.66    Male     No  Sun  Dinner     3
2       21.01  3.50    Male     No  Sun  Dinner     3
3       23.68  3.31    Male     No  Sun  Dinner     2
4       24.59  3.61  Female     No  Sun  Dinner     4

df = tips.groupby(['smoker','time']).mean()
print(df)
               total_bill       tip      size
smoker time
Yes    Lunch    17.399130  2.834348  2.217391
       Dinner   21.859429  3.066000  2.471429
No     Lunch    17.050889  2.673778  2.511111
       Dinner   20.095660  3.126887  2.735849

df.swaplevel()
               total_bill       tip      size
time   smoker
Lunch  Yes      17.399130  2.834348  2.217391
Dinner Yes      21.859429  3.066000  2.471429
Lunch  No       17.050889  2.673778  2.511111
Dinner No       20.095660  3.126887  2.735849

df.unstack() # level=-1  innermost goes to column
       total_bill                  tip                size
time        Lunch     Dinner     Lunch    Dinner     Lunch    Dinner
smoker
Yes     17.399130  21.859429  2.834348  3.066000  2.217391  2.471429
No      17.050889  20.095660  2.673778  3.126887  2.511111  2.735849

df.unstack(level=0) # outermost index label
       total_bill                  tip                size
smoker        Yes         No       Yes        No       Yes        No
time
Lunch   17.399130  17.050889  2.834348  2.673778  2.217391  2.511111
Dinner  21.859429  20.095660  3.066000  3.126887  2.471429  2.735849
```

