# Built-ins
```python
# highlight max
df.style.apply(highlight_max, color='darkorange', axis=None) # axis=None is max of whole dataframe
df.style.apply(highlight_max, subset=['B', 'C', 'D'])
df.style.applymap(color_negative_red,subset=pd.IndexSlice[2:5, ['B', 'D']])

# number format
df.style.format("{:.2%}")
df.style.format({'B': "{:0<4.0f}", 'D': '{:+.2f}'})
df.style.format({"B": lambda x: "±{:.2f}".format(abs(x))})

# chaining styles
df.style\
  .apply(highlight_max)\
  .set_precision(2)
  
# some examples
df['col1'].value_counts(normalize=True).to_frame().style.format("{:.2%}")
```

# Highlight rows
```python
# example 1
df1 = df.style.apply(lambda x: ['background: lightgreen' if (x.colA == 'foo' or x.colA == 'bar')
                                else '' for i in x], axis=1)

# example 2
def color(row):
    if row.colA == 1:
        return ['background-color: red'] * len(row)
    elif row.colB == 2:
        return ['background-color: yellow'] * len(row)
    return [''] * len(row)

df.style.apply(color, axis=1)
```

# Highlight row or column
```python
import pandas as pd
df = pd.DataFrame([[1,0],[0,1]])

# highlight column
df.style.apply(lambda x: ['background: lightblue' if x.name == 0 else '' for i in x])

# hightlight row
df.style.apply(lambda x: ['background: lightgreen' if x.name == 0 else '' for i in x], 
               axis=1)
               
# both rows and columns
(df.style
     .apply(lambda x: ['background: yellow' if (x.name == 'colA')
                                else '' for i in x], axis=0)
     .apply(lambda x: ['background: lightblue' if (x.name == 'colB')
                                else '' for i in x], axis=0)
     .apply(lambda x: ['background: lightblue' if (x.name == 'colC')
                                else '' for i in x], axis=0)
     .apply(lambda x: ['background: lightgreen' if (x['colD'] == True)
                                else '' for i in x], axis=1)
)
```
