# Checking nans
```python

# Find total number of nans in columns
null_columns=df.columns[df.isnull().any()]
df[null_columns].isnull().sum()

# All total nans
df.isnull().sum().sum()

# Given column nans rows
nan_rows = df[df['genres'].isnull()]
print(nan_rows.shape)
nan_rows.head()
```
