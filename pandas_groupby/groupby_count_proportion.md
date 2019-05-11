# Get proportion of group elements
```python
df = pd.DataFrame({'Name': ['name1', 'name1', 'name1', 'name2', 'name2'],
          'Number': [700, 600, 600, 300, 400],
          'Year': [1998, 1999, 2000, 1998, 1999],
          'Sex': ['Male', 'Male', 'Male', 'Male', 'Male'],
          'Criteria': ['N', 'N', 'N', 'Y', 'Y']})
print(df)
    Name  Number  Year   Sex Criteria
0  name1     700  1998  Male        N
1  name1     600  1999  Male        N
2  name1     600  2000  Male        N
3  name2     300  1998  Male        Y
4  name2     400  1999  Male        Y

# Solution
g_sum = df.groupby(["Sex", "Year", "Criteria"]).sum()
print(g_sum)
                    Number
Sex  Year Criteria
Male 1998 N            700
          Y            300
     1999 N            600
          Y            400
     2000 N            600

Again, grouby with level 0 and 1
df1 = g_sum.groupby(level=[0, 1]).apply(lambda g: g / g.sum())
print(df1)

Sex  Year Criteria
Male 1998 N            0.7
          Y            0.3
     1999 N            0.6
          Y            0.4
     2000 N            1.0
```
