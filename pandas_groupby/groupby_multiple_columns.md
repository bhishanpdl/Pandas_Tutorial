# sum of two columns and divide by group sum
```python
df = pd.DataFrame({'a': list("aabbb"),
                   'w': [1,2,3,4,5],
                   'x': [10,40,30,40,130],
                   'y': [40,60,70,60,70],
                   'z': [1,4,3,1,6]})

print(df)
   a  w    x   y  z
0  a  1   10  40  1
1  a  2   40  60  4
2  b  3   30  70  3
3  b  4   40  60  1
4  b  5  130  70  6




# There is no ungroup, just assessing new column name manually!
df['new'] = df.groupby('a')\
    .apply(lambda grp: (grp.x + grp.y) / grp.z.sum()).values

print(df)
   a  w    x   y  z   new
0  a  1   10  40  1  10.0
1  a  2   40  60  4  20.0
2  b  3   30  70  3  10.0
3  b  4   40  60  1  10.0
4  b  5  130  70  6  20.0

 ```
