# grouby group_keys example
```python
Note: group_keys=False  no need to do .values
Note: as_index=False    no need to do reset_index()


df = pd.DataFrame({'A':['A','A','B','B'],'B':[10,20,30,40],'C':[1,3,3,5]})
print(df)
   A   B  C
0  A  10  1
1  A  20  3
2  B  30  3
3  B  40  5

# If we use group_keys=False, we dont need to use .values to get values
df['D'] = df.groupby('A', group_keys=False).apply(lambda x: x.B - x.C.mean())
print(df)
   A   B  C     D
0  A  10  1   8.0
1  A  20  3  18.0
2  B  30  3  26.0
3  B  40  5  36.0

# groupkeys True
x = df.groupby('A', group_keys=True).apply(lambda x: x.B - x.C.mean())
print(x)
A
A  0     8.0
   1    18.0
B  2    26.0
   3    36.0

NOTE: to get the values we can use .values
# groupkeys True
df['D1'] = df.groupby('A', group_keys=True).apply(lambda x: x.B - x.C.mean()).values
print(df)

```
