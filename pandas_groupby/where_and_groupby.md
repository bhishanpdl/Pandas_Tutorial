Link: https://stackoverflow.com/questions/54838069/create-new-col-based-on-transformation-on-some-group-based-on-condition/54838118#54838118

```python
d = dict(group=[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3], times=[0,1,2,3,4]*3, values=np.random.rand(15))
df = pd.DataFrame.from_dict(d)
df

# For each group, I'd like to get the max value for which time is <= 3
df['new_value'] = df['values'].where(df.times < 3).groupby(df.group).transform('max')
df
```
