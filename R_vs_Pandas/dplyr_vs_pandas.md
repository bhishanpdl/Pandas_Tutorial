# Example 1
```R
standardized_flights2 <- flights %>%
  filter(!is.na(air_time)) %>%
  group_by(dest, origin) %>%
  mutate(
    air_time_median = median(air_time),
    air_time_iqr = IQR(air_time),
    n = n(),
    air_time_standard = (air_time - air_time_median) / air_time_iqr
  )

```
```python
dfagg = flights.dropna(subset=['air_time'],how='any'
                      ).groupby(['dest','origin']).agg({'air_time':
                       [np.median, IQR, 'count']})

dfagg.columns = [i[0]+'_'+i[1] if i[1] else i[0] for i in dfagg.columns.ravel()]
dfagg = dfagg.reset_index()
df1 = flights.dropna(subset=['air_time'],how='any')
df2 = df1.merge(dfagg, on=['dest','origin'], how='left')
standardized_flights2 = df2.eval("air_time_standard = (air_time - air_time_median) / air_time_iqr",inplace=False)
```

# Example 2
For each plane, count the number of flights before the first delay of greater than 1 hour
```R
df = flights %>%
  arrange(tailnum, year, month, day) %>%
  group_by(tailnum) %>%
  mutate(delay_gt1hr = dep_delay > 60) %>%
  mutate(before_delay = cumsum(delay_gt1hr)) %>%
  filter(before_delay < 1) %>%
  count(sort = TRUE)
```
```python
df = (flights.filter(items=['tailnum', 'year', 'month', 'day','dep_delay'])
      .dropna(subset=['dep_delay'],how='any')
      .sort_values(['tailnum','year','month','day'])
      .assign( 
          delay_gt1hr = lambda dff: dff.groupby('tailnum')['dep_delay']
                        .apply(lambda x: x > 60).astype(int),
                        
          before_delay = lambda dff: dff.groupby('tailnum')['delay_gt1hr'].cumsum()
              )
      .query("before_delay < 1")
      .tailnum.value_counts()
      
     )
```
