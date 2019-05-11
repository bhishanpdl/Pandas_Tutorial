# Create rank, relative value and relative_label for 10 largest values
```python
gdp has columns: ['country_name', 'country_code', 'year', 'value']

gdp = gdp.groupby('year').apply(lambda x: x.nlargest(10,columns='value')).reset_index(drop=True)

**without using apply FASTER
gdp.dropna(subset=['value']).sort_values('value').groupby('year').tail(10).sort_values(by=['year','value'],ascending=False)

# all the data is already sorted by year, adding rank is easy
gdp['rank'] = list(range(1,11)) * (int(gdp.shape[0]/10))
gdp['value_relative'] = gdp.value.values / np.repeat(gdp.value[::10].values, 10)
gdp['label_relative'] = np.round(gdp['value'].values/1e9, 0).astype(int)
```
