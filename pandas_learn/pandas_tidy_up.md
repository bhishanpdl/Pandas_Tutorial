Table of Contents
=================
   * [Tidy up when single column contains multiple variables](#tidy-up-when-single-column-contains-multiple-variables)
   * [Tidy up when single cell contains multiple values](#tidy-up-when-single-cell-contains-multiple-values)

# Tidy up when single column contains multiple variables
#==============================================================================
```python
df = pd.DataFrame({'Name': list('ABC'),
                  'Date': np.arange(2010,2013),
                 'Info': ['Height', 'Weight','Size'],
                 'Value': [6, 160, 3]})
                 
(df.set_index(['Name','Date', 'Info'])
          .squeeze()
          .unstack('Info')
          .reset_index()
          .rename_axis(None, axis='columns'))
          
# using pivot_table
(df.pivot_table(index=['Name', 'Date'], 
                        columns='Info', 
                        values='Value', 
                        aggfunc='first')
           .reset_index()
           .rename_axis(None, axis='columns'))
```


# Tidy up when single cell contains multiple values
#==============================================================================
```python

      City             Geolocation
0  Houston  29.7604° N, 95.3698° W
1   Dallas  32.7767° N, 96.7970° W
2   Austin  30.2672° N, 97.7431° W

g1 = '29.7604° N, 95.3698° W'
g2 = '32.7767° N, 96.7970° W'
g3 = '30.2672° N, 97.7431° W'
df = pd.DataFrame({'City': ['Houston','Dallas','Austin'],
                       'Geolocation': [g1,g2,g3]})

regexp = r'(?P<latitude>[0-9.]+).\s* (?P<latitude_dir>N|S), '+ \
         r'(?P<longitude>[0-9.]+).\s* (?P<longitude_dir>E|W)'
df = pd.concat([df['City'], 
          df['Geolocation'].str.extract(regexp, expand=True)
          ], axis='columns')

df[['latitude','longitude']] =df[['latitude','longitude']].astype(np.float)
df['latitude_dir'] =df['latitude_dir'].astype('category')
df['longitude_dir'] =df['longitude_dir'].astype('category')
# note: multiple columns astype is not supported for category for pandas24.0
df.dtypes
```
