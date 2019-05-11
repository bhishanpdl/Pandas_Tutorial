# Create json from two columns of df
```python
PlayerID    Name     Current Player  First Season    Last Season
76001   Abdelnaby, Alaa       0     1990            1994
76002   Abdul-Aziz, Zaid      0     1968            1977
76003   Abdul-Jabbar, Kareem  0     1969            1988
51      Abdul-Rauf, Mahmoud   0     1990            2000
1505    Abdul-Wahad, Tariq    0     1997            2003

df = pd.read_clipboard('\s\s+')

## using json
myjson = [{'label': name, 'value': pid} 
           for pid,name in zip(df['PlayerID'].values, df['Name'].values)]

import json
with open('myjson.json','w') as fo:
    json.dump(myjson,fo,indent=4)

## using pandas  (This method is more than 100 times slower than dict comp!)
myjson = (df.reindex(['Name', 'PlayerID'], axis=1)
   .set_axis(['label', 'value'], axis=1, inplace=False) # Also: .to_dict('r') or .to_dict('record') gives json.
   .to_json('myjson2.json', orient='records')  # NOTE: if we keep only orient, it will display json in jupyter-notebook
)
```
