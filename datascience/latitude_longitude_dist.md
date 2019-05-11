# distance between latitude and latitude
https://stackoverflow.com/questions/55464087/how-to-calculate-distance-using-latitude-and-longitude-in-a-pandas-dataframe
```python
df = pd.DataFrame({'Lat': [43.937845, 44.310739, 44.914698],
          'Lon': [-97.905537, -97.58882, -99.003517]})
print(df)
         Lat        Lon
0  43.937845 -97.905537
1  44.310739 -97.588820
2  44.914698 -99.003517


from sklearn.neighbors import DistanceMetric

dfr = df.copy()
dfr.Lat = np.radians(df.Lat)
dfr.Lon = np.radians(df.Lon)
hs = DistanceMetric.get_metric("haversine")

matrix = (hs.pairwise(dfr)*6371) # Earth radius in km.
                                 # element i,j is distance between i and j
matrix

array([[  0.        ,  48.56264446, 139.2836099 ],
       [ 48.56264446,   0.        , 130.57312786],
       [139.2836099 , 130.57312786,   0.        ]])

matrix[0,1] = 48.56264446 km  is distance between row0 and row1.

(hs.pairwise(dfr.iloc[:2,:])*6371)[0,1] # 48.56264446
```
