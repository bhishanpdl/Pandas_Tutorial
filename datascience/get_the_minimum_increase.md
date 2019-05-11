# get the minimum increase in values
```python
import pandas as pd

df = pd.DataFrame({'A': [0, 100, 50, 100],
                   'B': [5, 2, 2, 0],
                   'C': [10, 20, 40, 400]})
     A  B    C
0    0  5   10
1  100  2   20
2   50  2   40
3  100  0  400

# answer
df.diff()[df.diff() >0].min().fillna(0)
A    50.0
B     0.0
C    10.0


note: df.diff()
       A    B      C
0    NaN  NaN    NaN
1  100.0 -3.0   10.0
2  -50.0  0.0   20.0
3   50.0 -2.0  360.0
```
