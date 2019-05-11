# pandas cut
```python
import numpy as np
import pandas as pd

t = pd.Series([4,16,17,27,28,38,39,47,48,53,54,116,117])

#=======================================
# Multiple if/else using pands cut
bins = [-np.inf, 16, 27, 38, 47, 53, 54, 116, np.inf]
labels=['G','F','T','A','E','D','C','B'] 

t1 = pd.cut(t, bins=bins, labels=labels,include_lowest=True,right=False).rename(index=t)

#=======================================
# multiple if/else using pandas apply
def group_estimator(i):
    a = 0
    if i<16:
        a = "G"
    elif i>=16 and i<27:
        a = "F"
    elif i>=27 and i<38:
        a = "T"
    elif i>=38 and i<47:
        a = "A"
    elif i>= 47 and i<53:
        a = "E"
    elif i>= 53 and i<54:
        a = "D"
    elif i>=54 and i<116:
        a = 'C'
    else:
        a = "B"
    return a
    
t2 = t.apply(group_estimator).rename(index=t)
print(pd.DataFrame([t1,t2]).T)
     0  1
4    G  G
16   F  F
17   F  F
27   T  T
28   T  T
38   A  A
39   A  A
47   E  E
48   E  E
53   D  D
54   C  C
116  B  B
117  B  B
```
