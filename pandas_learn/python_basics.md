Table of Contents
=================
   * [list append vs extend](#list-append-vs-extend)
   * [check two objects are view or copy](#check-two-objects-are-view-or-copy)
   * [changing list inside function](#changing-list-inside-function)
   * [Python Basics](#python-basics)
   * [Reloading a module](#reloading-a-module)
   * [Counter and from_iterable](#counter-and-from_iterable)

# list append vs extend
```python
a = [1,2,3,4,5]
b = [6,7,8,9]
a.append(b) # [1, 2, 3, 4, 5, [6, 7, 8, 9]]
a.extend(b) # [1, 2, 3, 4, 5, 6, 7, 8, 9]]
```

# check two objects are view or copy
```python
a = np.array([1,2,3,4,5])
b = a[:2]
b.flags   # if OWNDATA is False when we change a, b changes.
```

# changing list inside function
```python
def fun(x):
    x[0] = 5
    return x

g = [10,11,12]

fun(g), g # list g is chaned both are [5,11,12] now.
```

# Python Basics
```python
import itertools

# trick 1
lst = [1, 'hello', 'there']
print(*lst, sep=' ')

# flatten list
lsts = [['hello','hi'], ['how'], ['are'], ['you'],['today']]
flatten = lambda x: list(itertools.chain.from_iterable(x))
lst = flatten(lsts)

# if else
beta = 10
alpha = 100 if beta ==10 else 200 if beta == 20 else 0
print('alpha = ', alpha)

# pretty print
from pprint import pprint
j = { "name":"John", "age":30, "car": 'Farrari' }  # json
print(j)
pprint(j, width=20)

# Some regex recap
re.sub('(?<=a)b', r'd', 'abxb') # lookbehind 'adxb'
re.sub(r'(foo)', r'\g<1>123', 'foobar') # groups 'foo123bar'

re.sub("30 apples", r"apples 30", 'Look 30 apples.')
re.sub("(\d+) (\w+)", r"\2 \1", 'Look 30 apples.')
re.sub("(?P<num>\d+) (?P<fruit>\w+)", r"\g<fruit> \g<num>", 'Look 30 apples.')
re.sub(r"(\d+)(\s+)(apples)", r"\3\2\1", 'Look 30 apples.')
```

# Reloading a module
```python
import matplotlib.pyplot as plt

plt.xlabel = 'xlabel' # this is wrong and instead of creating xlabel, it changes the plt.xlabel object

plt.xlabel('xlabel') # TypeError: 'str' object is not callable

To fix:
================
from importlib import reload
reload(matplotlib.pyplot)

import matplotlib.pyplot as plt

plt.plot([10,20])
plt.xlabel('xlabel')

```

# Counter and from_iterable
```python
# Find count of unique keys of each dictionaries
a = [{'x': 10, 'y': 20},{'x': 2, 'z': 2},{'y': 2, 'z': 4}]
b = [2,4,6]
df = pd.DataFrame({'a': a, 'b': b})

from collections import Counter
from itertools import chain

Counter(chain.from_iterable(df['a'])) # gives oak:2 since there are two oak keys.
Counter(y for x in df['a'] for y in x)
pd.concat(map(pd.Series, df['a'])).index.value_counts().to_dict()
```

# python multiprocessing
```python
import numpy as np
import pandas as pd
import multiprocessing

np.random.seed(100)
s = pd.Series(np.random.randint(0,10, 10))

def myfunc(x):
    return x+100

def parallelize(data, func):
    ncores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(ncores)


    data_split = np.array_split(data, ncores)

    data = np.concatenate(pool.map(myfunc,data_split))
    data = pd.Series(data)
    pool.close()
    pool.join()
    return data

result = parallelize(s.values, myfunc)

print(s.values)
print(result.values)
[8 8 3 7 7 0 4 2 5 2]
[108 108 103 107 107 100 104 102 105 102]

s = pd.Series(np.random.randint(0,10, 1000_000))
%timeit parallelize(s.values, myfunc)
1 loop, best of 3: 689 ms per loop

%timeit s.values + 100
10 loops, best of 3: 37 ms per loop  # multiprocessing is slow sometimes
```
