# Basic Python and Numpy
```python
# looping
params = [10,20,30]
for e,i in enumerate(reversed(range(len(params)))):
    print(e,i,params[i])

for i in range(len(params)):
    print(i, ~i, params[i], params[~i])

# numpy axis increase
x = np.array([1,2,3]) # shape (3,) 1d array  (we can get this from flatten(), or ravel() )
x[None,:]       #  : becomes 3 so shape is 1,3 
x[:,None]       #  : becomes 3 so shape is 3,1
x.reshape(-1,1) # -1 becomes 3 so shape is 3,1
x.reshpae(1,-1) # -1 becomes 3 so shape is 1,3
x[:,np.newaxis] #  : becomes 3 so shape is 3,1
x[np.newaxis,:] #  : becomes 3 so shpae is 1,3

## numpy elements between two values
a[np.logical_and(a>low, a<high)]
a[(a>low) & (a<high)]
idx = np.where((a >low) & (a <high)) # this gives the indices

# vandermonde matrix
x = np.array([1,2,3])
one_x_x2 = np.vander(x,3,increasing=True) # shape 3,3 first column is all 1 second column is x.

# Sort by column 0, then by column 1
lexsorted_index = np.lexsort((a[:, 1], a[:, 0])) 
a[lexsorted_index]
** Using Pandas
df = pd.DataFrame(a)
df.sort_values(by=[0,1])

# vectorize
# alternatively we can use: @np.vectorize  above func and
# no need to define foo_v.
def foo(x):
    return x+1 if x > 10 else x-1
foo_v = np.vectorize(foo, otypes=[float])
print('x = [10, 11, 12] returns ', foo_v([10, 11, 12]))

# apply_along_axis
def max_minus_min(x):
    return np.max(x) - np.min(x)
print('Row wise: ', np.apply_along_axis(max_minus_min, axis=1, arr=a))

# digitize (each elements of array belogs to which group?)
x = np.arange(10)
bins = np.array([0, 3, 6, 9])
print(x)
print(np.digitize(x, bins)) # element 0 of x belongs to first bin 1 from bin and so on

# unique counts of an array
x = np.array([1,1,1,2,2,2,5,25,1,1])
unique, counts = np.unique(x, return_counts=True) # using numpy
uniq_counts = np.stack(unique, counts).T   # same as vstack and also np.asarray((unique, counts)).T
pd.Series(x).value_counts() # using pandas .values gives np array
c = collections.Counter(x)  # using collections.Counter, it also has .most_commont(N) 
np.array(list(c.items()))   # c.items() is dict, make list and then make array

# bincount
x = np.array([1,1,2,2,2,4,4,6,6,6,5]) # doesn't need to be sorted
print(x)
print(list(range(x.max())))
print(np.bincount(x)) # [0, 2, 3, 0, 2, 1, 3]  0 occurs 0 times, 1 occurs 2 times

# Histogram example
x = np.array([1,1,2,2,2,4,4,6,6,6,5,5])
counts, bins = np.histogram(x, [0, 2, 4, 6, 8])
print('Counts: ', counts) # number of items in that bin
print('Bins: ', bins) # just the second argument of hist

# Create date sequence
npdates = np.arange(np.datetime64('2018-02-01'), np.datetime64('2018-02-10'))
bus = npdates[np.is_busday(npdates)] # Check if its a business day
dates = npdates.tolist() # convert npdate to datetime.datetime 
years = [dt.year for dt in dates]

# timedelta
date64 = np.datetime64('2018-02-24') # default is Day, explicit is 'D'. if we add 3 it will add 3 days.
dt64 = np.datetime64('2018-02-24', 'D')   # conversion: np.datetime_as_string(dt64)
tenminutes = np.timedelta64(10, 'm')      # 10 minutes
tenseconds = np.timedelta64(10, 's')      # 10 seconds
tennanoseconds = np.timedelta64(10, 'ns') # 10 nanoseconds

print('Add 10 days: ', dt64 + 10)
print('Add 10 minutes: ', dt64 + tenminutes)
print('Add 10 seconds: ', dt64 + tenseconds)
print('Add 10 nanoseconds: ', dt64 + tennanoseconds)

# some operations
a = np.array([[1,3,5],[2,4,6]])   # a.*?  will give all attributes on jupyter-notebook
b = np.c_[np.ones(len(a)), a]
c = np.vstack([b,range(4)])
d = np.r_[c,np.array([[10,20,30,40]])] # r_ needs same dimension
a1 = np.sort(np.array([4,8,2]); np.searchsorted(a,2)  # gives the index 0
diag = np.diag(d) # gives diagonal elements
diag_mat = np.diag([1,3,4]) # gives diagonal matrix
np.eye(3) # diagonal matrix with all 3

# vsplit and hsplit
x = np.array([[1,3,5,7],
              [2,4,6,8]])
np.vsplit(x,2) # gives first row as first split
np.hsplit(x,2) # gives first two columns as first split

# numpy attributes and methods
# type a. and hit tab
arr attributes: shape, size, ndim, dtype, argmax, argsort, astype, tofile, tostring, tolist,
universal functions ufuncs: add, subtract, multiply, maximum, minimum, etc
statistics: min, max, mean, std, var, cov, 
np.zeros, np.ones, np.zeros_like, np.ones_like
np.vstack, np.hstack, np.c_[], np.r_[], np.reshape, np.ravel, np.clip, np.array_split, 
np.sort(axis=0), np.argsort, np.diag, np.tile, np.vsplit, np.hsplit,

# array_split allows split into unequal part but np.split does not
x = np.arange(8.0)
parts = np.array_split(x, 3)

x = np.arange(2.0, 8.0) # [2., 3., 4., 5., 6., 7.]
idx = [3, 5, 6, 10]
parts = np.split(x, idx) # 234, 56, 7, empty

# isin
np.isin([1,3,7], [1,2,3,4,5]) # [ True,  True, False]

# set methods
np.setdiff1d([1,5,2,3], [2,8,10,11]) # [1, 3, 5]
np.setxor1d([1,5,2,3], [2,8,10,11]) # [ 1,  3,  5,  8, 10, 11]

# union, intersection
np.union1d([1,5,2,3], [2,8,10,11]) # [ 1,  2,  3,  5,  8, 10, 11]
np.intersect1d([1,5,2,3], [2,8,10,11]) # [ 1,  2,  3,  5,  8, 10, 11]

from functools import reduce
reduce(np.intersect1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2])) # [3]

# np.c_ needs 2d array but np.stack() works on 1d array
# add new column of sums
xyz = np.stack((x,y,z),axis=-1)
last = np.sum(xyz,axis=1,keepdims=True)  # or, last = np.sum(xyz,axis=1)
xyzz = np.c_[xyz,last]                   # then, xyzz = np.c_[xyz,last[None:,]]

# numpy printoptions
print(np.array_repr(x, precision=2, suppress_small=True))
with np.printoptions(precision=2,suppress=True):
    print(x)
```

# Create arrays
```python
a = np.fromstring('1 2') # Value Error
a = np.fromstring('1 2', sep=' ') # [1.0 2.0]
a = np.fromstring('1 2', dtype=int, sep=' ') # [1 2]
a1 = np.array('1 2'.split()) # ['1', '2']

b = np.array([x*x for x in range(5)])
iterable = (x*x for x in range(5))
b = np.fromiter(iterable, float)

# numpy index
for i in range(5):
    print([str(i) + str(j) for j in range(5)])
['00', '01', '02', '03', '04']
['10', '11', '12', '13', '14']
['20', '21', '22', '23', '24']
['30', '31', '32', '33', '34']
['40', '41', '42', '43', '44']


c = np.fromfunction(lambda i, j: i + j, (3, 4), dtype=int) # 012 123 234
[[0 1 2 3]  # i=0 and j = 0123 we have shape m,n so we must have i and j two parameters
 [1 2 3 4]  # i=1 and j = 0123
 [2 3 4 5]] # i=2 and j = 0123 Here, second row = 1+first row, since we have i + something

# just an example
def f(i,j):
    return sum((i+1)//k for k in np.arange(1,j+2))
d = np.fromfunction(np.vectorize(f), (5, 5), dtype=int)

array([[ 1,  1,  1,  1,  1],
       [ 2,  3,  3,  3,  3],
       [ 3,  4,  5,  5,  5],
       [ 4,  6,  7,  8,  8],
       [ 5,  7,  8,  9, 10]])
       
note:
10000
21000
31100
42110
52111 sum means 5,7,8,9,10

note: 
i,j = 0,4
[(i+1)//k for k in np.arange(1,j+2) ] # [1, 0, 0, 0, 0] here arange is 12345
```

# elements between given range
```python
a = np.array([1, 3, 5, 6, 9, 10, 14, 15, 56])
idx = np.argwhere((a>=6) & (a<=10)).ravel()  # [3, 4, 5]  # best method
idx = np.where((a>=6)& (a<=10))[0]  # [3, 4, 5]
idx = np.where(np.logical_and(a>=6, a<=10))  # [3, 4, 5]
idx = np.nonzero(np.logical_and(a>=6, a<=10))[0]
```

# new column of sum
```python
import numpy as np

x = np.array([1, 2, 3])
y = np.array([2,3,5])
z = np.array([2,3,4])

xyz = np.stack((x,y,z),axis=-1)
last = np.sum(xyz,axis=1)
xyzz = np.c_[xyz,last[None:,]]
```

# numpy functions
```python
## repeat repeats elements, while tile makes tiles of rows.
a = np.array([[1,2,3],
             [10,20,30]])

b = np.repeat(a,3,axis=1) or a.repeat(3,1) # two rows, 111 222 333 is first row.

np.tile(a,2) # 123 123 is first row.
```

# unfunc attributes
```python
a = np.array([1, 2, 3, 4])
np.add.at(a, [0, 1, 2, 2], 1)
print(a) # array([2, 3, 5, 4])
```

# string element of one array is substring element of another array
```python

# find substring in one element
np.core.char.find(['ab.c'],['ab'])        # array([0])  0th position matche started
np.core.char.find(['ab.cax'],['ax'])      # array([4])  4th position match started
np.core.char.find(['ab.cax'],['ax']) !=-1 # array([ True])   boolean result
np.core.char.find(['ab.c'],['abc'])       # -1 means seq not found

# find substring in multiple elements
np.core.defchararray.find(['abc','def'], 'bc') # array([ 1, -1]) 'bc' is found in 'abc' at 1th position

# count substrings
c = np.array(['aAaAaA', '  aA  ', 'abBABba'])
np.char.count(c, 'A')  # array([3, 1, 1])
np.char.count(c, 'aA') # array([3, 1, 0])
np.char.count(c, 'A', start=1, end=4) # array([2, 1, 1])
np.char.count(c, 'A', start=1, end=3) # array([1, 0, 0])
```

