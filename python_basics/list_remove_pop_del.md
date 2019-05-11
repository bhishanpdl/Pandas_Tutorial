# python list remove pop del
```python
a = [0, 2, 3, 2]

# remove is inplace
a = [1,2,2,2,3]
a.remove(3)
a # [1, 2, 2, 2]

# pop returns value, but does not support range
a = [1,2,2,2,3]
a.pop(3) # pop returns value, e.g. 2
a # [1, 2, 2, 3]

# del returns None but support range
a = [1,2,2,2,3]
del a[3] # delete returns nothing
a # [1, 2, 2, 3]  same as pop(3)

a = [1,2,2,2,3]
del a[3:] # delete returns nothing and supports range
a # [1, 2, 2]
```
