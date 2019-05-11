Table of Contents
=================
   * [Create datetime pandas](#create-datetime-pandas)
   * [Convert number to time](#convert-number-to-time)
   * [Create timedelta numpy](#create-timedelta-numpy)
   * [Create timedelta from integer](#create-timedelta-from-integer)
   * [Resample weekly from column](#resample-weekly-from-column)
   * [dateutil.parse parse](#dateutilparse-parse)
   * [plot timedelta xticklabels](#plot-timedelta-xticklabels)
   * [datetime strftime](#datetime-strftime)

# Create datetime pandas
```python
import pandas as pd
from datetime import datetime
z = pd.DataFrame({'a':[datetime.strptime('20150101', '%Y%m%d')],
                  'b':[datetime.strptime('20140601', '%Y%m%d')]})

z.a - z.b # 214 days dtype: timedelta64[ns]
```

# Convert number to time
```python
df = pd.DataFrame({'a': [0,5,10,100,105,2000,2355]})
df['a_date'] = pd.to_datetime(df.a.astype(str).str.rjust(4,'0'),format='%H%M').dt.time
df['a_date'] = pd.to_datetime(df['a'].astype(str).str.zfill(4), format = '%H%M').dt.time
```

# Create timedelta numpy
```python
x = np.array([10,20,30])
x = x.astype('timedelta64[m]')
x.astype('timedelta64[s]')  # array([ 600, 1200, 1800], dtype='timedelta64[s]')


## Example 2
y = np.arange(1,5,dtype='timedelta64[m]') # [ 1 2 3 4 ]
y.astype('timedelta64[s]')  # array([ 60, 120, 180, 240], dtype='timedelta64[s]')

## arithmetic
np.timedelta64(1, 'm') + np.timedelta64(62,'s') # numpy.timedelta64(122,'s')
```

# Create timedelta from integer
```python
import numpy as np
import pandas as pd

df = pd.DataFrame({'a': [1.0,100,205,1050,2359,2400,np.nan]})

# first make 2400 to zero
df.a.replace(2400,0.0,inplace=True)

# faster method
v = np.divmod(df.a.values,100)
df['a_timedelta1'] = pd.to_timedelta(v[0], unit='h') + pd.to_timedelta(v[1], unit='m')

# easier method
df['a_timedelta'] = pd.to_timedelta(df.a//100, unit='h') + pd.to_timedelta(df.a%100, unit='m')


# column of minutes
df['a_mins'] = df.a_timedelta.dt.seconds / 60

# timedelta as string
df['a_timedelta_str'] = df.a_timedelta.apply(str)

print(df)

        a a_timedelta1 a_timedelta  a_mins  a_timedelta_str
0     1.0     00:01:00    00:01:00     1.0  0 days 00:01:00
1   100.0     01:00:00    01:00:00    60.0  0 days 01:00:00
2   205.0     02:05:00    02:05:00   125.0  0 days 02:05:00
3  1050.0     10:50:00    10:50:00   650.0  0 days 10:50:00
4  2359.0     23:59:00    23:59:00  1439.0  0 days 23:59:00
5     0.0     00:00:00    00:00:00     0.0  0 days 00:00:00
6     NaN          NaT         NaT     NaN              NaT

```

# Resample weekly from column
```python
import numpy as np
import pandas as pd
np.random.seed(2019)

dates = pd.date_range('2018-01-01', '2018-1-21', freq='D')
colors = np.random.randint(0,3, len(dates))
values = np.random.normal(10,20, len(dates))

df = pd.DataFrame({'dates': dates,
                   'colors': colors,
                   'values': values})

df.groupby('colors').resample(rule='W', on='dates')['values'].sum().reset_index()
```

# dateutil.parse parse
```python
import numpy as np
import pandas as pd
from dateutil.parser import parse

ser = pd.Series(['01 Jan 2010', '02-02-2011', '20120303', '2013/04/04', '2014-05-05', '2015-06-06T12:20'])

s = pd.to_datetime(ser)
s = ser.apply(parse)
```

# plot timedelta xticklabels
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import text as TEXT


df = pd.DataFrame({'d': [np.timedelta64(5,'h'), np.timedelta64(7,'h')],
                 'v': [100,200]})

ax = df.set_index('d').plot.bar()
ax.set_xticklabels([l.get_text().split()[-1] for l in ax.get_xticklabels()]) # this removes 0 days from xticklabels
```

# datetime strftime
![](images/datetime_strftime1.png)
![](images/datetime_strftime2.png)
![](images/datetime_strftime3.png)

