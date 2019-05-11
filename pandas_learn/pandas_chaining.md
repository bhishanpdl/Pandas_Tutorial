Table of Contents
=================
   * [Compare avergae delay of a carriers flights to the avg of others in same route](#compare-avergae-delay-of-a-carriers-flights-to-the-avg-of-others-in-same-route)
   * [Chaining example](#chaining-example)
   * [Another chaining example](#another-chaining-example)

# Compare avergae delay of a carriers flights to the avg of others in same route
```python
import numpy as np
import pandas as pd

flights = pd.read_csv('https://github.com/bhishanpdl/Datasets/blob/master/nycflights13.csv?raw=true')

(# filter arr delay null values
flights.loc[flights.arr_delay.notna()]
     .groupby(['origin','dest','carrier'])['arr_delay']
     .agg(['sum','count'])
     .reset_index()
     .rename(columns={'count': 'flights', 'sum': 'arr_delay'})

    # New columns for total arr_delay and total flights
    # after groupby origin dest
    .assign(
          arr_delay_total = lambda x: (x.groupby(['origin','dest'])
                             ['arr_delay'].transform('sum')),
          flights_total = lambda x: (x.groupby(['origin','dest'])
                             ['flights'].transform('sum'))
          )

    # average delay of each carrier - average delay of other carriers
    .eval("""
        arr_delay_others = (arr_delay_total - arr_delay) \
                           / (flights_total - flights)
        arr_delay_mean = arr_delay / flights
        arr_delay_diff = arr_delay_mean - arr_delay_others
                   """)

    .dropna(subset=['arr_delay_diff'])
        # average over all airports it flies to
        .groupby('carrier',as_index=False)['arr_delay_diff'].mean()
        .sort_values('arr_delay_diff',ascending=False)
)

```

# Chaining example
```python
# For each destination, compute the total minutes of delay.
# For each flight, compute the proportion of the total delay
# for its destination.

flights = pd.read_csv('https://github.com/bhishanpdl/Datasets/blob/master/nycflights13.csv?raw=true')
print(flights.shape)

df = (flights.query("arr_delay > 0")

    .assign(# new column using transform
          arr_delay_total = lambda x: (x.groupby(['dest'])
                             ['arr_delay'].transform('sum')),

          # then just divide two columns
          arr_delay_prop = lambda x: x.eval("arr_delay / arr_delay_total")
            )
     .filter(items=['dest', 'month', 'day', 'dep_time',
                       'carrier', 'flight', 'arr_delay',
                       'arr_delay_prop'])
      .sort_values(['dest','arr_delay_prop'],ascending=[True,False])
      )


# R-equivalent
# note: group_by mutate keeps all rows.
df = flights %>%
  filter(arr_delay > 0) %>%
  group_by(dest) %>%
  mutate(
    arr_delay_total = sum(arr_delay),
    arr_delay_prop = arr_delay / arr_delay_total
  ) %>%
  select(
    dest, month, day, dep_time, carrier, flight,
    arr_delay, arr_delay_prop
  ) %>%
  arrange(dest, desc(arr_delay_prop))

```


# Another chaining example
```R
standardized_flights <- flights %>%
  filter(!is.na(air_time)) %>%
  group_by(dest, origin) %>%
  mutate(
    air_time_mean = mean(air_time),
    air_time_sd = sd(air_time),
    n = n()
  ) %>%
  ungroup() %>%
  mutate(air_time_standard = (air_time - air_time_mean) / (air_time_sd + 1))
```
```python
df = (flights.dropna(subset=['air_time'],how='any')
      .assign(
      air_time_mean = lambda dff: dff.groupby(['dest','origin'])
                      ['air_time'].transform('mean'),
      air_time_sd = lambda dff: dff.groupby(['dest','origin'])
                      ['air_time'].transform('std'),
      air_time_count = lambda dff: dff.groupby(['dest','origin'])
                      ['air_time'].transform('count')
             )
      .eval("""
      air_time_standard = (air_time - air_time_mean) / (air_time_sd + 1)
      """)
     )
```

