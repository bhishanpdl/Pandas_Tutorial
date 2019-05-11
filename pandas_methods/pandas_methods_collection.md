# Pandas dataframe methods
```python
abs            divide             iterrows          quantile        tail        
add            dot                itertuples        query           take        
add_prefix     drop               ix                radd            to_clipboard
add_suffix     drop_duplicates    join              rank            to_csv      
agg            dropna             keys              rdiv            to_dense    
aggregate      duplicated         kurt              reindex         to_dict     
align          eq                 kurtosis          reindex_axis    to_excel    
all            equals             last              reindex_like    to_feather  
any            eval               last_valid_index  rename          to_gbq      
append         ewm                le                rename_axis     to_hdf      
apply          expanding          loc               reorder_levels  to_html     
applymap       ffill              lookup            replace         to_json     
as_matrix      fillna             lt                resample        to_latex    
asfreq         filter             mad               reset_index     to_msgpack  
asof           first              mask              rfloordiv       to_panel    
assign         first_valid_index  max               rmod            to_parquet  
astype         floordiv           mean              rmul            to_period   
at             from_dict          median            rolling         to_pickle   
at_time        from_records       melt              round           to_records  
between_time   ge                 memory_usage      rpow            to_sparse   
bfill          get                merge             rsub            to_sql      
bool           get_dtype_counts   min               rtruediv        to_stata    
boxplot        get_ftype_counts   mod               sample          to_string   
clip           get_values         mode              select          to_timestamp
clip_lower     groupby            mul               select_dtypes   to_xarray   
clip_upper     gt                 multiply          sem             transform   
combine        head               ne                set_axis        transpose   
combine_first  hist               nlargest          set_index       truediv     
compound       iat                notna             shift           truncate    
copy           idxmax             notnull           skew            tshift      
corr           idxmin             nsmallest         slice_shift     tz_convert  
corrwith       iloc               nunique           sort_index      tz_localize 
count          infer_objects      pct_change        sort_values     unstack     
cov            info               pipe              squeeze         update      
cummax         insert             pivot             stack           var         
cummin         interpolate        pivot_table       std             where       
cumprod        isin               plot              sub             xs          
cumsum         isna               pop               subtract      
describe       isnull             pow               sum           
diff           items              prod              swapaxes      
div            iteritems          product           swaplevel     
```

# Pandas dataframe grouby methods
```python
import IPython
import numpy as np
import pandas as pd
import seaborn as sns

df = sns.load_dataset('tips')
grouped = df.groupby('tip')

meth = [method_name for method_name in dir(grouped)
    if callable(getattr(grouped, method_name)) 
    & ~method_name.startswith('_')]


print(IPython.utils.text.columnize(meth))
agg        corr      cumsum     get_group  mean     pct_change  sem    transform
aggregate  corrwith  describe   head       median   pipe        shift  tshift   
all        count     diff       hist       min      plot        size   var      
any        cov       expanding  idxmax     ngroup   prod        skew 
apply      cumcount  ffill      idxmin     nth      quantile    std  
backfill   cummax    fillna     last       nunique  rank        sum  
bfill      cummin    filter     mad        ohlc     resample    tail 
boxplot    cumprod   first      max        pad      rolling     take 
```
