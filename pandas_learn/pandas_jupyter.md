Table of Contents
=================
   * [Side by side dataframes](#side-by-side-dataframes)
   * [Display dataframes with headers](#display-dataframes-with-headers)
   * [Display dataframes side to side with given spacing](#display-dataframes-side-to-side-with-given-spacing)
   * [Add paths to jupyter-notebook](#add-paths-to-jupyter-notebook)

# Side by side dataframes
```python
from IPython.display import display_html
def displayss(*args):
    ''' Display pandas dataframe side by side in Jupyter notebook.
    Usage: dispalyss(df1, df2)
    '''
    html_str=''
    for df in args:
        if type(df) == pd.core.series.Series:
            df = df.to_frame()
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
```

# Display dataframes with headers
```python
class displayh(object):
    """Display HTML representation of multiple objects with headers

    Usage: displayh('df', "df.groupby('col_0').std()")
    """
    template = """<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
    </div>"""
    def __init__(self, *args):
        self.args = args
        
    def _repr_html_(self):
        return '\n'.join(self.template.format(a, eval(a)._repr_html_())
                         for a in self.args)
    
    def __repr__(self):
        return '\n\n'.join(a + '\n' + repr(eval(a))
                           for a in self.args)
```

# Display dataframes side to side with given spacing
```python
def display_frames(frames, num_spaces=0):
    from IPython.display import display_html
    t_style = '<table style="display: inline;"'
    tables_html = [df.to_html().replace('<table', t_style) for df in frames]

    space = '&nbsp;' * num_spaces
    display_html(space.join(tables_html), raw=True)

display_frames([df1,df2], 30)
```

# Add paths to jupyter-notebook
```python
%load_ext autoreload
%autoreload 2

import sys
sys.path.append('../..')

# Display all cell outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
```
