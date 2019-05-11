# Dataset links
http://www.randomservices.org/random/



# Practice dataset
#==============================================================================
```python
## using seaborn
##------------------------------
import seaborn as sns
iris = sns.load_dataset('iris')

print(sns.get_dataset_names())
'anscombe', 'attention', 'brain_networks', 'car_crashes', 'diamonds', 'dots', 'exercise', 'flights', 'fmri', 
'gammas', 'iris', 'mpg', 'planets', 'tips', 'titanic'


## using statsmodels.api
##------------------------------
import statsmodels.api as sm

## built-ins
co2 = sm.datasets.co2  # time series data for co2
print(co2.DESCRLONG)
df = sm.datasets.co2.load_pandas().data

**
co2 = sm.datasets.co2.load_pandas() # statsmodels data object
co2.names # ('date', 'co2')
co2.data # pandas dataframe
co2.raw_data # numpy rec array without names
co2.values() # numpy rec array with names
co2.data.values # numpy array from pandad dataframe

**
## dataframe from numpy records object
## NOTE: Compare the dataframe with the one directly 
##       downloaded from below link:
## https://raw.githubusercontent.com/statsmodels/statsmodels/master/statsmodels/datasets/co2/co2.csv
##
df = pd.DataFrame.from_records(sm.datasets.co2.load().data)
df['date'] = df.date.apply(lambda x: x.decode('utf-8'))
df['date'] = pd.to_datetime(df.date, format='%Y%m%d')
df['co2'] = pd.to_numeric(df.co2, errors='coerce')
df = df.set_index('date')
df.head()
==>  aliter:
df = pd.read_csv('co2.csv',index_col=0)
df.index = pd.to_datetime(df.index,format='%Y%m%d')
df.co2.sum(), df2.co2.bfill().sum()

## using rdataset
iris = sm.datasets.get_rdataset('iris').data
dataset_iris = sm.datasets.get_rdataset(dataname='iris', package='datasets')
print(dataset_iris.__doc__)

## using sklearn
##------------------------------
from sklearn import datasets # sklearn.datasets does not work
iris = datasets.load_iris() 
print(datasets.load_iris().DESCR)
'''
attributes:
dir(datasets.load_iris()) ==> .data .feature_names .target .target_names .filename .DESCR
'''
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

[ i for i in dir(datasets) if 'load_' in i]

'load_boston', 'load_breast_cancer','load_diabetes', 'load_digits', 'load_files'
'load_iris', 'load_linnerud','load_mlcomp','load_sample_image','load_sample_images',
'load_svmlight_file', 'load_svmlight_files', 'load_wine'
```
Statsmodels provides access to 1173 datasets from the [Rdatasets project](https://github.com/vincentarelbundock/Rdatasets).
Also, it has following built in datasets:
- anes96:  [American National Election Survey 1996](http://www.statsmodels.org/dev/datasets/generated/anes96.html)
- cancer: [Breast Cancer Data](http://www.statsmodels.org/dev/datasets/generated/cancer.html)
- ccard: [Bill Greene’s credit scoring data.](http://www.statsmodels.org/dev/datasets/generated/ccard.html)
- china_smoking: [Smoking and lung cancer in eight cities in China.](http://www.statsmodels.org/dev/datasets/generated/china_smoking.html)
- co2: [Mauna Loa Weekly Atmospheric CO2 Data](http://www.statsmodels.org/dev/datasets/generated/co2.html)
- committee: [First 100 days of the US House of Representatives 1995](http://www.statsmodels.org/dev/datasets/generated/committee.html)
- copper: [World Copper Market 1951-1975 Dataset](http://www.statsmodels.org/dev/datasets/generated/copper.html)	
- cpunish: [US Capital Punishment dataset](http://www.statsmodels.org/dev/datasets/generated/cpunish.html)
- elnino: [El Nino - Sea Surface Temperatures	](http://www.statsmodels.org/dev/datasets/generated/elnino.html)
- engel: [Engel (1857) food expenditure data](http://www.statsmodels.org/dev/datasets/generated/engel.html)
- fair: [Affairs dataset](http://www.statsmodels.org/dev/datasets/generated/fair.html)
- fertility: [World Bank Fertility Data](http://www.statsmodels.org/dev/datasets/generated/fertility.html)
- grunfeld: [Grunfeld (1950) Investment Data](http://www.statsmodels.org/dev/datasets/generated/grunfeld.html)
- heart: [Transplant Survival Data](http://www.statsmodels.org/dev/datasets/generated/heart.html)
- longley: [Longley dataset](http://www.statsmodels.org/dev/datasets/generated/longley.html)
- macrodata: [United States Macroeconomic data](http://www.statsmodels.org/dev/datasets/generated/macrodata.html)
- modechoice: [Travel Mode Choice](http://www.statsmodels.org/dev/datasets/generated/modechoice.html)
- nile: [Nile River flows at Ashwan 1871-1970](http://www.statsmodels.org/dev/datasets/generated/nile.html)
- randhie: [RAND Health Insurance Experiment Data](http://www.statsmodels.org/dev/datasets/generated/randhie.html)
- scotland: [Taxation Powers Vote for the Scottish Parliamant 1997](http://www.statsmodels.org/dev/datasets/generated/scotland.html)
- spector: [Spector and Mazzeo (1980) - Program Effectiveness Data](http://www.statsmodels.org/dev/datasets/generated/spector.html)
- stackloss: [Stack loss data](http://www.statsmodels.org/dev/datasets/generated/stackloss.html)
- star98: [Star98 Educational Dataset](http://www.statsmodels.org/dev/datasets/generated/star98.html)
- statecrime: [Statewide Crime Data 2009](http://www.statsmodels.org/dev/datasets/generated/statecrime.html)
- strikes: [U.S. Strike Duration Data](http://www.statsmodels.org/dev/datasets/generated/strikes.html)
- sunspots: [Yearly sunspots data 1700-2008](http://www.statsmodels.org/dev/datasets/generated/sunspots.html)
