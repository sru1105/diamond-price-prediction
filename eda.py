#IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import missingno
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import learning_curve

#IMPORTING DATASETS
df = pd.read_csv('../input/diamonds-prices/Diamonds Prices2022.csv')
print('Dataset imported successfully\n')

#NO OF ROWS AND COLUMNS
print('Dataset contains {} rows and {} columns.\n'.format(df.shape[0],df.shape[1]))
df.info()
print('\n')
df.head()

#NUBER OF NULL VALUES

df.isnull().sum()

#SEPARATION OF NUMERICAL AND CATEGORICAL COLUMNS
num = df.select_dtypes('number')
cat = df.select_dtypes('O')
for c in cat.columns:
    print(cat[c].unique())
for n in num.columns:
    print(num[n].unique())

#DESCRIBING THE DATA
df.describe().transpose()


#Visualising categorical distributions
i = 1
plt.figure(figsize=(30,20))
for x in cat.columns:
    plt.subplot(3, 3, i)
    sns.countplot(x=df[x].sort_values())
    i+=1

#handling outliers
i = 1
plt.figure(figsize=(25, 14))
for y in num.columns:
    plt.subplot(3, 3, i)
    sns.boxplot(x=df[y])
    i+=1


#NUMBER OF OUTLIERS 
df.isna().sum()
df.dtypes
df.describe()
lower_bound = []
upper_bound = []
numerics = ['float64','int64']
outliers = 0
df1 = df.select_dtypes(include=numerics)
for col in df1.columns.values:
        Q1=(np.percentile(df1[col],25))
        Q3=(np.percentile(df1[col],75))
        iqr=(Q3-Q1)
        upper_bound = (Q3 +(1.5*iqr))
        lower_bound = (Q1 -(1.5*iqr)) 
        for val in df1[col]:
            if(val < lower_bound or val > upper_bound):
                outliers += 1
outliers


# No inconsistencies - inferred from unique values of each column

dup=df[df.duplicated()]
print("Duplicate rows: ",dup)

#correlation - pearson coefficient, spearman coefficient, kendall coefficient
df.corr()

df.corr(method='spearman')

df.corr(method='kendall')

train, test = train_test_split(df, test_size=0.3)


#Dimensionality Reduction - not required as only 11 columns
