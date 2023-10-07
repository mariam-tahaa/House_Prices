#!/usr/bin/env python
# coding: utf-8

# In[415]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


# In[416]:


#Read_Data
data = pd.read_csv("D:\\machine learning\\python_task\\house-prices-regression\\train.csv")


# In[417]:


data.head()


# In[274]:


data.shape


# In[275]:


data.describe()


# In[276]:


plt.hist(data["SalePrice"])


# In[277]:


corr = data.corr()
plt.subplots(figsize= (30,30))
sns.heatmap(corr, vmax= .7, square= True, annot= True)


# In[278]:


#zoomed heatmap with the 10 most affective values
cols = corr.nlargest(10, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, annot=True, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[279]:


#showing the plots between 'SalePrice' and correlated variables 
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(data[cols], height= 2.5)
plt.show()


# In[280]:


#dealing with missing data
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(10)


# In[281]:


data = data.drop(data.loc[data['Electrical'].isnull()].index)
data= missing_data.drop((missing_data[missing_data['Total'] > 1]), axis=1)
data.isnull().sum().max()


# In[287]:


#scaling the data
scaler= StandardScaler().fit_transform(data['SalePrice'][:,np.newaxis])
low_range = scaler[scaler[:,0].argsort()][:10]
high_range= scaler[scaler[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)


# In[ ]:


gliv = 'GrLivArea'
new_data= pd.concat([data['SalePrice'],data['GrLivArea']], axis= 1)
new_data.plot.scatter(x= 'GrLivArea', y= 'SalePrice')


# In[ ]:


#there is two points should be deleted to avoid missleading
data = data.drop(data[data['Id'] == 1299].index)
data = data.drop(data[data['Id'] == 524].index)


# In[ ]:


#same plot after deleting the points
gliv = 'GrLivArea'
new_data= pd.concat([data['SalePrice'],data['GrLivArea']], axis= 1)
new_data.plot.scatter(x= 'GrLivArea', y= 'SalePrice')


# In[ ]:


TotalB= 'TotalBsmtSF'
new_data= pd.concat([data['SalePrice'],data['TotalBsmtSF']], axis= 1)
new_data.plot.scatter(x= 'TotalBsmtSF', y= 'SalePrice')


# In[ ]:


from scipy import stats
from scipy.stats import norm
sns.distplot(data['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(data['SalePrice'], plot=plt)


# In[ ]:


#regularization the data
data['SalePrice']= np.log(data['SalePrice'])


# In[ ]:


#plot of saleprice after regularization
sns.distplot(data['SalePrice'], fit=norm);
plt.figure()
res = stats.probplot(data['SalePrice'], plot=plt)


# In[ ]:


#we will repeat the same steps in GrLivArea and TotalBsmtSF
data['GrLivArea']= np.log(data['GrLivArea'])


# In[ ]:


sns.distplot(data['GrLivArea'], fit= norm)
plt.figure()


# In[ ]:


sts= stats.probplot(data['GrLivArea'], plot= plt)


# In[288]:


#test homoscedasticity
plt.scatter(data['GrLivArea'], data['SalePrice'])


# In[261]:


#convert categorical variable into num
data = pd.get_dummies(data)


# In[292]:


#split the data into process in X and result in y
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
X_train= data[predictor_cols]
y_train= data.SalePrice


# In[293]:


print(X_train.head(10))


# In[294]:


print(y_train.head(10))


# In[266]:


#X_train= SimpleImputer(missing_values= 'nan', strategy= 'mean')


# In[366]:


random_forest = RandomForestRegressor(n_estimators= 30)
random_forest.fit(X_train, y_train)


# In[418]:


test_data= pd.read_csv("D:\\machine learning\\python_task\\house-prices-regression\\test.csv")


# In[419]:


test_data.shape


# In[420]:


X_test= test_data[predictor_cols]


# In[421]:


y_test= random_forest.predict(X_test)


# In[422]:


print(y_test)


# In[372]:


random_forest.score(X_train, y_train)


# In[405]:


random_forest.score(X_test, y_test)

