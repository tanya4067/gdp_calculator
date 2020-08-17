import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_array
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import math
reg = linear_model.LinearRegression()

#from sklearn.externals import joblib


df=pd.read_csv('gdp1.csv')
#print(plt.scatter(df['year'],df['gdp']))
#print(df)
medians=math.floor(df.Unemployement_Rate.median())
df.Unemployement_Rate.fillna(medians,inplace=True)
y=df[['gdp','Unemployement_Rate','Inflation_Rate']]
X=df.drop(['gdp','Unemployement_Rate','Inflation_Rate'],axis=1)

#X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=123)

#print(medians_Unemployement_Rate)


reg.fit(X,y)
t=reg.coef_
#print(t)
t1=reg.intercept_
#print(t1)
print('Enter the values')
print('Enter the year when you want to start your business')
i=float(input())

p=reg.predict([[i]])
a=p.tolist()

print('VALUES ARE')
print('gdp rate:',a[0][0],'%')
print('inflation rate:',a[0][1],'%')
print('Unemployement rate:',a[0][2],'%')