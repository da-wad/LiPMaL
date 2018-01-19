# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 08:37:17 2018

@author: andd
"""

#Data analysis and regression of lithofacies vs well logs
#
#from datetime import datetime
#import scipy as sp
#from sklearn.linear_model import LinearRegression
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.ensemble import AdaBoostRegressor
#from sklearn.neural_network import MLPRegressor
#from sklearn.metrics import roc_curve
#from sklearn.metrics import auc
#from sklearn.model_selection import GridSearchCV
#import matplotlib
#import seaborn as sns
#import re
#from sklearn.metrics import mean_squared_error
#from IPython.display import display
#from sklearn.feature_extraction import DictVectorizer

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

import sklearn

from scipy.fftpack import fft, fftfreq
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.signal import detrend




############################################################
# PART 1: Read data and prepare dataframe for analysis
# Can be omitted by loading F:\Statoil Documents\Python\Dataframe_DW.txt
############################################################
df = pd.read_excel('F:\\Statoil Documents\\Matlab\\LiPMaL_database_DW_final.xlsx', decimal =",")

#Remove all columns that are not relevant for the research
df = df.drop(['Country','Basin','Age','Dominant lithology','Sand distribution','Dominant sandstone grainsize','Vp','Saturation'],1)
Y = pd.factorize(df['Facies type'])[0]
df = pd.get_dummies(df, columns = ['Facies type'], prefix = 'Facies')

# Calculate properties for each individual Facies recording
groups = df.groupby('Facies')
facies_var = {}
cnt=0
for fname, group in groups:
    grvar =  np.var(group['Gamma Ray'].dropna())
    npstd =  np.nanstd(group['Neutron porosity'].dropna())
    grstd = np.nanstd(group['Gamma Ray'].dropna())
    gr_absdiff = group['Gamma Ray'].diff().dropna().abs().mean()
    
    #Calculate wave number
    y = group['Gamma Ray'].dropna()
    x = group['Tvdml'].dropna()
    y2 = detrend(y, type = 'linear')
    label = fname
    
    n = x.size
    fy = fft(y2, n)
    freq = fftfreq(y2.size, np.mean(np.diff(x)))
    pctile = np.percentile(np.abs(fy[:int(n/2)]),90)
#    plt.plot(cnt,pctile, 'g.', label = fname)
    
    #Neutron density
    Npor = group['Neutron porosity']
    Npor[Npor>0.6] = 0.6
    Rho = group['Density']
    Rho[Rho>2.7]=2.7
    
    Pordens = (2.65-Rho)/(2.65-1)
    
    NeuDen = np.nanstd([Pordens - Npor])
         
    facies_var[fname] = [npstd, grstd, gr_absdiff, NeuDen, pctile]
    cnt +=1
#    print(np.percentile(np.abs(fy[:int(n/2)]),90))


df2 = df
df2 = pd.DataFrame.from_dict(facies_var, orient = 'index')
df2.columns=['Gr std','Neutron std','Gr_absdiff','Neutron Density','Frequency P90']


df2 = df.join(df2, on='Facies')

# Save or load the dataframe 
#df2.to_csv('F:\Statoil Documents\Python\Dataframe_DW.txt')
#df2 = pd.read_csv('F:\Statoil Documents\Python\Dataframe_DW.txt')

# Group data by Facies and prepare for ML

df3 = df2.drop_duplicates(subset = 'Gr std', keep = 'first').groupby(['Facies']).apply(lambda x: x.set_index('Facies'))
df3 = df3.dropna()

# ML predictions

features = ['Gr std','Gr_absdiff','Neutron Density','Frequency P90']

X    = df3[features].values
Facies_name = ['Facies_Overbank','Facies_Lobe_Complex','Facies_Lobe','Facies_Fan_System','Facies_Drape','Facies_Channel_System','Facies_Channel_Complex']

# Test out linear regression
Accuracy_lin = pd.Series(np.ones_like(Facies_name),index = Facies_name)
cnt=0
for name in Facies_name:
    y    = df3[name].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=123)
    
    lg = sklearn.linear_model.LinearRegression()
    # Train Linear regression
    lg.fit(X_train,y_train)

    # Predict values
    y_train_pred_lin = lg.predict(X_train)
    y_test_pred_lin  = lg.predict(X_test)

    print('Score linear regression... train:',lg.score(X_train,y_train),', test:',lg.score(X_test,y_test))

    pred_lin = y_test_pred_lin
    pred_lin[pred_lin<.5]=0
    pred_lin[pred_lin>=.5]=1
    residual_lin   = y_test - pred_lin
    
    num_corr = np.shape(np.where(residual_lin == 0))
    pct_corr = np.float(np.around(np.true_divide(num_corr[1],len(residual_lin))*100, 1))
    
    plt.figure()
    plt.hist(residual_lin,label=name)
    plt.xlabel('Residual for classification of ' + name)
    plt.title(name + ': Linear model, Accuracy = '+str(pct_corr) + ' %')
    Accuracy_lin[name]=pct_corr

plt.figure()
Accuracy_lin.plot(kind = 'bar')
plt.ylabel('Percentage of accurate prediction')
plt.title('Linear Regression model, Average accuracy = ' + str(np.float(np.around(np.mean(Accuracy_lin),1)))+ ' %')
# Test out Random Forest
Accuracy_rf = pd.Series(np.ones_like(Facies_name),index = Facies_name)
cnt=0
for name in Facies_name:
    y    = df3[name].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=123)
    
    lg = lg = RandomForestRegressor()
    # Train Linear regression
    lg.fit(X_train,y_train)

    # Predict values
    y_train_pred_rf = lg.predict(X_train)
    y_test_pred_rf  = lg.predict(X_test)

    print('Score linear regression... train:',lg.score(X_train,y_train),', test:',lg.score(X_test,y_test))

    pred_rf = y_test_pred_rf
    pred_rf[pred_rf<.5]=0
    pred_rf[pred_rf>=.5]=1
    residual_rf   = y_test - pred_rf
    
    num_corr = np.shape(np.where(residual_rf == 0))
    pct_corr = np.float(np.around(np.true_divide(num_corr[1],len(residual_rf))*100, 1))
    
    plt.figure()
    plt.hist(residual_rf,label=name)
    plt.xlabel('Residual for classification of ' + name)
    plt.title(name + ': Linear model, Accuracy = '+str(pct_corr) + ' %')
    
    Accuracy_rf[name]=pct_corr
    cnt+=1
     
Accuracy_rf.plot(kind = 'bar')

plt.figure()
Accuracy_rf.plot(kind = 'bar')
plt.ylabel('Percentage of accurate prediction')
plt.title('Random Forest, Average accuracy = ' + str(np.float(np.around(np.mean(Accuracy_rf),1)))+ ' %')