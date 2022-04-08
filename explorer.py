# To access the MySQL server, for data acquisition
import os 
from env import get_db_url

# Bare necessities 
import pandas as pd 
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# feature selection imports
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

# import scaling methods
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer
import scipy
from scipy import stats

# Clustering necessities
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing, cluster


# Modeling Methodology
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import explained_variance_score

# Personal Effects 
import wrangle as w
import mitosheet
import folium
from folium.plugins import FastMarkerCluster
from describe import nulls
import branca.colormap as cm

def plot_continuous_duo(df, x, y):
    '''
        Create line and scatter plots along with a regression line for two 
        continuous variables. User provides a Pandas DataFrame and strings
        capturing the column names to be used for the independent variable, x,
        and dependent variable, y.  
    '''

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (4, 12))
    mean = df[y].mean()

    sns.lineplot(data = df, x = x, y = y, ax = ax[0])
    ax[0].axhline(mean, ls='--', color='grey')

    sns.scatterplot(data = df, x = x, y = y, ax = ax[1], alpha = 0.3, color = 'blue')
    ax[1].axhline(mean, ls='--', color='grey')

    plt.show()


    
def california(train):
    # create a map and set the center as the Greater Los Angeles Area
    folium_map = folium.Map(location=[34.0522, -118.2437])
    
    # use the following line if you want to run the map as is. 
    #folium.Map()
    
    # Cluster the residences 
    FastMarkerCluster(data=list(zip(train['latitude'].values, train['longitude'].values))).add_to(folium_map)
    
    # Add the cluster as a layer
    folium.LayerControl().add_to(folium_map)
    folium_map
    
    # create a LinearColorMap and assign colors, vmin, and vmax
    # the colormap will show green for new homes all the way up to red for homes in the 3rd quartile and beyond
    colormap = cm.LinearColormap(colors=['green', 'yellow', 'red'], vmin=1, vmax=120)
    
    # create our map again.  This time I am using a different tileset for a new look
    m = folium_map
    
    # Same as before... go through each home in set, make circle, and add to map.
    # This time we add a color using price and the colormap object
    for i in range(len(train)):
        folium.Circle(
            location=[train.iloc[i]['latitude'], train.iloc[i]['longitude']],
            radius=10,
            fill=True,
            color=colormap(train.iloc[i]['age']),
            fill_opacity=0.2
        ).add_to(m)
    
    # the following line adds the scale directly to our map
    m.add_child(colormap)
    folium_map = m
    
    return folium_map

def landquest(train):
    
    print('First off, Mr. Warning Message/Error from LoZ 2, YOU\'RE NOT MY REAL DAD')
    
    fig, axes = plt.subplots(1, 3, figsize=(16,8))
    fig.suptitle('Similar Distributions Among Lot Size, County-Level Use Code, and Land Tax')
    
    sns.histplot(ax=axes[0], x=train.lotsizesquarefeet)
    axes[0].set_title('Size of the Lot |sqft|')
    
    sns.histplot(ax=axes[1], x=train.propertycountylandusecode)
    axes[1].set_title('Land Use Permissions Codified')
    
    sns.histplot(ax=axes[2], x=train.landtaxvaluedollarcnt)
    axes[2].set_title('Land Tax')
    plt.show()
    
    print('''             
                                The vast majority of land is well under 1/5 of an acre
                             With a spike in low costs, and low area. 
                             
                                Each of the four-digit land-use codes beginning 
                                with zero belong to Los Angeles, hence their 
                                 over abundance of representation.
          '''
         )
                
    print('---------------------------------------------------------------\
    ---------------------------------------------------------------------------')
    
def costpersqft(train):
    fig, axes = plt.subplots(1, 3, figsize=(16,8))
    fig.suptitle('Similar Distributions Between Calculatedfinishedsqfeet and Structuretaxvalue')
    
    sns.histplot(ax=axes[0], x=train.fiscal_quarter)
    axes[0].set_title('Transaction Dates by Fiscal Quarter. \n Q4 not present in DataFrame.')
    
    sns.histplot(ax=axes[1], x=train.land_cost_per_sqft )
    axes[1].set_title('Cost of Land per Square Foot.')
    
    sns.histplot(ax=axes[2], x=train.structure_cost_per_sqft)
    axes[2].set_title('Cost of Structure per Square Foot.')
    plt.show()
    
    print('''
                            Unsurprisingly, structural costs per sq ft blow land costs out of the water.
                            This year, some estimates suggest the median square foot of land in Los Angeles
                            is worth between $11 and $40 Moreover, the cost. Quite a range for median. Pure speculation.
                            Nevertheless, I'll check. Moreover, some projections suggest the cost for the structure ranges
                            from $100-$400 this year, we can also check if there is consistency with 2017's data 
                            by examining the margins more closely.
         
          ''')
    
    print('---------------------------------------------------------------------\
    ---------------------------------------------------------------------')