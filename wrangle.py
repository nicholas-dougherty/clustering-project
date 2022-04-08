from env import host, username, password, get_db_url
import os
import pandas as pd 
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from datetime import date

def get_exploration_data():
    train, validate, test = wrangle_zillow()
    return train


def wrangle_zillow_anew(target):
    '''
    '''
    
    # Cannot stratify by logerror
    train, validate, test = prep_zillow_splitter(acquire_zillow_data())
    
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    # Change series into data frame for y 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)
    
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test
    


def wrangle_zillow():
    df = prep_zillow(acquire_zillow_data())
    
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    return train, validate, test


def acquire_zillow_data(use_cache=True):
    '''
    This function returns a snippet of zillow's database as a Pandas DataFrame. 
    When this SQL data is cached and extant in the os directory path, return the data as read into a df. 
    If csv is unavailable, aquisition proceeds regardless,
    reading the queried database elements into a dataframe, creating a cached csv file
    and lastly returning the dataframe for some sweet data science perusal.
    '''

    # If the cached parameter is True, read the csv file on disk in the same folder as this file 
    if os.path.exists('zillow.csv') and use_cache:
        print('Using cached CSV')
        return pd.read_csv('zillow.csv', dtype={'buildingclassdesc': 'str', 'propertyzoningdesc': 'str'})

    # When there's no cached csv, read the following query from Codeup's SQL database.
    print('CSV not detected.')
    print('Acquiring data from SQL database instead.')
    df = pd.read_sql(
        '''
 SELECT
    prop.*,
    predictions_2017.logerror,
    predictions_2017.transactiondate,
    air.airconditioningdesc,
    arch.architecturalstyledesc,
    build.buildingclassdesc,
    heat.heatingorsystemdesc,
    landuse.propertylandusedesc,
    story.storydesc,
    construct.typeconstructiondesc
FROM properties_2017 prop
JOIN (
    SELECT parcelid, MAX(transactiondate) AS max_transactiondate
    FROM predictions_2017
    GROUP BY parcelid
) pred USING(parcelid)
JOIN predictions_2017 ON pred.parcelid = predictions_2017.parcelid
                      AND pred.max_transactiondate = predictions_2017.transactiondate
LEFT JOIN airconditioningtype air USING (airconditioningtypeid)
LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid)
LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid)
LEFT JOIN storytype story USING (storytypeid)
LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid)
WHERE prop.latitude IS NOT NULL
  AND prop.longitude IS NOT NULL
  AND transactiondate <= '2017-12-31';             
        '''
                    , get_db_url('zillow'))
    
    df.propertyzoningdesc.astype(str)
    
    
    print('Acquisition Complete. Dataframe available and is now cached for future use.')
    # create a csv of the dataframe for the sake of efficiency. 
    df.to_csv('zillow.csv', index=False)
    
    return df

def remove_columns(df, cols_to_remove):
    '''
    This function takes in a pandas dataframe and a list of columns to remove. It drops those columns from the original df and returns the df.
    '''
    df = df.drop(columns=cols_to_remove)
    return df
                 
                 
def handle_missing_values(df, prop_required_column=0.5 , prop_required_row=0.5):
    '''
    This function takes in a pandas dataframe, default proportion of required columns (set to 50%) and proprtion of required rows (set to 75%).
    It drops any rows or columns that contain null values more than the threshold specified from the original dataframe and returns that dataframe.
    
    Prior to returning that data, it will print statistics and list counts/names of removed columns/row counts 
    '''
    original_cols = df.columns.to_list()
    original_rows = df.shape[0]
    threshold = int(round(prop_required_column * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=threshold)
    remaining_cols = df.columns.to_list()
    remaining_rows = df.shape[0]
    dropped_col_count = len(original_cols) - len(remaining_cols)
    dropped_cols = list((Counter(original_cols) - Counter(remaining_cols)).elements())
    print(f'The following {dropped_col_count} columns were dropped because they were missing more than {prop_required_column * 100}% of data: \n{dropped_cols}\n')
    dropped_rows = original_rows - remaining_rows
    print(f'{dropped_rows} rows were dropped because they were missing more than {prop_required_row * 100}% of data')
    return df

# combined in one function
def data_prep(df, cols_to_remove=[], prop_required_column=0.5, prop_required_row=0.5):
    '''
    This function calls the remove_columns and handle_missing_values to drop columns that need to be removed. It also drops rows and columns that have more 
    missing values than the specified threshold.
    '''
    df = remove_columns(df, cols_to_remove)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    return df

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:
        # get quartiles
        q1, q3 = df[f'{col}'].quantile([.25, .75])  
        # calculate interquartile range
        iqr = q3 - q1   
        # get upper bound
        upper_bound = q3 + k * iqr 
        # get lower bound
        lower_bound = q1 - k * iqr   

        # return dataframe without outliers
        
        df = df[(df[f'{col}'] > lower_bound) & (df[f'{col}'] < upper_bound)]
        
    return df

def moveDecimalPoint(series, decimal_places):
    '''
    Move the decimal place in a given number.

    args:
        num (int)(float) = The number in which you are modifying.
        decimal_places (int) = The number of decimal places to move.
    
    returns:
        (float)
    
    ex. moveDecimalPoint(11.05, -2) returns: 0.1105
    '''
    for _ in range(abs(decimal_places)):

        if decimal_places>0:
            series *= 10; #shifts decimal place right for each row
        else:
            series /= 10.; #shifts decimal place left for each row 

    return series


def prep_zillow_og(df):
    
    df = data_prep(df)
    
    df = df[(df.propertylandusedesc == 'Single Family Residential') |
      (df.propertylandusedesc == 'Mobile Home') |
      (df.propertylandusedesc == 'Manufactured, Modular, Prefabricated Homes') |
      (df.propertylandusedesc == 'Cluster Home')]
    
    # Remove properties that couldn't even plausibly be a studio. 
    df= df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0)]
    
    # Remove properties where there is not a single bathroom.
    df = df[df.bathroomcnt > 0]
    
 # keep only properties with square footage greater than 70 (legal size of a bedroom)
    df = df[df.calculatedfinishedsquarefeet > 70]
    
    # Minimum lot size of single family units.
    df = df[df.lotsizesquarefeet >= 5000].copy()

    # Clear indicators of single unit family. Other codes non-existent or indicate commercial sites. 
   # 0100 - Single Residence
   # 0101 Single residence with pool
   # 0104 - Single resident with therapy pool 
    df = df[(df.propertycountylandusecode == '0100') |
            (df.propertycountylandusecode == '0101') |
            (df.propertycountylandusecode == '0104') |
            (df.propertycountylandusecode == '122') | 
            (df.propertycountylandusecode == '1111') |
            (df.propertycountylandusecode == '1110') |
            (df.propertycountylandusecode == '1')
           ]
    
    
    # Remove 13 rows where unit count is 2. The NaN's can be safely assumed as 1 and were just mislabeled in other counties.  
    df = df[df['unitcnt'] != 2]
    df['unitcnt'].fillna(1)
    
    
    # Property where finished area is 152 but bed count is 5. 
    df = df.drop(labels=75325, axis=0)
    
      
            
    # Redudant columns or uninterpretable columns
    # Unit count was dropped because now its known that theyre all 1. 
    # Finished square feet is equal to calculated sq feet. 
    # full bathcnt and calculatedbathnbr are equal to bathroomcnt
    # property zoning desc is unreadable. 
    # assessment year is unnecessary, all values are 2016. 
    # property land use desc is always single family residence 
    # same with property landuse type id. 
    # room count must be for a different category, as it is always 0.
    # regionidcounty reveals the same information as FIPS. 
    # heatingorsystemtypeid is redundant. Encoded descr. 
    # Id does nothing, and parcelid is easier to represent. 

    
    df =df.drop(columns= ['finishedsquarefeet12', 'fullbathcnt', 'calculatedbathnbr',
                      'propertyzoningdesc', 'unitcnt', 'propertylandusedesc',
                      'assessmentyear', 'roomcnt', 'regionidcounty', 'propertylandusetypeid',
                      'heatingorsystemtypeid', 'id', 'heatingorsystemdesc', 'buildingqualitytypeid',
                         'rawcensustractandblock'],
            axis=1)
    
    
    # The last nulls can be dropped altogether. 
    df = df.dropna()
    
    # the city code is supposed to have five digits. Converted to integer to do an accurate length count as a subsequent string. 
    df.regionidcity = df.regionidcity.astype(int)
    df = df[df.regionidcity.astype(str).apply(len) == 5]
    
    # the same applies to the zip code. 
    
    df.regionidzip = df.regionidzip.astype(int)
    df = df[df.regionidzip.astype(str).apply(len) == 5]
    

    df['yearbuilt'] = df['yearbuilt'].astype(int)
    df.yearbuilt = df.yearbuilt.astype(object) 
    df['age'] = 2017-df['yearbuilt']
    df = df.drop(columns='yearbuilt')
    df['age'] = df['age'].astype('int')
    print('Yearbuilt converted to age. \n')
                          
    df['county'] = df.fips.apply(lambda fips: '0' + str(int(fips)))
    df['county'].replace({'06037': 'los_angeles', '06059': 'orange', '06111': 'ventura'}, inplace=True)
    
    # Feature Engineering
     # create taxrate variable
    df['taxrate'] = round(df.taxamount/df.taxvaluedollarcnt*100, 2)
    # dollar per square foot- structure
    df['structure_cost_per_sqft'] = df.structuretaxvaluedollarcnt/df.calculatedfinishedsquarefeet
    # dollar per square foot- land
    df['land_cost_per_sqft'] = df.landtaxvaluedollarcnt/df.lotsizesquarefeet
    
    df = remove_outliers(df, 3, ['lotsizesquarefeet', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt',
                                'landtaxvaluedollarcnt', 'taxamount', 'calculatedfinishedsquarefeet', 'structure_cost_per_sqft',
                                'taxrate', 'land_cost_per_sqft', 'bedroomcnt', 'bathroomcnt'])
    
    # create quarters based on transaction date
    # first convert from string to datetime format
    df['transactiondate'] = pd.to_datetime(df['transactiondate'], infer_datetime_format=True, errors='coerce')
    # then use pandas feature dt.
    df['fiscal_quarter'] = df['transactiondate'].dt.quarter
    # drop transaction date, since it can't be represented in a histogram 
    # actual dates can be retrieved from parcelid for those interested
    df = df.drop(columns='transactiondate')
    
    # lastly, even after removing outliers from those columns, a few tax rates under 
    # 1% are present. This is unacceptable, as the Maximum Levy (in other words the 
    # bare minimum, too) is 1%. Additional fees can be added, but there's no getting 
    # under 1%. thus, rows falling beneath this must go. 
    df = df[df.taxrate >= 1.0]
    
    # move decimal points so lat
    # and long are correct. 
    
    lats = df['latitude']
    longs = df['longitude']
    
    round(moveDecimalPoint(lats, -6), 6)
    round(moveDecimalPoint(longs, -6), 6)
    
    
    #finally set the index
    df = df.set_index('parcelid')
    
        # A row where the censustractandblock was out of range. Wasn't close to the raw, unlike the others, and started with 483 instead of 60, 61. Too large. 
    df = df.drop(labels=12414696, axis=0)
    
    return df


def prep_zillow_splitter(df):
    
    df = data_prep(df)
    
    df = df[(df.propertylandusedesc == 'Single Family Residential') |
      (df.propertylandusedesc == 'Mobile Home') |
      (df.propertylandusedesc == 'Manufactured, Modular, Prefabricated Homes') |
      (df.propertylandusedesc == 'Cluster Home')]
    
    # Remove properties that couldn't even plausibly be a studio. 
    df= df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0)]
    
    # Remove properties where there is not a single bathroom.
    df = df[df.bathroomcnt > 0]
    
 # keep only properties with square footage greater than 70 (legal size of a bedroom)
    df = df[df.calculatedfinishedsquarefeet > 70]
    
    # Minimum lot size of single family units.
    df = df[df.lotsizesquarefeet >= 5000].copy()

    # Clear indicators of single unit family. Other codes non-existent or indicate commercial sites. 
   # 0100 - Single Residence
   # 0101 Single residence with pool
   # 0104 - Single resident with therapy pool 
    df = df[(df.propertycountylandusecode == '0100') |
            (df.propertycountylandusecode == '0101') |
            (df.propertycountylandusecode == '0104') |
            (df.propertycountylandusecode == '122') | 
            (df.propertycountylandusecode == '1111') |
            (df.propertycountylandusecode == '1110') |
            (df.propertycountylandusecode == '1')
           ]
    
    
    # Remove 13 rows where unit count is 2. The NaN's can be safely assumed as 1 and were just mislabeled in other counties.  
    df = df[df['unitcnt'] != 2]
    df['unitcnt'].fillna(1)
    
    
    # Property where finished area is 152 but bed count is 5. 
    df = df.drop(labels=75325, axis=0)
    
      
            
    # Redudant columns or uninterpretable columns
    # Unit count was dropped because now its known that theyre all 1. 
    # Finished square feet is equal to calculated sq feet. 
    # full bathcnt and calculatedbathnbr are equal to bathroomcnt
    # property zoning desc is unreadable. 
    # assessment year is unnecessary, all values are 2016. 
    # property land use desc is always single family residence 
    # same with property landuse type id. 
    # room count must be for a different category, as it is always 0.
    # regionidcounty reveals the same information as FIPS. 
    # heatingorsystemtypeid is redundant. Encoded descr. 
    # Id does nothing, and parcelid is easier to represent. 

    
    df =df.drop(columns= ['finishedsquarefeet12', 'fullbathcnt', 'calculatedbathnbr',
                      'propertyzoningdesc', 'unitcnt', 'propertylandusedesc',
                      'assessmentyear', 'roomcnt', 'regionidcounty', 'propertylandusetypeid',
                      'heatingorsystemtypeid', 'id', 'heatingorsystemdesc', 'buildingqualitytypeid',
                         'rawcensustractandblock'],
            axis=1)
    
    
    # The last nulls can be dropped altogether. 
    df = df.dropna()
    
    # the city code is supposed to have five digits. Converted to integer to do an accurate length count as a subsequent string. 
    df.regionidcity = df.regionidcity.astype(int)
    df = df[df.regionidcity.astype(str).apply(len) == 5]
    
    # the same applies to the zip code. 
    
    df.regionidzip = df.regionidzip.astype(int)
    df = df[df.regionidzip.astype(str).apply(len) == 5]
    

    df['yearbuilt'] = df['yearbuilt'].astype(int)
    df.yearbuilt = df.yearbuilt.astype(object) 
    df['age'] = 2017-df['yearbuilt']
    df = df.drop(columns='yearbuilt')
    df['age'] = df['age'].astype('int')
    print('Yearbuilt converted to age. \n')
                          
    df['county'] = df.fips.apply(lambda fips: '0' + str(int(fips)))
    df['county'].replace({'06037': 'los_angeles', '06059': 'orange', '06111': 'ventura'}, inplace=True)
    
    # Feature Engineering
     # create taxrate variable
    df['taxrate'] = round(df.taxamount/df.taxvaluedollarcnt*100, 2)
    # dollar per square foot- structure
    df['structure_cost_per_sqft'] = df.structuretaxvaluedollarcnt/df.calculatedfinishedsquarefeet
    # dollar per square foot- land
    df['land_cost_per_sqft'] = df.landtaxvaluedollarcnt/df.lotsizesquarefeet
    
    df = remove_outliers(df, 3, ['lotsizesquarefeet', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt',
                                'landtaxvaluedollarcnt', 'taxamount', 'calculatedfinishedsquarefeet', 'structure_cost_per_sqft',
                                'taxrate', 'land_cost_per_sqft', 'bedroomcnt', 'bathroomcnt'])
    
    # create quarters based on transaction date
    # first convert from string to datetime format
    df['transactiondate'] = pd.to_datetime(df['transactiondate'], infer_datetime_format=True, errors='coerce')
    # then use pandas feature dt.
    df['fiscal_quarter'] = df['transactiondate'].dt.quarter
    # drop transaction date, since it can't be represented in a histogram 
    # actual dates can be retrieved from parcelid for those interested
    df = df.drop(columns='transactiondate')
    
    # lastly, even after removing outliers from those columns, a few tax rates under 
    # 1% are present. This is unacceptable, as the Maximum Levy (in other words the 
    # bare minimum, too) is 1%. Additional fees can be added, but there's no getting 
    # under 1%. thus, rows falling beneath this must go. 
    df = df[df.taxrate >= 1.0]
    
    # move decimal points so lat
    # and long are correct. 
    
    lats = df['latitude']
    longs = df['longitude']
    
    round(moveDecimalPoint(lats, -6), 6)
    round(moveDecimalPoint(longs, -6), 6)
    
    
    #finally set the index
    df = df.set_index('parcelid')
    
        # A row where the censustractandblock was out of range. Wasn't close to the raw, unlike the others, and started with 483 instead of 60, 61. Too large. 
    df = df.drop(labels=12414696, axis=0)
    
    dummy_df = pd.get_dummies(df['county'],
                                 drop_first=False)
       # add the dummies as new columns to the original dataframe
    df = pd.concat([df, dummy_df], axis=1)
    df = df.drop(columns=['ventura', 'county', 'fips'])
    # may drop county later, might just opt to not use it. 
    
    # drop sets that are fed from one another. Ones that were used to create features. 
    df = df.drop(columns=['lotsizesquarefeet', 'regionidcity', 'structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt'])
    
    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    
    
    return train, validate, test
    
    
def get_exploration_data():
    train, validate, test = wrangle_zillow()
    return train



def scale_data(X_train, X_validate, X_test, return_scaler=False):
    '''
    Scales the 3 data splits.
    
    takes in the train, validate, and test data splits and returns their scaled counterparts.
    
    If return_scaler is true, the scaler object will be returned as well.
    
    Target is not scaled.
    
    columns_to_scale was originally used to check whether los_angeles and orange would cause trouble
    '''
    columns_to_scale = ['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet',
       'latitude', 'longitude', 'lotsizesquarefeet',
       'propertycountylandusecode', 'regionidcity', 'regionidzip',
       'structuretaxvaluedollarcnt', 'taxvaluedollarcnt',
       'landtaxvaluedollarcnt', 'taxamount', 'censustractandblock', 'age',
       'taxrate', 'structure_cost_per_sqft', 'land_cost_per_sqft',
       'fiscal_quarter', 'los_angeles', 'orange']
    
    X_train_scaled = X_train.copy()
    X_validate_scaled = X_validate.copy()
    X_test_scaled = X_test.copy()
    
    scaler = StandardScaler()
    scaler.fit(X_train_scaled[columns_to_scale])
    
    X_train_scaled[columns_to_scale] = scaler.transform(X_train[columns_to_scale])
    X_validate_scaled[columns_to_scale] = scaler.transform(X_validate[columns_to_scale])
    X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
    
    if return_scaler:
        return scaler, X_train_scaled, X_validate_scaled, X_test_scaled
    else:
        return X_train_scaled, X_validate_scaled, X_test_scaled

#def get_modeling_data(scale_data=False):
#    df = get_mallcustomer_data()
#    df = one_hot_encode(df)
#    train, validate, test = split(df)
#    if scale_data:
#        return scale(train, validate, test)
#    else:
#        return train, validate, test