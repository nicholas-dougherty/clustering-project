from env import host, username, password, get_db_url
import os
import pandas as pd 
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from datetime import date

def get_exploration_data():
    '''
    This acquires, prepares, pre-processes, and splits the data frame
    but only returns the train data set, since that is the only DF used
    for exploration.
    '''
    train, validate, test = prep_zillow_splitter(acquire_zillow_data())
    return train


def wrangle_zillow_anew(target):
    '''
    An improvement upon wrangle which goes through the pipeline up to modeling,
    acquires, prepares, pre-processes, feature engineers, splits, and then sets
    the X and y for each--- dependent and independent variable--- by including
    a string of the target (in this case logerror) each of the useful modeling
    sets will be returned. 
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
    '''
    The initial version of wrangle, which combines the steps of acquisition, preparation,
    pre-processing, and splitting, returning each of the three. Train can be interpreted freely,
    but test must be left untouched (with some exceptions) and certainly unseen until running the final
    model as a matter of performance checking. Validate is used to fine-tune hyperparameters.
    '''
    df = prep_zillow_og(acquire_zillow_data())
    
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
    #print(f'The following {dropped_col_count} columns were dropped because they were missing more than {prop_required_column * 100}% of data: \n{dropped_cols}\n')
    dropped_rows = original_rows - remaining_rows
    #print(f'{dropped_rows} rows were dropped because they were missing more than {prop_required_row * 100}% of data')
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
        and return that dataframe. The quantiles are designated
        within the function, but k is optional. 1.5 would be a thorough
        removal and 3 is less harsh in its content loss. For this dataframe,
        visually, the two accomplish the same, so to prevent major loss
        3 is implemented. 
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
    '''
    The full preparation cycle outlined by print-statements
    '''
    
    print('Beginning preparation...\n')
    print('Detecting Nulls; set to delete columns and then rows are comprised of 50% nulls')      
    df = data_prep(df)
    
    print('''
    The following 34 columns were dropped because they were missing more than 50.0% of data: 
['airconditioningtypeid', 'architecturalstyletypeid', 'basementsqft', 'buildingclasstypeid', 'decktypeid', 'finishedfloor1squarefeet', 'finishedsquarefeet13', 'finishedsquarefeet15', 'finishedsquarefeet50', 'finishedsquarefeet6', 'fireplacecnt', 'garagecarcnt', 'garagetotalsqft', 'hashottuborspa', 'poolcnt', 'poolsizesum', 'pooltypeid10', 'pooltypeid2', 'pooltypeid7', 'regionidneighborhood', 'storytypeid', 'threequarterbathnbr', 'typeconstructiontypeid', 'yardbuildingsqft17', 'yardbuildingsqft26', 'numberofstories', 'fireplaceflag', 'taxdelinquencyflag', 'taxdelinquencyyear', 'airconditioningdesc', 'architecturalstyledesc', 'buildingclassdesc', 'storydesc', 'typeconstructiondesc']

0 rows were dropped because they were missing more than 50.0% of data''')
    
    print('Selecting propertylanduse by the potential that they classify as single family residential, although indirectly\n')
    df = df[(df.propertylandusedesc == 'Single Family Residential') |
      (df.propertylandusedesc == 'Mobile Home') |
      (df.propertylandusedesc == 'Manufactured, Modular, Prefabricated Homes') |
      (df.propertylandusedesc == 'Cluster Home')]
    print('Plausible candidates: Mobile Home; Manufactured, Modular, Prefabricated Homes; Cluster Home; Single Family Residential\n')
    
    print('Removing properties that have zero bathrooms and zero bedrooms\n')
    # Remove properties that couldn't even plausibly be a studio. 
    df= df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0)]
    # Remove properties where there is not a single bathroom.
    df = df[df.bathroomcnt > 0]
    
    print('Preserving properties with a finished living area greater than 70 square feet\n')
 # keep only properties with square footage greater than 70 (legal size of a bedroom)
    df = df[df.calculatedfinishedsquarefeet > 70]
    print('California housing standards classify a bedroom as being greater than 70 sqft, any properties beneath this are undeniably not homes\n')
    
    print('The minimum lot size of single family units is 5000 square feet. DataFrame excludes fields that fail to meet this criteria.\n')
    # Minimum lot size of single family units.
    df = df[df.lotsizesquarefeet >= 5000].copy()

    # Clear indicators of single unit family. Other codes non-existent or indicate commercial sites. 
   # 0100 - Single Residence
   # 0101 Single residence with pool
   # 0104 - Single resident with therapy pool 
    print('Preserving certain property county landuse codes based on research\n')
    df = df[(df.propertycountylandusecode == '0100') |
            (df.propertycountylandusecode == '0101') |
            (df.propertycountylandusecode == '0104') |
            (df.propertycountylandusecode == '122') | 
            (df.propertycountylandusecode == '1111') |
            (df.propertycountylandusecode == '1110') |
            (df.propertycountylandusecode == '1')
           ]
    
    print('Removed 13 rows where unit count is 2. Duplexes are not permitted. After reasoning through remainders, Orange county had many false detailed Nulls that were actually HIGHLY LIKELY single family residences. Unit Count NA filled with 1.\n')
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
    print('Dropping columns that are redundant or provide no discernible value to this set. Details included in wrangle.py \n')
    
    df =df.drop(columns= ['finishedsquarefeet12', 'fullbathcnt', 'calculatedbathnbr',
                      'propertyzoningdesc', 'unitcnt', 'propertylandusedesc',
                      'assessmentyear', 'roomcnt', 'regionidcounty', 'propertylandusetypeid',
                      'heatingorsystemtypeid', 'id', 'heatingorsystemdesc', 'buildingqualitytypeid',
                         'rawcensustractandblock'],
            axis=1)
    
    print('Remaining Null Count miniscule. Less than 0.000% of DF. Dropping values. \n')
    # The last nulls can be dropped altogether. 
    df = df.dropna()
    
    print('City and Zip Code must have five place-holders. Many do not, are were dropped \n')
    # the city code is supposed to have five digits. Converted to integer to do an accurate length count as a subsequent string. 
    df.regionidcity = df.regionidcity.astype(int)
    df = df[df.regionidcity.astype(str).apply(len) == 5]
    
    # the same applies to the zip code. 
    
    df.regionidzip = df.regionidzip.astype(int)
    df = df[df.regionidzip.astype(str).apply(len) == 5]
    
    print('Calculating and creating an age column')
    df['yearbuilt'] = df['yearbuilt'].astype(int)
    df.yearbuilt = df.yearbuilt.astype(object) 
    df['age'] = 2017-df['yearbuilt']
    df = df.drop(columns='yearbuilt')
    df['age'] = df['age'].astype('int')
    print('Yearbuilt converted to age. \n')
    
    print('County column created using Federal Information Processing Standards')
    df['county'] = df.fips.apply(lambda fips: '0' + str(int(fips)))
    df['county'].replace({'06037': 'los_angeles', '06059': 'orange', '06111': 'ventura'}, inplace=True)
    
    print('Creating columns for tax rate and cost per square foot for structure and land \n')
    # Feature Engineering
     # create taxrate variable
    df['taxrate'] = round(df.taxamount/df.taxvaluedollarcnt*100, 2)
    # dollar per square foot- structure
    df['structure_cost_per_sqft'] = df.structuretaxvaluedollarcnt/df.calculatedfinishedsquarefeet
    # dollar per square foot- land
    df['land_cost_per_sqft'] = df.landtaxvaluedollarcnt/df.lotsizesquarefeet
    
    print('Using k=3, removing outliers from continuous variables')
    df = remove_outliers(df, 3, ['lotsizesquarefeet', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt',
                                'landtaxvaluedollarcnt', 'taxamount', 'calculatedfinishedsquarefeet', 'structure_cost_per_sqft',
                                'taxrate', 'land_cost_per_sqft', 'bedroomcnt', 'bathroomcnt'])
    print('Creating fiscal quarters according to transaction date \n')
    # create quarters based on transaction date
    # first convert from string to datetime format
    df['transactiondate'] = pd.to_datetime(df['transactiondate'], infer_datetime_format=True, errors='coerce')
    # then use pandas feature dt.
    df['fiscal_quarter'] = df['transactiondate'].dt.quarter
    # drop transaction date, since it can't be represented in a histogram 
    # actual dates can be retrieved from parcelid for those interested
    df = df.drop(columns='transactiondate')
    print('As per Californian tax lexy standard, taxrate listed as less than 1% removed from dataframe. Error either on input or tax record itself.')
    # lastly, even after removing outliers from those columns, a few tax rates under 
    # 1% are present. This is unacceptable, as the Maximum Levy (in other words the 
    # bare minimum, too) is 1%. Additional fees can be added, but there's no getting 
    # under 1%. thus, rows falling beneath this must go. 
    df = df[df.taxrate >= 1.0]
    
    # move decimal points so lat
    # and long are correct. 
    print('Moving decimal points 6 to the left for latitude and longitude, as is necessary to capture the appropriate coordinates for this area of interest. \n')
    lats = df['latitude']
    longs = df['longitude']
    
    round(moveDecimalPoint(lats, -6), 6)
    round(moveDecimalPoint(longs, -6), 6)
    
    print('Setting parcelid as the index. \n')
    #finally set the index
    df = df.set_index('parcelid')
    
        # A row where the censustractandblock was out of range. Wasn't close to the raw, unlike the others, and started with 483 instead of 60, 61. Too large. 
    df = df.drop(labels=12414696, axis=0)
    
    print('\n\n\nPreparation Complete. Begin Exploratory Data Analysis.')
    return df


def prep_zillow_splitter(df):
    
    '''
    performs all that is found in the former preparation function
    but excludes the print statements and also splits the dataframe into
    train, validate, test. 
    '''
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

    
    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    
    
    return train, validate, test




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