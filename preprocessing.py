import numpy as np
import pandas as pd

# Function to handle null values
def null_imputation(df, cols, typ='num'):
    '''
    Input: df, cols, typ
    '''
    try: # exception handling
        if typ == 'num': # code for numeric columns
            for col in cols:
                result = df[col].fillna(df[col].quantile(.50)) # filling numerical nulls with median because median is not biased toward outliers.
                df[col] = result
        elif typ == 'cat': # code for categorical columns
            for col in cols:
                top = df[col].describe()['top']
                result = df[col].fillna(top) # filling with top
                df[col] = result
        else:
            print('Enter valid informations!')
    except: # exception handling
        print('Exception incounterred please check type of the input columns')

# Function for preprocessing data to make ready for eda
def preprocessingOne(df):
    '''
    Input: df
    '''
    high_null_cols= df.isna().sum()[(df.isna().sum() > 500) & (df.isna().sum() > 0)].index # columns with high nulls
    less_null_cols= df.isna().sum()[(df.isna().sum() < 500) & (df.isna().sum() > 0)].index # columns with low nulls
    numeric_nulls = df[less_null_cols].describe().columns # columns with less null and numerical values
    categorical_nulls = list(set(less_null_cols) - set(numeric_nulls)) # columns with less null and categorical values
    df.drop(high_null_cols,axis=1, inplace=True) # drop columns with more than 50% nulls
    null_imputation(df, cols=numeric_nulls, typ='num') # Imputing null for numerical columns with median
    null_imputation(df, cols=categorical_nulls, typ='cat') # Imputing null for categorical columns with mod
    print('data preprocessed successfully!')
    return df
# Function for feature manipulation
def preprocessingTwo(df):
  cols_to_drop = ['Id', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
                  'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch', 'GarageArea']
  df['totalArea'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']+df['GrLivArea']+df['GarageArea'] # Feature Manipulation
  df['totalPorchArea'] = df['WoodDeckSF'] + df['OpenPorchSF'] + df['EnclosedPorch'] + df['ScreenPorch'] # Feature Manipulation
  df.drop(cols_to_drop, axis=1, inplace=True) # drop unwanted columns
  return df
# Funtion making data ready for model building
def preprocessingThree(df, isTrain=1):
  if isTrain == 1:
    good_features = ['totalArea', 'SalePrice']
  else:
    good_features = ['totalArea']
  df = preprocessingOne(df)
  df = preprocessingTwo(df)
  categorical_features = list(df.describe(include='object').columns)
  good_features.extend(categorical_features)
  df = df[good_features]
  df = pd.get_dummies(df, columns=categorical_features, drop_first=True, dtype=int) # Encoding categorical variables
  return df

# Outlier treatment
def treatOutliers(df,col):
  q1 = df[col].quantile(.25)
  q3 = df[col].quantile(.75)
  iqr = q3-q1
  lower_bound = q1 - 1.5*iqr
  upper_bound = q3 + 1.5*iqr
  print(col)
  print('q1: ',q1)
  print('q3: ',q3)
  print('iqr: ',iqr)
  print('lower: ',lower_bound)
  print('upper: ',upper_bound)
  df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
  df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
  return df