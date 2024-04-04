'''
Functions for various uses in project

'''
import pandas as pd
import numpy as np

def feat_eng_nom(df, colName):
    '''
    assigns values to nominal data

    df: pandas DataFrame
    colName: column name of nominal data in df to be changed to numerical values
    categories: ndarray of the values associated with the categories
    '''
    #sort dataframe by colName to make it alphabetical
    #helpful in this project as there are predefined nominal columns of string values "number - words"
    df.sort_values(by=[colName])
    categories = np.append('zero', df[colName].unique()) #add zero at the beginning so categories start at 1
    #categories = df[colName].unique()

    #loop through df and assign category values
    for i, row in df.iterrows():
        num = np.where(categories == row[colName])[0][0]
        df.at[i, colName] = num

    return df, categories


def feat_eng_nominal(df, colName):
    return (df)

   