'''
Functions for various uses in project

'''
import pandas as pd
import numpy as np
import numpy.random as npr

def feat_eng_nom(df, colName):
    '''
    assigns values to nominal data
    ---
    returns: df, pandas dataframe with newly categorized data
             categories, numpy array of number corresponding to category
    ---
    args: df, original pandas dataframe to be changed
          colName: column name of nominal data in df to be changed to numerical values
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

def onehot(df=None, labels=None, train_x=None):
    '''
    performs one-hot encoding for categories
    ---
    returns: cols, numpy array of the onehot encoded data
    ---
    '''
    cols = np.empty((len(df),0))

    for label in labels:
        onehot = pd.get_dummies(df[label])
        train_cols = pd.get_dummies(train_x[label]).columns
        #check if the onehot dataframe has the same columns as train...this only matters for the test values
        #if not, need to create onehot encoding in different way
        if onehot.columns.all() != train_cols.all():
            #new onehot df that matches train_cols
            onehot = pd.DataFrame(columns=train_cols, index = range(len(df)))
            
            #set all values to for each category, but then make correct value 1
            for i in range(len(df)):
               onehot.iloc[i,:] = 0
               cat = df[label].iloc[i]

               #make sure that category is in train_cols
               if cat in train_cols:
                onehot.at[i, cat] = 1

        cols = np.concatenate((np.asarray(onehot),cols), axis = 1)
    return cols

def onehot_2(df=None, columns=None, train_x=None):
    '''
    onehot encoder
    ---
    returns: df_onehot, pandas dataframe of onehot encoded data from df in columns argument list
    ---
    args: df, pandas dataframe with data to be onehot encoded
          columns, list of columns in df to be onehot encoded
          train_x, pandas dataframe of training dataset...needed to check for test entries that aren't in the training set
    '''
    df_onehot = pd.DataFrame() #final df

    #encode through every col in columns
    for col in columns:
        train_cols = pd.get_dummies(train_x[col]).columns
        onehot = pd.DataFrame(0, columns=train_cols, index=range(len(df)))
        for i in range(len(df)):
            entry = df[col].iloc[i]

            #check if entry is in the training set
            #if not, it will not be added to the onehot encoded data
            if entry in train_cols:
                onehot.at[i, entry] = 1

        df_onehot = pd.concat([onehot, df_onehot], axis = 1)
    
    return df_onehot

def perceptron(X=None,y=None,w_0=None,maxchecks=1000,maxiters=1000):
    '''
    perceptron algorithm from homework 2
    ---
    returns: w, weight matrix for Xw = y
    ---
    args: X, pandas dataframe of feature data
          y, pandas series of target of feature data X
          w_0: initial guess at w
          maxchecks: maximum number of times algorithm checks for incorrect classification, for decreasing runtime
          maxiters: maximum number of times algorithm updates w, for decreasing runtime
    '''
    if w_0 is None:
        w_0 = npr.randn(X.shape[1])
    w = [x for x in w_0] # Make a copy of the intialized weight, since the weight vector is mutable
    n = len(y)
    step = 0
    laststep = 0
    for i in range(0,maxchecks):
        print('iters: ' + str(step))
        print('checks: ' + str(i))
        if np.dot(y.iloc[i%n, 0] * X.iloc[i%n], w) < 0: # Loop over the data points until we find one that violates perceptron condition
            step += 1
            laststep = i
            w += y.iloc[i%n,0] * X.iloc[i%n]
            #if plot_progress and step%5==0:
                #plot_perceptron(X,y,w)
        if i - laststep > n or step > maxiters: # No more violations or number of steps exceed maxiters limit
            #if plot_progress:
                #plot_perceptron(X,y,w)
            break
                
    return w

   