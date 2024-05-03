'''
Functions for various uses in project

'''
import pandas as pd
import numpy as np
import numpy.random as npr
from sklearn.model_selection import KFold

def nominal(df, colName):
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

def onehot(df=None, columns=None):
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
        onehot = pd.get_dummies(df[col])
        df_onehot = pd.concat([onehot, df_onehot], axis = 1)
        '''
        train_cols = pd.get_dummies(train_x[col]).columns
        onehot = pd.DataFrame(0, columns=train_cols, index=range(len(df)))
        for i in range(len(df)):
            entry = df[col].iloc[i]

            #check if entry is in the training set
            #if not, it will not be added to the onehot encoded data
            if entry in train_cols:
                onehot.at[i, entry] = 1

        df_onehot = pd.concat([onehot, df_onehot], axis = 1)
        '''
    return df_onehot

def perceptron(X=None,y=None,w_0=None,maxchecks=10000,maxiters=10000):
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
        if np.dot(y.iloc[i%n,0] * X.iloc[i%n], w) < 0: # Loop over the data points until we find one that violates perceptron condition
            step += 1
            laststep = i
            w += y.iloc[i%n,0] * X.iloc[i%n]
            #if plot_progress and step%5==0:
                #plot_perceptron(X,y,w)
        if i - laststep > n or step > maxiters: # No more violations or number of steps exceed maxiters limit
            #if plot_progress:
                #plot_perceptron(X,y,w)
            break
    #print(f'checks: {i}')
    #print(f'iters: {step}') 
    return w

def sign(val):
    '''
    get sign for perceptron loss evaluation
    ---
    returns: sign, the result from the sign function for dot(w,x)
    ---
    args: result of dot(w,x)
    '''
    sign = 1
    if val < 0:
        sign = -1

    return sign

def tgus(shl, koc, star = False):
    '''
    tgus and tgus* equation
    ---
    returns: tgus, result from equation
    ---
    args: shl, soil halflife for pesticide
          koc, partitioning coefficient for pesticide
          star, boolean for whether or not to calculate tgus or tgus*
    '''
    if star == True: tgus = round(0.025*shl * (3.4 - np.log10(koc)),1)     
    else: tgus = round(shl * (3.4 - np.log10(koc)),1)

    return tgus

def zero_one_loss(x,y, w):
    '''
    calculates accuracy  using 0-1 loss
    ---
    returns: acc, the percentage of accurately predicted outcomes in y
    ---
    args: x, data points 
          y, outcome
          w, weight vector
    '''
    losses = 0
    for idx, row in x.iterrows():
        result = sign(np.dot(w, row))

        if result != y.iloc[idx]:
            losses += 1

    acc = round((1 - losses / len(x)), 2) * 100
    return acc

def kfold_crossval(df, clf, modelName):
    '''
    performs k-fold cross validation with model clf on df
    ---
    returns: best_model, the best model from validation
             best_train_score, the best training accuracy
             best_val_score, the best validation accuracu
             test_acc, the model accuracy on the remaining test data
    ---
    args: df, the pandas dataframe to perform cross validation on
          clf, the model to be used
          modelName, string name of model (perceptron, svc, etc.)
    '''
    df=df.sample(frac=1) 
    train_proportion = 0.8 
    n = len(df)
    t = int(train_proportion * n)

    # separate training and test sets
    y = df['detected']
    X = df.loc[:, ~df.columns.isin(['detected'])]

    #features in training set
    train_x = X.iloc[:t,:].reset_index().iloc[:,1:]
    #features in test set
    test_x = X.iloc[t:,:].reset_index().iloc[:,1:]
    #targets in train set
    train_y = pd.Series(y[:t].reset_index().iloc[:,1:].iloc[:,0])
    #targets in test set
    test_y = pd.Series(y[t:].reset_index().iloc[:,1:].iloc[:,0])

    # perform K-fold cross validation
    kf = KFold(n_splits=8)
    avg_accuracy = []
    KFold(n_splits=8, random_state=None, shuffle=False)
    for i, (train_index, val_index) in enumerate(kf.split(train_x)):
        # separate split training set to get validation
        # get indices for kfold
        xt = train_x.loc[train_index,:].reset_index().iloc[:,1:]
        yt = pd.Series(train_y.loc[train_index].reset_index().iloc[:,1:].iloc[:,0])
        xv = train_x.loc[val_index,:].reset_index().iloc[:,1:]
        yv = pd.Series(train_y.loc[val_index].reset_index().iloc[:,1:].iloc[:,0])

        # assess model accuracy on train/validation sets
        train_score, model, acc = model_assessment(modelName, clf, xt, yt, xv, yv)
        avg_accuracy = np.append(acc, avg_accuracy)

        # keep best model
        if acc >= np.max(avg_accuracy):
            best_model = model
            best_train_score = train_score
            best_val_score = acc

    # check model on remaining test data
    test_acc = test_accuracy(modelName, test_x, test_y, best_model)

    return best_model, best_train_score, best_val_score, test_acc

def model_assessment(modelName, clf, xt, yt, xv, yv):
    '''
    evaluates model accuracy with fitting set (xt,yt) and accuracy on validation set (xv, yv)
    ---
    returns: train_score, the accuracy of the trained model on the training set (xt,yt)
             model/w, the fitted model based on what modelName is
             acc, the accuracy of the trained model on the validation set (xv, yv)
    ---
    args: modelName, string name of model being built
          clf, actual sklearn model
          xt, training data
          yt, training output
          xv, validation data
          yv, validation output
    '''
    if modelName == 'perceptron':
        clf.fit(xt,yt)
        train_score = round(clf.score(xt, yt),2) * 100
        w = clf.coef_
        acc = zero_one_loss(xv,yv,w)
        return train_score, w, acc
    
    elif modelName == 'svc':
        clf.fit(xt,yt)
        train_score = round(clf.score(xt,yt), 2) * 100
        model = clf
        acc = round(clf.score(xv,yv), 2) * 100
        return train_score, model, acc

def test_accuracy(modelName, test_x, test_y, model):
    '''
    evaluates model accuracy on test data
    ---
    returns: calculated accuracy depending on which model is passed
    ---
    args: modelName, string name of model being built
          test_x, testing data
          test_y, testing output
          model, actual sklearn model
    '''
    if modelName == 'perceptron':
        return zero_one_loss(test_x, test_y, model)
    if modelName == 'svc':
        return round(model.score(test_x, test_y),2) * 100