'''
Functions for various uses in project

'''
import pandas as pd
import numpy as np
import numpy.random as npr
from sklearn.model_selection import KFold
import sklearn.metrics as skm
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

def ordinal(df, colName):
    '''
    assigns values to ordinal data
    ---
    returns: df, pandas dataframe with newly categorized data
             categories, numpy array of number corresponding to category
    ---
    args: df, original pandas dataframe to be changed
          colName: column name of nominal data in df to be changed to numerical values
    '''
    #sort dataframe by colName to make it alphabetical
    #helpful in this project as there are predefined nominal columns of string values "number - words"
    dfc = df.copy()
    if colName == 'drainage_class':
        categories = np.array(['Somewhat excessively drained', 'Well drained', 'Moderately well drained', 'Somewhat poorly drained',
                                          'Poorly drained', 'Very poorly drained'])
    elif colName == 'aquifer_vulnerability':
        categories = np.array(['high', 'medium', 'low'])

    #loop through df and assign category values
    for i, row in df.iterrows():
        num = np.where(categories == row[colName])[0][0]
        dfc.at[i, colName] = num

    return dfc

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

    return df_onehot


def tgus(shl, app, det, om, bulk, koc):
    '''
    calculates tgus equation from soil and pesticide parameters
    ---
    returns: tgus, result from equation
    ---
    args: shl, soil halflife for pesticide [m3/Mg]
          app, pesticide application rate [mg/m2]
          det, pesticide detection limit [mg/m3]
          om, soil organic matter fraction
          bulk, soil bulk density [Mg/m3]
          koc, partitioning coefficient for pesticide
    '''
    t = 100 # current assumption, [days]
    pf = 0.004 # current assumption
    phi = app / (det * 0.01)
    foc = 0.6 * om / 100
    xi = (pf * phi) / (foc * bulk)
    tgus = round(0.025 * shl * (np.log10(xi/koc)) - 0.0075*t,1)

    return tgus

def kfold_crossval(df, clf, modelName):
    '''
    performs k-fold cross validation with model clf on df
    ---
    returns: best_model, the best model from validation
             best_train_score, the best training accuracy
             best_val_score, the best validation accuracy
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
    kf = KFold(n_splits=4)
    avg_accuracy = []
    KFold(n_splits=4, random_state=None, shuffle=False)
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
    p_i = perm_imp(best_model, test_x, test_y)
    test_acc, tfpn = test_accuracy(modelName, test_x, test_y, best_model)
    return best_model, best_train_score, best_val_score, test_acc, tfpn, p_i

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
    clf.fit(xt,yt)
    train_score = round(clf.score(xt, yt),3) * 100
    pred = clf.predict(xv)
    acc = round((1 - skm.zero_one_loss(yv,pred, normalize=True)) * 100, 1)
    return train_score, clf, acc



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
    if modelName in ['perceptron','svc', 'bagging']:
        pred = model.predict(test_x)
        
        #find true/false positive/negatives
        tfpn = pd.DataFrame(data = np.zeros((1,4)),columns = ['true positive', 'true negative', 'false positive', 'false negative'])
        for i in range(len(pred)):
            if (pred[i] == test_y[i] and pred[i] == 1):
                tfpn.loc[0,'true positive'] += 1
            elif (pred[i] == test_y[i] and pred[i] == -1):
                tfpn.loc[0,'true negative'] += 1
            elif (pred[i] != test_y[i] and pred[i] == 1):
                tfpn.loc[0,'false positive'] += 1
            else:
                tfpn.loc[0,'false negative'] += 1    
           
        return round((1 - skm.zero_one_loss(test_y, pred)) * 100, 1), tfpn
    #if modelName == 'svc':
       # return round(model.score(test_x, test_y),2) * 100

def perm_imp(clf, X_test, y_test):
    result = permutation_importance(clf, X_test, y_test, n_repeats=50, random_state=42, n_jobs=2)

    sorted_importances_idx = result.importances_mean.argsort()
    importances = pd.DataFrame(result.importances[sorted_importances_idx].T, columns=X_test.columns[sorted_importances_idx],)

    '''
    ax = importances.plot.box(vert=False, whis=10)
    ax.set_title("Permutation Importances (test set)")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in accuracy score")
    ax.figure.tight_layout()
    '''

    return importances