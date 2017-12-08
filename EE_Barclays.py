import pandas as pd
import pickle
import numpy as np
from math import sqrt
import time
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

# Create dataframe
df = pd.read_csv('EE_FinalDF.csv')

def traintestsplitfor2016():
    X_train = df.iloc[152:,0:-1]
    y_train = df.iloc[152:,-1]
    X_test  = df.iloc[0:152,0:-1]
    y_test  = df.iloc[0:152,-1]
    return X_train,X_test,y_train,y_test

def main():
    while True:
        print "\n"
        print "Here are your options :"
        print "1. Model trained with data from 2011-2015."
        print "2. Model trained with data from 2011-2016."
        print "3. Exit"
        user_input = input("Please select 1 or 2 or 3.\n")

        if user_input == 1:
            print "Selecting best model..."
            time.sleep(1)
            print "Model Selected : ElasticNet"
            model = pickle.load(open('ElasticNetPred2016.sav', 'rb'))
            X_train,X_test,y_train,y_test = traintestsplitfor2016()
            model = model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            print "RMSE \t:",sqrt(mean_squared_error(y_pred,y_test))
            print "r2_score:",r2_score(y_pred,y_test)
            user_yorn =  raw_input("Would you like to select your own input? 'y'or'n'\n")
            if user_yorn == 'y':
                code = input("Please input LA code below \n")
                if (len(df.loc[df['X1'].isin([code])]) == 0):
                              code = input("Invalid LA code. Please enter correct code.\n")
                mask = np.where((df['X1'] == code) & (df['X0'] == 2016))
                X = df.loc[mask]
                X_test = X.iloc[:,0:-1]
                y_test = X.iloc[:,-1]
                model = model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
                print "Predicted value : ",y_pred[0]," Actual Value : ",y_test.iloc[0],"\n"
                user_yorn = raw_input("Would you like to continue to main menu? 'y'or'n'\n")
                if user_yorn == 'y':
                    continue
                elif user_yorn == 'n':
                    print "Thank You!"
                    break
            elif user_yorn == 'n':
                print "Thank You!"

        elif user_input == 2:
            print "Selecting best model..."
            time.sleep(1)
            print "Model Selected : Lasso"
            model = pickle.load(open('LassoAllYear.sav', 'rb'))
            X = df.iloc[:,0:-1]
            y = df.iloc[:,-1]
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=9)
            model = model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            print "RMSE \t:",sqrt(mean_squared_error(y_pred,y_test))
            print "r2_score:",r2_score(y_pred,y_test)
            user_yorn =  raw_input("Would you like to select your own input? 'y'or'n'\n")
            if user_yorn == 'y':
                code = input("Please input LA code below \n")
                if (len(df.loc[df['X1'].isin([code])]) == 0):
                              code = input("Invalid LA code. Please enter correct code.\n")
                mask = np.where((df['X1'] == code) & (df['X0'] == 2016))
                X = df.loc[mask]
                X_test = X.iloc[:,0:-1]
                y_test = X.iloc[:,-1]
                model = model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
                print "Predicted value : ",y_pred[0]," Actual Value : ",y_test.iloc[0],"\n"
                user_yorn = raw_input("Would you like to continue to main menu? 'y'or'n'\n")
                if user_yorn == 'y':
                    continue
                elif user_yorn == 'n':
                    print "Thank You!"
                    break
            elif user_yorn == 'n':
                print "Thank You!"
                
        elif user_input == 3:
            print "Thank You!"
            break
        
    
main()
    
    