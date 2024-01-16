import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV, Lasso
from sklearn.preprocessing import StandardScaler

def get_order_book_data(date):
    df = pd.read_csv(f"./data/order_book_data_{date}.csv", sep=",")
    return df

def resample_data(df):
    df_sub = df.copy()

    decimals = -4 # -3 means rounded to 1ms
    df_sub['timestamp_new'] = df_sub.timestamp.round(decimals=decimals)
    df_sub  = df_sub.groupby(by = "timestamp_new").last()

    new_idx = pd.RangeIndex(start=df_sub.index.min(), stop = df_sub.index.max()+1, step = 10_000)
    df_sub = df_sub.reindex(new_idx)
    df_sub = df_sub.ffill()

    return df_sub


def data_preprocessing(df, date):
    """ Preprocessing of the data """
    # Take only data where there is a best/ask 
    #mask = (~df["bp0"].isnull()) & (~df["ap0"].isnull())
    #df_sub = df.loc[mask].copy()
    df_sub = df.copy()

    #df_sub['datetime'] = date.timestamp()  + df_sub['timestamp']/1_000_000
    #df_sub['datetime'] = pd.to_datetime(df_sub['datetime'], unit = 's', utc = True)
    return df_sub

def feature_selection(df):

    df_sub = df.copy()
    df_sub['mid_price'] = (df_sub['bp0']*df_sub['aq0'] + df_sub['ap0']*df_sub['bq0'])/(df_sub['bq0'] + df_sub['aq0'])
    df_sub['simple_mid_price'] = (df_sub["ap0"] + df_sub["bp0"])/2
    df_sub['spread_abs'] = (df_sub["ap0"] - df_sub["bp0"])
    df_sub['spread_bps'] = 10_000*df_sub["spread_abs"]/df_sub["simple_mid_price"]
    df_sub['flow_imbalance'] = df_sub['bq0']/(df_sub['bq0'] + df_sub['aq0'])
    return df_sub


if __name__ == "__main__":

    input_dates = ["20190610", 
                   #"20190611", 
                   #"20190612", 
                   #"20190613", 
                   #"20190614"
                   ]

    for datestr in input_dates:
        df1 = get_order_book_data(datestr)
        date = dt.datetime.strptime(datestr,"%Y%m%d")
        df2 = data_preprocessing(df1, date)
        df3 = resample_data(df2)
        df4 = feature_selection(df3)

        features = ['spread_bps', 'flow_imbalance']
        target = ['mid_price']

        X   = df4[features].values
        y = df4[target].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)
        print("The dimension of X_train is {}".format(X_train.shape))
        print("The dimension of X_test is {}".format(X_test.shape))

        #Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        #Lasso regression model

        print("\nLasso Model............................................\n")
        lasso = Lasso(alpha = 10)
        lasso.fit(X_train,y_train)
        train_score_ls =lasso.score(X_train,y_train)
        test_score_ls =lasso.score(X_test,y_test)
        
        print("The train score for ls model is {}".format(train_score_ls))
        print("The test score for ls model is {}".format(test_score_ls))

        #print(f"======{datestr}======")
        #print(df2["spread_abs"].sort_values(ascending=True).unique())
        #print(df2["spread_bps"].sort_values(ascending=True).unique())
