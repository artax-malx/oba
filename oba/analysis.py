import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV, Lasso
from sklearn.preprocessing import StandardScaler


def get_order_book_data(date):
    df = pd.read_csv(f"./data/output/order_book_data_{date}.csv", sep=",")
    return df

def get_test_order_book_data():

    arr = np.array([[1.1000e+02, 1.0045e+04, 6.2000e+01, 1.0055e+04, 9.8000e+01],
        [1.7500e+02, 1.0065e+04, 4.6000e+01, 1.0075e+04, 4.2000e+01],
        [2.2000e+02, 1.0075e+04, 9.0000e+00, 1.0080e+04, 2.5000e+01],
        [2.6000e+02, 1.0075e+04, 6.7000e+01, 1.0080e+04, 4.5000e+01],
        [3.3300e+02, 1.0075e+04, 3.8000e+01, 1.0080e+04, 1.0000e+00],
        [3.4500e+02, 1.0060e+04, 8.4000e+01, 1.0070e+04, 2.6000e+01],
        [4.7000e+02, 1.0060e+04, 5.3000e+01, 1.0070e+04, 3.8000e+01],
        [5.5500e+02, 1.0045e+04, 1.0000e+00, 1.0050e+04, 5.0000e+01]])
    df = pd.DataFrame(arr, columns = ['timestamp','bp0', 'bq0', 'ap0','aq0'])
    return df 

def get_random_sample(df,n):

    df_sub = df.sample(n=n, random_state=1)
    df_sub = df_sub.sort_values(by = "timestamp", ascending=True)

    return df_sub

def plot_and_save_fig(df):

    cols = ['mid_price','inv_mid_price']
    ax = df.plot(y = cols)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    fig = ax.get_figure()

    fig.savefig(f"./data/figures/time_series_{'_'.join(cols)}.png")

    ax = (df['timestamp']/1_000_000).plot(kind = "hist", bins = 50)
    fig = ax.get_figure()
    fig.savefig(f"./data/figures/hist_timestamps.png")


def resample_data(df,precision):
    """ Resamples the data by flooring the timestamp up to the nearest freq
    specified by precision

    df (pd.DataFrame): Must have column timestamp in micros.
                      So ceiling up to nearest 10_000, creates a grid spaced by 10ms
    """

    df_sub = df.copy()

    #df_sub['timestamp_new'] = df_sub['timestamp'].round(decimals=-1*precision)

    freq = 10**precision
    df_sub['timestamp_new'] = np.ceil(df_sub['timestamp']/freq).astype(int)*freq
    df_sub  = df_sub.groupby(by = "timestamp_new").last()

    new_idx = pd.RangeIndex(start=df_sub.index.min(), stop = df_sub.index.max()+1, step = freq)
    df_sub = df_sub.reindex(new_idx)
    df_sub = df_sub.ffill()

    return df_sub

def describe_data():

    input_dates = ["20190610", 
                   "20190611", 
                   "20190612", 
                   "20190613", 
                   "20190614"
                   ]
    data = []
    for datestr in input_dates:
        df1 = get_order_book_data(datestr)
        min_time = df1['timestamp'].min()
        max_time = df1['timestamp'].max()
        data.append([datestr, min_time, max_time])

    out = pd.DataFrame(data, columns = ['date', 'min_time','max_time'])

    return out

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
    df_sub['mid_price'] = (df_sub["ap0"] + df_sub["bp0"])/2
    df_sub['inv_mid_price'] = (df_sub['bp0']*df_sub['aq0'] + df_sub['ap0']*df_sub['bq0'])/(df_sub['bq0'] + df_sub['aq0'])
    df_sub['spread_abs'] = (df_sub["ap0"] - df_sub["bp0"])
    df_sub['spread_bps'] = 10_000*df_sub["spread_abs"]/df_sub["mid_price"]
    df_sub['flow_imbalance'] = df_sub['bq0']/(df_sub['bq0'] + df_sub['aq0'])
    return df_sub


def train_model():
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

if __name__ == "__main__":
    datestr = "20190610"
    df1 = get_order_book_data(datestr)
    df2 = resample_data(df1, 5)

    df3 = feature_selection(df2)
    #plot_and_save_fig(df3, cols)
