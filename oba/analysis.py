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
from sklearn.metrics import mean_squared_error
import logging


def get_order_book_data(date):
    filename = f"./data/output/order_book_data_{date}.csv"
    return pd.read_csv(filename, sep=",")


def get_test_order_book_data():
    arr = np.array(
        [
            [1.1000e02, 1.0045e04, 6.2000e01, 1.0055e04, 9.8000e01],
            [1.7500e02, 1.0065e04, 4.6000e01, 1.0075e04, 4.2000e01],
            [2.2000e02, 1.0075e04, 9.0000e00, 1.0080e04, 2.5000e01],
            [2.6000e02, 1.0075e04, 6.7000e01, 1.0080e04, 4.5000e01],
            [3.3300e02, 1.0075e04, 3.8000e01, 1.0080e04, 1.0000e00],
            [3.4500e02, 1.0060e04, 8.4000e01, 1.0070e04, 2.6000e01],
            [4.7000e02, 1.0060e04, 5.3000e01, 1.0070e04, 3.8000e01],
            [5.5500e02, 1.0045e04, 1.0000e00, 1.0050e04, 5.0000e01],
        ]
    )
    df = pd.DataFrame(arr, columns=["timestamp", "bp0", "bq0", "ap0", "aq0"])
    return df

def describe_data():
    """ Quick function to get descriptive data on orderbook datasets """
    input_dates = ["20190610", "20190611", "20190612", "20190613", "20190614"]
    data = []
    for datestr in input_dates:
        df1 = get_order_book_data(datestr)
        min_time = df1["timestamp"].min()
        max_time = df1["timestamp"].max()
        avg_mid = (df1["bp0"] / 2 + df1["ap0"] / 2).mean()
        avg_order_vol = (
            df1["bq0"]
            + df1["aq0"]
            + df1["bq1"]
            + df1["aq1"]
            + df1["bq2"]
            + df1["aq2"]
            + df1["bq3"]
            + df1["aq3"]
            + df1["bq4"]
            + df1["aq4"]
        ).mean()
        data.append([datestr, min_time, max_time, avg_mid, avg_order_vol])

    out = pd.DataFrame(
        data, columns=["date", "min_time", "max_time", "avg_mid", "avg_order_vol"]
    )

    return out


def get_random_sample(df, n):
    """Helper function to get a random sample of a dataframe"""
    df_sub = df.sample(n=n, random_state=1)
    df_sub = df_sub.sort_values(by="timestamp", ascending=True)

    return df_sub


def plot_timeseries_fig(datestr, cols):
    """ Quick function to plot time series of the data """

    ob = get_order_book_data(datestr)
    ob_res = resample_data(ob, resample_freq = 5)
    df = feature_selection(ob_res, forecast_period = 1)

    ax = df.plot(y=cols)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
    fig = ax.get_figure()

    fig.savefig(f"./data/figures/time_series_{datestr}_{'_'.join(cols)}.png")
    plt.close(fig)


def get_mse_from_model(model, X_train, y_train, X_test, y_test):
    pred_train = model.predict(X_train)
    mse_train = mean_squared_error(y_train, pred_train)

    pred = model.predict(X_test)
    mse_test = mean_squared_error(y_test, pred)

    return mse_test, mse_train

def resample_data(df, resample_freq=5):
    """Resamples the data by flooring the timestamp up to the nearest freq
    specified by resample_freq

    Args:
        df (pd.DataFrame): Must have column timestamp in micros. 

        resample_freq (int): raised 10 to this power and use number
                             as resample frequency.

    Example: If resample_freq = 5, function wil ceilup to nearest 100_000, 
            creating a grid spaced by 100ms
    """

    df_sub = df.copy()
    freq = 10**resample_freq
    df_sub["timestamp_new"] = np.ceil(df_sub["timestamp"] / freq).astype(int) * freq
    df_sub = df_sub.groupby(by="timestamp_new").last()

    new_idx = pd.RangeIndex(
        start=df_sub.index.min(), stop=df_sub.index.max() + 1, step=freq
    )
    df_sub = df_sub.reindex(new_idx)
    df_sub = df_sub.ffill()

    return df_sub

def feature_selection(df, forecast_period):
    """ Calculates relevant set of features
        
    Args:
        df (pd.DataFrame): orderbook data
        forecast_period: forecast horizon for the target variable
    """
    df_sub = df.copy()
    df_sub["mid_price"] = (df_sub["ap0"] + df_sub["bp0"]) / 2
    df_sub["inv_mid_price"] = (
        df_sub["bp0"] * df_sub["aq0"] + df_sub["ap0"] * df_sub["bq0"]
    ) / (df_sub["bq0"] + df_sub["aq0"])

    df_sub["inv_mid_price2"] = (
        df_sub["bp0"] * df_sub["aq0"]
        + df_sub["ap0"] * df_sub["bq0"]
        + df_sub["bp1"] * df_sub["aq1"]
        + df_sub["ap1"] * df_sub["bq1"]
    ) / (df_sub["bq0"] + df_sub["aq0"] + df_sub["bq1"] + df_sub["aq1"])

    df_sub["spread"] = df_sub["ap0"] - df_sub["bp0"]
    df_sub["spread_bps"] = 10_000 * df_sub["spread"] / df_sub["mid_price"]

    df_sub["ord_im"] = df_sub["bq0"] - df_sub["aq0"]
    df_sub["ord_im_rel"] = df_sub["bq0"] / (df_sub["bq0"] + df_sub["aq0"])

    df_sub["ord_im1"] = df_sub["bq1"] - df_sub["aq1"]
    df_sub["ord_im_rel1"] = df_sub["bq1"] / (df_sub["bq1"] + df_sub["aq1"])

    df_sub["diff_bq0"] = df_sub["bq0"].diff(periods=1)
    df_sub["diff_aq0"] = df_sub["aq0"].diff(periods=1)

    df_sub["diff_mid_price"] = df_sub["mid_price"].diff(periods=1)
    df_sub["log_diff_mid_price"] = 100 * np.log(df_sub["mid_price"]).diff(periods=1)

    shift_period = -1 * forecast_period
    df_sub["mid_price_fut"] = df_sub["mid_price"].shift(shift_period)

    return df_sub


def train_model(datestr, forecast_period=1, alpha=10, resample_freq=5):
    """ Main module to train Lasso model

    Args:
        datestr (str): date of the input file
        forecast_period (int): the forcecast horizion for the target variable 
        alpha (float): Alpha used in the Lasso regression
        resample_freq (int): power of ten, used to calculate 
                             the discretization step = 10**resample_freq
    """
    df1 = get_order_book_data(datestr)
    df2 = resample_data(df1, resample_freq=resample_freq)
    df3 = feature_selection(df2, forecast_period=forecast_period)

    features = [
        "spread_bps",
        "ord_im_rel",
        "ord_im_rel1",
        "diff_bq0",
        "diff_aq0",
        "inv_mid_price2",
    ]

    target = ["mid_price_fut"]

    # Lasso model fitting does not accept NaN values
    df3 = df3.loc[:, target + features].copy()
    df3 = df3.dropna(axis=0, how="any")

    X = df3[features].values
    y = df3[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=17
    )

    logging.info("The dimension of X_train is {}".format(X_train.shape))
    logging.info("The dimension of X_test is {}".format(X_test.shape))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logging.info("Lasso Model............................................")

    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    train_score_ls = lasso.score(X_train, y_train)
    test_score_ls = lasso.score(X_test, y_test)

    mse_test, mse_train = get_mse_from_model(lasso, X_train, y_train, X_test, y_test)
    scores = [datestr, train_score_ls, mse_train, test_score_ls, mse_test]

    logging.info("The train score for ls model is {}".format(train_score_ls))
    logging.info("The test score for ls model is {}".format(test_score_ls))
    logging.info(".......................................................")

    coeffs = pd.Series(lasso.coef_, features, name="coefficient")

    return scores, coeffs

if __name__ == "__main__":
    datestr = "20190612"
    cols = ["inv_mid_price","mid_price"]
    plot_timeseries_fig(datestr, cols)
