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


def data_preprocessing(df, date):
    """Preprocessing of the data"""
    # Take only data where there is a best/ask
    # mask = (~df["bp0"].isnull()) & (~df["ap0"].isnull())
    # df_sub = df.loc[mask].copy()
    df_sub = df.copy()

    # df_sub['datetime'] = date.timestamp()  + df_sub['timestamp']/1_000_000
    # df_sub['datetime'] = pd.to_datetime(df_sub['datetime'], unit = 's', utc = True)
    return df_sub


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


def get_random_sample(df, n):
    df_sub = df.sample(n=n, random_state=1)
    df_sub = df_sub.sort_values(by="timestamp", ascending=True)

    return df_sub


def plot_timeseries_fig(df, cols, datestr):
    ax = df.plot(y=cols)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.4f}"))
    fig = ax.get_figure()

    fig.savefig(f"./data/figures/time_series_{datestr}_{'_'.join(cols)}.png")
    plt.close(fig)


def plot_hist_fig(df):
    ax = (df["timestamp"] / 1_000_000).plot(kind="hist", bins=50)
    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))

    fig = ax.get_figure()
    fig.savefig(f"./data/figures/hist_timestamps.png")
    plt.close(fig)


def resample_data(df, precision):
    """Resamples the data by flooring the timestamp up to the nearest freq
    specified by precision

    df (pd.DataFrame): Must have column timestamp in micros.
                      So ceiling up to nearest 10_000, creates a grid spaced by 10ms
    """

    df_sub = df.copy()

    # df_sub['timestamp_new'] = df_sub['timestamp'].round(decimals=-1*precision)

    freq = 10**precision
    df_sub["timestamp_new"] = np.ceil(df_sub["timestamp"] / freq).astype(int) * freq
    df_sub = df_sub.groupby(by="timestamp_new").last()

    new_idx = pd.RangeIndex(
        start=df_sub.index.min(), stop=df_sub.index.max() + 1, step=freq
    )
    df_sub = df_sub.reindex(new_idx)
    df_sub = df_sub.ffill()

    return df_sub


def describe_data():
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


def feature_selection(df):
    df_sub = df.copy()
    df_sub["mid_price"] = (df_sub["ap0"] + df_sub["bp0"]) / 2
    df_sub["inv_mid_price"] = (
        df_sub["bp0"] * df_sub["aq0"] + df_sub["ap0"] * df_sub["bq0"]
    ) / (df_sub["bq0"] + df_sub["aq0"])

    df_sub["spread"] = df_sub["ap0"] - df_sub["bp0"]
    df_sub["spread_bps"] = 10_000 * df_sub["spread"] / df_sub["mid_price"]

    df_sub["ord_im"] = df_sub["bq0"] - df_sub["aq0"]
    df_sub["ord_im_rel"] = df_sub["bq0"] / (df_sub["bq0"] + df_sub["aq0"])

    df_sub["ord_im1"] = df_sub["bq1"] - df_sub["aq1"]
    df_sub["ord_im_rel1"] = df_sub["bq1"] / (df_sub["bq1"] + df_sub["aq1"])

    df_sub["returns"] = df_sub["mid_price"].diff(periods=1)
    df_sub["returns_log"] = 100 * np.log(df_sub["mid_price"]).diff(periods=1)

    df_sub["returns_fut"] = df_sub["returns"].shift(-1)
    df_sub["returns_log_fut"] = df_sub["returns_log"].shift(-1)

    df_sub["mid_price_fut"] = df_sub["mid_price"].shift(-1)

    return df_sub


def train_model():
    input_dates = ["20190610", "20190611", "20190612", "20190613", "20190614"
            ]

    for datestr in input_dates:
        date = dt.datetime.strptime(datestr, "%Y%m%d")
        print(date)
        df1 = get_order_book_data(datestr)
        df2 = resample_data(df1,5)
        df3 = feature_selection(df2)

        features = ["spread", "ord_im", "ord_im_rel", "bq0", "aq0"]
        target = ["mid_price_fut"]

        df3 = df3.loc[~df3[target[0]].isnull()].copy()

        X = df3[features].values
        y = df3[target].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=17
        )
        print("The dimension of X_train is {}".format(X_train.shape))
        print("The dimension of X_test is {}".format(X_test.shape))

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Lasso regression model

        print("\nLasso Model............................................\n")
        lasso = Lasso(alpha=0.01)
        lasso.fit(X_train, y_train)
        train_score_ls = lasso.score(X_train, y_train)
        test_score_ls = lasso.score(X_test, y_test)

        print("The train score for ls model is {}".format(train_score_ls))
        print("The test score for ls model is {}".format(test_score_ls))
        ax = pd.Series(lasso.coef_, features).sort_values(ascending = True).plot(kind = "bar")
        ax.tick_params(axis='x', labelrotation=20)
        plt.show()



if __name__ == "__main__":
    datestr = "20190610"
    df1 = get_order_book_data(datestr)
    # plot_hist_fig(df1)

    df2 = resample_data(df1, 5)
    df3 = feature_selection(df2)
    #train_model()

    # cols = ["mid_price", "inv_mid_price"]
    # cols = ["spread_bps"]
    # cols = ["returns_abs"]
    #cols = ["order_imbalance"]
    #plot_timeseries_fig(df3, cols, datestr)
