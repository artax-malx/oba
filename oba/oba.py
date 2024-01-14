import pandas as pd
import numpy as np
from sortedcontainers import SortedDict
import itertools
import time
import logging
#from logger import init_logger
#
#logger = init_logger("./logs/oba.log")

def get_data(date, test=False):
    if test:
        df = pd.read_csv("./data/test_input.csv", sep=",")
    else:
        df = pd.read_csv(f"./data/res_{date}.csv", sep=",")

    return df


def print_ob_dict(input_dict):

    print("Size | Bid Price")
    for i in range(5):
        print(input_dict[f"bq{i}"], " | ",input_dict[f"bp{i}"])

    print("==="*30)
    print("Size | Ask Price")
    for i in range(5):
        print(input_dict[f"aq{i}"], " | ",input_dict[f"ap{i}"])

def aggregate_order_book(dict_orders):
    """ Helper function that aggregates dictionary of live orders and 
    returns 5 best levels of both sides of the order book
    """

    dfo = pd.DataFrame.from_dict(dict_orders, orient="index")

    bids = dfo.loc[dfo["side"] == "b"]
    asks = dfo.loc[dfo["side"] == "a"]

    bid_side = bids.groupby(by=["price"])["quantity"].sum()
    ask_side = asks.groupby(by=["price"])["quantity"].sum()

    bid_side = bid_side.sort_index(ascending=False)
    ask_side = ask_side.sort_index(ascending=True)

    b = bid_side.iloc[:5].reset_index()
    a = ask_side.iloc[:5].reset_index()

    b = b.rename(columns={"price": "bid", "quantity": "bid_quantity"})
    a = a.rename(columns={"price": "ask", "quantity": "ask_quantity"})

    b = b.loc[:, ["bid_quantity", "bid"]].copy()
    a = a.loc[:, ["ask_quantity", "ask"]].copy()

    return b, a


def get_n_levels(n, bid_dict, ask_dict):
    """ Return a dictionary with n best levels from bid_dict and ask_dict 

    If there are less than n levels available, bq = 0 and bp = NaN is filled in
    """

    bid_keys = bid_dict.__reversed__()
    bid_num = min(len(bid_dict), n)
    bid_best = list(itertools.islice(bid_keys, bid_num))
    bid_levels = [(price, bid_dict[price]) for price in bid_best]

    level_dict = {}
    for i in range(n):
        if (i + 1) <= bid_num:
            level_dict[f"bp{i}"] = bid_levels[i][0]
            level_dict[f"bq{i}"] = bid_levels[i][1]
        else:
            level_dict[f"bp{i}"] = np.nan
            level_dict[f"bq{i}"] = 0

    ask_num = min(len(ask_dict), n)
    ask_best = list(itertools.islice(ask_dict, ask_num))
    ask_levels = [(price, ask_dict[price]) for price in ask_best]

    for i in range(n):
        if (i + 1) <= ask_num:
            level_dict[f"ap{i}"] = ask_levels[i][0]
            level_dict[f"aq{i}"] = ask_levels[i][1]
        else:
            level_dict[f"ap{i}"] = np.nan
            level_dict[f"aq{i}"] = 0

    return level_dict


def process_order_updates(df):
    """ Processes dataframe consisting of order book updates and returns each update with 5 levels
    of the order book on both sides

    Args:
        df (pd.DataFrame) : each row is order book update

    Returns:
        data (list of dicts)
    """
    curr_orders = {}
    bid_dict = SortedDict()
    ask_dict = SortedDict()
    data=[]

    for index, row in df.iterrows():
        action = row["action"]
        ord_id = row["id"]
        timestamp = row["timestamp"]
        side = row["side"]
        price = row["price"]
        quantity = row["quantity"]

        if action == "a":
            if side == "b":
                bid_dict[price] = bid_dict.get(price, 0) + quantity
            elif side == "a":
                ask_dict[price] = ask_dict.get(price, 0) + quantity
            else:
                logging.error(f"Order id {ord_id} has incorrect side input")
                continue

            new_order = {
                "timestamp": timestamp,
                "side": side,
                "price": price,
                "quantity": quantity,
            }

            curr_orders[ord_id] = new_order
        elif action == "d":
            curr_ord = curr_orders.get(ord_id, None)
            if not curr_ord:
                logging.error(f"Can't delete Order id {ord_id}; not in the data")
                continue

            #TODO: take side, price and quantity from current order
            if side == "b":
                bid_dict[price] = bid_dict.get(price, 0) - quantity
                if bid_dict[price] == 0:
                    del bid_dict[price]
            elif side == "a":
                ask_dict[price] = ask_dict.get(price, 0) - quantity
                if ask_dict[price] == 0:
                    del ask_dict[price]
            else:
                logging.error(f"Order id {ord_id} has incorrect side input")
                continue
            del curr_orders[ord_id]
        elif action == "m":
            curr_ord = curr_orders.get(ord_id, None)
            if not curr_ord:
                logging.error(f"Order id {ord_id} not in the data")
                continue

            curr_side = curr_ord["side"]
            curr_price = curr_ord["price"]
            curr_quantity = curr_ord["quantity"]

            if curr_side == "b":
                bid_dict[curr_price] = bid_dict.get(curr_price, 0) - curr_quantity
                if bid_dict[curr_price] == 0:
                    del bid_dict[curr_price]
            elif curr_side == "a":
                ask_dict[curr_price] = ask_dict.get(curr_price, 0) - curr_quantity
                if ask_dict[curr_price] == 0:
                    del ask_dict[curr_price]
            else:
                # TODO: In theory not necessary to check, but might not be a bad idea
                pass

            if side == "b":
                bid_dict[price] = bid_dict.get(price, 0) + quantity
            elif side == "a":
                ask_dict[price] = ask_dict.get(price, 0) + quantity
            else:
                logging.error(f"Order id {ord_id} has incorrect side input")
                continue

            new_order = {
                "timestamp": timestamp,
                "side": side,
                "price": price,
                "quantity": quantity,
            }

            curr_orders[ord_id] = new_order

        out_dict = get_n_levels(5, bid_dict, ask_dict)
        #print(out_dict)

        temp_dict = {'timestamp' : timestamp,
                     'price' : price,
                     'side' : side,}

        final_dict = {**temp_dict,**out_dict}
        data.append(final_dict)

    #return data, curr_orders
    return data


if __name__ == "__main__":

    input_dates = ["20190610","20190611", "20190612","20190613", "20190614"]
    datestr = "20190614"
    logging.info(f"Started Order Book Analysis Run for {datestr}")

    df_res = get_data(datestr, test=True)

    start = time.time()
    #out, last_order_dict = process_order_updates(df_res)
    out = process_order_updates(df_res)
    end = time.time()
    print(f"Run time {end - start} sec")

    #last_level_dict = out[-1]
    #b,a = aggregate_order_book(last_order_dict)
    #print_ob_dict(last_level_dict)
