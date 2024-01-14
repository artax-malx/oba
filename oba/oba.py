import pandas as pd
import numpy as np

def get_data(date):

    df = pd.read_csv(f"./data/res_{date}.csv", sep = ",")
    df = pd.read_csv(f"./data/test_input.csv", sep = ",")

    return df

def aggregate_order_book(dict_orders):

    dfo = pd.DataFrame.from_dict(dict_orders, orient = "index")

    bids = dfo.loc[dfo['side'] == "b"]
    asks = dfo.loc[dfo['side'] == "a"]

    bid_side = bids.groupby(by = ['price'])['quantity'].sum()
    ask_side = asks.groupby(by = ['price'])['quantity'].sum()
    
    bid_side = bid_side.sort_index(ascending=False)
    ask_side = ask_side.sort_index(ascending=True)

    b = bid_side.iloc[:5].reset_index()
    a = ask_side.iloc[:5].reset_index()

    b = b.rename(columns = {'price' :'bid','quantity' : 'bid_quantity'})
    a = a.rename(columns = {'price' :'ask','quantity' : 'ask_quantity'})

    b = b.loc[:, ['bid_quantity','bid']].copy()

    return b,a


def process_order_updates(df):


    bps = [f'bp{i}' for i in range(5)]
    bqs = [f'bq{i}' for i in range(5)] 
    aps = [f'ap{i}' for i in range(5)]
    aqs = [f'aq{i}' for i in range(5)] 
    
    prices = bps + aps
    quants = bqs + aqs

    data = []
    curr_orders = {}

    for index, row in df.iterrows():

        action = row['action']
        ord_id = row['id']
        timestamp = row['timestamp']
        side = row['side']
        price = row['price']
        quantity = row['quantity']

        if action == "a":
            order = {'timestamp' : timestamp,
                      'side' : side,
                      'price' : price,
                      'quantity' : quantity,
                      }
            curr_orders[ord_id] = order
        elif action == "m":
            if not ord_id in curr_orders.keys():
                print(f"Order id {ord_id} not in the data")

            order = {'timestamp' : timestamp,
                      'side' : side,
                      'price' : price,
                      'quantity' : quantity,
                      }
            curr_orders[ord_id] = order
        elif action == "d":
            del curr_orders[ord_id]

        b, a = aggregate_order_book(curr_orders)

        print(pd.concat([b,a], axis = 1))
        print("==="*30)

        #temp_dict = {'timestamp' : row['timestamp'],
        #             'price' : row['price'],
        #             'side' : row['side'],}

        #for x in prices:
        #    temp_dict[x] = np.nan

        #for x in quants:
        #    temp_dict[x] = 0

        #data.append(temp_dict)

    #ob = pd.DataFrame(data)
    #return ob 
    return curr_orders


if __name__ == "__main__":
    df1 = get_data("20190610")
    out = process_order_updates(df1)

