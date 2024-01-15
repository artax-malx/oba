import pytest
import pandas as pd
from oba import oba
import numpy as np
from sortedcontainers import SortedDict


def make_expected_output():
    output = [
        {
            "timestamp": 0,
            "price": 15,
            "side": "b",
            "bp0": 15,
            "bq0": 3,
            "bp1": np.nan,
            "bq1": 0,
            "bp2": np.nan,
            "bq2": 0,
            "bp3": np.nan,
            "bq3": 0,
            "bp4": np.nan,
            "bq4": 0,
            "ap0": np.nan,
            "aq0": 0,
            "ap1": np.nan,
            "aq1": 0,
            "ap2": np.nan,
            "aq2": 0,
            "ap3": np.nan,
            "aq3": 0,
            "ap4": np.nan,
            "aq4": 0,
        },
        {
            "timestamp": 0,
            "price": 15,
            "side": "b",
            "bp0": 15,
            "bq0": 8,
            "bp1": np.nan,
            "bq1": 0,
            "bp2": np.nan,
            "bq2": 0,
            "bp3": np.nan,
            "bq3": 0,
            "bp4": np.nan,
            "bq4": 0,
            "ap0": np.nan,
            "aq0": 0,
            "ap1": np.nan,
            "aq1": 0,
            "ap2": np.nan,
            "aq2": 0,
            "ap3": np.nan,
            "aq3": 0,
            "ap4": np.nan,
            "aq4": 0,
        },
        {
            "timestamp": 1,
            "price": 20,
            "side": "b",
            "bp0": 20,
            "bq0": 3,
            "bp1": 15.0,
            "bq1": 5,
            "bp2": np.nan,
            "bq2": 0,
            "bp3": np.nan,
            "bq3": 0,
            "bp4": np.nan,
            "bq4": 0,
            "ap0": np.nan,
            "aq0": 0,
            "ap1": np.nan,
            "aq1": 0,
            "ap2": np.nan,
            "aq2": 0,
            "ap3": np.nan,
            "aq3": 0,
            "ap4": np.nan,
            "aq4": 0,
        },
        {
            "timestamp": 1,
            "price": 30,
            "side": "a",
            "bp0": 20,
            "bq0": 3,
            "bp1": 15.0,
            "bq1": 5,
            "bp2": np.nan,
            "bq2": 0,
            "bp3": np.nan,
            "bq3": 0,
            "bp4": np.nan,
            "bq4": 0,
            "ap0": 30.0,
            "aq0": 1,
            "ap1": np.nan,
            "aq1": 0,
            "ap2": np.nan,
            "aq2": 0,
            "ap3": np.nan,
            "aq3": 0,
            "ap4": np.nan,
            "aq4": 0,
        },
        {
            "timestamp": 2,
            "price": 15,
            "side": "b",
            "bp0": 20,
            "bq0": 3,
            "bp1": np.nan,
            "bq1": 0,
            "bp2": np.nan,
            "bq2": 0,
            "bp3": np.nan,
            "bq3": 0,
            "bp4": np.nan,
            "bq4": 0,
            "ap0": 30.0,
            "aq0": 1,
            "ap1": np.nan,
            "aq1": 0,
            "ap2": np.nan,
            "aq2": 0,
            "ap3": np.nan,
            "aq3": 0,
            "ap4": np.nan,
            "aq4": 0,
        },
    ]
    
    return output

def test_orderbook_generation():
    df_test = oba.get_data(None, test=True)
    output = oba.process_order_updates(df_test)
    expected_output = make_expected_output()

    assert len(output) == len(expected_output), "Test output length doesn't match length expected output"

    for i in range(len(output)):
        assert output[i] == expected_output[i]


def test_orderbook_update_and_delete():

    bid_dict = SortedDict()
    ask_dict = SortedDict()

    assert bid_dict == {}
    assert ask_dict == {}

    oba.order_book_update_add(bid_dict, ask_dict, 99, 2,'b')

    assert bid_dict == {99 : 2}
    assert ask_dict == {}

    oba.order_book_update_add(bid_dict, ask_dict, 95, 3,'b')
    assert bid_dict == {95 : 3, 99 : 2}
    assert ask_dict == {}

    oba.order_book_update_add(bid_dict, ask_dict, 99, 5,'b')
    assert bid_dict == {95 : 3, 99 : 7}
    assert ask_dict == {}

    oba.order_book_update_delete(bid_dict, ask_dict, 95, 3, 'b')
    assert bid_dict == {99 : 7}
    assert ask_dict == {}

    oba.order_book_update_add(bid_dict, ask_dict, 101, 4, 'a')
    assert bid_dict == {99 : 7}
    assert ask_dict == {101 : 4}

    oba.order_book_update_add(bid_dict, ask_dict, 103, 7, 'a')
    assert bid_dict == {99 : 7}
    assert ask_dict == {101 : 4, 103 : 7}

    oba.order_book_update_delete(bid_dict, ask_dict, 101, 4, 'a')
    assert bid_dict == {99 : 7}
    assert ask_dict == {103 : 7}
