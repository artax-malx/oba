import pytest
import pandas as pd
from oba import analysis
import numpy as np


    
def test_resample_data():
    
    df_test = analysis.get_test_order_book_data()

    df_samp  = analysis.resample_data(df_test, 2)

    exp_index = np.array([200, 300, 400, 500, 600])

    exp_output = np.array([[1.7500e+02, 1.0065e+04, 4.6000e+01, 1.0075e+04, 4.2000e+01],
       [2.6000e+02, 1.0075e+04, 6.7000e+01, 1.0080e+04, 4.5000e+01],
       [3.4500e+02, 1.0060e+04, 8.4000e+01, 1.0070e+04, 2.6000e+01],
       [4.7000e+02, 1.0060e+04, 5.3000e+01, 1.0070e+04, 3.8000e+01],
       [5.5500e+02, 1.0045e+04, 1.0000e+00, 1.0050e+04, 5.0000e+01]])

    assert np.array_equal(df_samp.index.values, exp_index)
    assert np.array_equal(df_samp.values, exp_output)
