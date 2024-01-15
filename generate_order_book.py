import os
from logger import init_logger
from oba import oba
import pandas as pd
import time

logger = init_logger("./logs/oba.log")


def main():

    input_dates = ["20190610", "20190611", "20190612", "20190613", "20190614"]

    for datestr in input_dates:
        logger.info(f"Started converting raw updates to Order Book for {datestr}")

        df_res = oba.get_data(datestr)

        start = time.time()
        out = oba.process_order_updates(df_res)
        end = time.time()

        logger.info(f"Finished converting data, run time: {end - start} sec")
        df_out = pd.DataFrame.from_dict(out, orient="columns")

        logger.info("Wrote file to data folder")
        df_out.to_csv(f"./data/order_book_data_{datestr}.csv", sep=",", index=False)


if __name__ == "__main__":
    main()
