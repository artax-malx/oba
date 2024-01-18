import os
from logger import init_logger
from oba import analysis
import pandas as pd
import time

logger = init_logger("./logs/analysis.log")


def main():
    input_dates = ["20190610", "20190611", "20190612", "20190613", "20190614"]
    scores_data = []

    for_per = 1
    alpha = 0.04
    res_freq = 5

    for datestr in input_dates:
        logger.info(f"Training model for date {datestr}")

        suffix = f"{for_per}_{alpha}_{res_freq}"
        fname_coeff = f"./data/lasso_model/lasso_coeffs_{datestr}_{suffix}.csv"

        scores, coeffs = analysis.train_model(
            datestr, forecast_period=for_per, alpha=alpha, resample_freq=res_freq
        )
        coeffs.to_csv(fname_coeff, sep=",")
        logger.info(f"Wrote coefficients to {fname_coeff}")
        scores_data.append(scores)

    df_scores = pd.DataFrame(
        scores_data,
        columns=["date", "test_score", "test_mse", "train_score", "train_mse"],
    )

    fname_score = f"./data/lasso_model/scores_{suffix}.csv"
    df_scores.to_csv(fname_score, sep=",", index=False)
    logger.info(f"Wrote scores to {fname_score}")


if __name__ == "__main__":
    main()
