import pandas as pd
import numpy as np

columns = ["AGE", "SEX", "STEROID", "ANTIVIRALS", "FATIGUE", "MALAISE", "ANOREXIA", "LIVER_BIG",
           "LIVER_FIRM", "SPLEEN_PALPABLE", "SPIDERS", "ASCITES", "VARICES", "BILIRUBIN", "ALK_PHOSPHATE", "SGOT", "ALBUMIN", "PROTIME", "HISTOLOGY", "CLASS"]


def read_dataset():

    train_dataset = pd.read_csv(
        "train_dataset.csv", header=0, names=columns, na_values="?")

    test_dataset = pd.read_csv(
        "test_dataset.csv", header=0, names=columns, na_values="?")

    train_dataset = train_dataset.dropna()
    test_dataset = test_dataset.dropna()

    print(train_dataset.head())

    return train_dataset, test_dataset
