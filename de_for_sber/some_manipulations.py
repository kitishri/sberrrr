import pickle

import dill
import pandas as pd



def new():
    df = pd.read_csv("C:/Users/Ekaterina/sber_de/de_for_sber/data/raw/ga_hits.csv")
    pd.set_option('display.max_columns', None)
    print(df.head())
    print(df.shape)
    print(df.dtypes)

    print(df.describe(include='all'))





if __name__ == "__main__":
    new()