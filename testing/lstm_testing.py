import pandas as pd
import yfinance as yf
import numpy as np
import pandas
import datetime

df = yf.download("^GSPC", start="2010-01-01")
# print(df)

df = df[["Close"]]  # keeps only the closing price column
print(df)
print(df.index)

"""
def string_to_datetime (s):
    split = s.split("-")
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year = year, month = month, day = day)
    

df.index = df.index.apply(string_to_datetime)
"""
