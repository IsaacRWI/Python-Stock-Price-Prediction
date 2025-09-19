import pandas as pd
import yfinance as yf
import numpy as np
import pandas
import datetime
import matplotlib.pyplot as plt

df = yf.download("^GSPC", start="2010-01-01", multi_level_index=False)  # multi level index includes ticker name and other unnecessary for training data
# print(df)

df = df[["Close"]]  # keeps only the closing price column
print(df)
print(df.index)


def str_to_datetime (s):
    split = s.split("-")
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year = year, month = month, day = day)
    

# df.index = df.index.apply(string_to_datetime)


plt.plot(df.index, df["Close"])
plt.show()

def df_to_windowed_df (dataframe, first_date_str, last_date_str, n=3):  # turns dataframe into training data for model, with n being the number of previous datapoints it looks at for predictions
    first_date = str_to_datetime(first_date_str)
    last_date = str_to_datetime(last_date_str)

    target_date = first_date

    dates = []
    X, Y = [], []

    last_time = False
    while True:
        df_subset = dataframe.loc[:target_date].tail(n + 1)

        if len(df_subset) != n + 1:
            print(f'Error: Window of size {n} is too large for date {target_date}')
            return

        values = df_subset['Close'].to_numpy()
        x, y = values[:-1], values[-1]

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        next_week = dataframe.loc[target_date:target_date + datetime.timedelta(days=7)]
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split('T')[0]
        year_month_day = next_date_str.split('-')
        year, month, day = year_month_day
        next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))

        if last_time:
            break

        target_date = next_date

        if target_date == last_date:
            last_time = True

    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates

    X = np.array(X)
    for i in range(0, n):
        X[:, i]
        ret_df[f'Target-{n - i}'] = X[:, i]

    ret_df['Target'] = Y

    return ret_df

windowed_df = df_to_windowed_df(df, "2010-01-07", "2025-09-18")
print(windowed_df)