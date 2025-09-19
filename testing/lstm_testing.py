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


# plt.plot(df.index, df["Close"])
# plt.show()  # uncomment for plotting results

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

# for uni-variate forecasting as we only use 1 variable in the prediction that being the closing price
def windowed_df_to_date_X_y (windowed_dataframe):  # function to turn windowed dataframe into 3 numpy arrays to train tensorflow model
    df_as_np = windowed_dataframe.to_numpy()
    dates = df_as_np[:, 0]  # turns all rows in the first column :, 0
    middle_matrix = df_as_np[:, 1:-1]  # all rows starting from the second column until the last one exclusively
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))  # first dimension being the len of dates ie number of observations or datapoints
                                                                        # matrix.shape[1] shapes all the target- data into 1 3-dimensional variable?? i think
                                                                        # last 1 as we are only using 1 variable, a 3-dimensional variable but still only 1  3 different values of the variable and how it changes over time but still only 1
    y = df_as_np[:, -1]
    return dates, X.astype(np.float32), y.astype(np.float32)

dates, X, y = windowed_df_to_date_X_y(windowed_df)

print(dates.shape, X.shape, y.shape)

# splitting data into training, validation and testing
q_80 = int(len(dates) * 0.8)
q_90 = int(len(dates) * 0.9)

dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

plt.plot(dates_train, y_train)
plt.plot(dates_val, y_val)
plt.plot(dates_test, y_test)
plt.legend(["training", "validation", "testing"])
plt.show()