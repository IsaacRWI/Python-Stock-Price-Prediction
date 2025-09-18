import yfinance as yf
import numpy as np

df = yf.download("^GSPC", start="2010-01-01")

print(df)