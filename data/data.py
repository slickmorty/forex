import yfinance as yf
import datetime as dt
import pandas as pd


end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=730)

data_df = yf.download("^GSPC", period='max', interval="1d")
data_df = data_df.reset_index()
data_df.to_csv('data/s&p500.csv')
