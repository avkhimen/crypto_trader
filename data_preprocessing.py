import numpy as np
import pandas as pd
df = pd.read_csv('data/XBTUSD_60.csv', header=None)
df.columns = ['unix_timestamp','open_price','high_price','low_price','close_price','other_1','other_2']
df['unix_timestamp'] = df['unix_timestamp'].astype(int)
df['timestamp'] = pd.to_datetime(df['unix_timestamp'], unit='s')
df_xbt = df[['timestamp','unix_timestamp','open_price','high_price','low_price','close_price','other_1','other_2']]

df = pd.read_csv('data/ETHUSD_60.csv', header=None)
df.columns = ['unix_timestamp','open_price','high_price','low_price','close_price','other_1','other_2']
df['unix_timestamp'] = df['unix_timestamp'].astype(int)
df['timestamp'] = pd.to_datetime(df['unix_timestamp'], unit='s')
df_eth = df[['timestamp','unix_timestamp','open_price','high_price','low_price','close_price','other_1','other_2']]

df_merged = pd.merge(df_xbt, df_eth, on=['unix_timestamp', 'timestamp'], suffixes=['_xbt', '_eth'])
