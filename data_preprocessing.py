import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Merge two files on a common column')
parser.add_argument('-c1', '--cur1', required=True, help='Name of the first currency (ex: xbt)')
parser.add_argument('-c2', '--cur2', required=True, help='Name of the second currency *ex: eth)')
parser.add_argument('-p', '--price_type', required=True, default='open', help="Type of price to extract ['open', 'high', 'low', 'close']")

# Parse the command-line arguments
args = parser.parse_args()

cur1 = args.cur1
cur2 = args.cur2

file1 = cur1.upper() + 'USD' + '_60.csv'
file2 = cur2.upper() + 'USD' + '_60.csv'

df = pd.read_csv('data/original_files/' + file1, header=None)
df.columns = ['unix_timestamp','open_price','high_price','low_price','close_price','other_1','other_2']
df['unix_timestamp'] = df['unix_timestamp'].astype(int)
df['timestamp'] = pd.to_datetime(df['unix_timestamp'], unit='s')
df_cur1 = df[['timestamp','unix_timestamp','open_price','high_price','low_price','close_price','other_1','other_2']]

df = pd.read_csv('data/original_files/' + file2, header=None)
df.columns = ['unix_timestamp','open_price','high_price','low_price','close_price','other_1','other_2']
df['unix_timestamp'] = df['unix_timestamp'].astype(int)
df['timestamp'] = pd.to_datetime(df['unix_timestamp'], unit='s')
df_cur2 = df[['timestamp','unix_timestamp','open_price','high_price','low_price','close_price','other_1','other_2']]

df_merged = pd.merge(df_cur1, df_cur2, on=['unix_timestamp', 'timestamp'], suffixes=['_' + cur1, '_' + cur2])
df_merged = df_merged.resample('H', on='timestamp').first()
df_merged = df_merged.fillna(method='ffill')
df_merged = df_merged.reset_index(drop=False)

df = df_merged[['open_price_' + cur1,'open_price_' + cur2]]
n_rows = int(len(df) * 0.75)
df = df[-n_rows:]

df.to_csv('data/processed_files/' + cur1 + '_' + cur2 + '_' + args.price_type + '.csv', index=False)