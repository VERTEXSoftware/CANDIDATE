import pandas as pd


df = pd.read_csv('./Data/driver_order_result.csv')


counts = df['driver_id'].value_counts().reset_index()
counts.columns = ['driver_id', 'count']


pd.set_option('display.max_rows', None)

print(counts)

pd.reset_option('display.max_rows')