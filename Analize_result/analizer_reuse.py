import pandas as pd


df = pd.read_csv('./Data/driver_order_result.csv')


counts = df['driver_id'].value_counts().reset_index()
counts.columns = ['driver_id', 'count']


print(counts)