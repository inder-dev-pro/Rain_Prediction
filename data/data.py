import pandas as pd
import os

data_dir='/content/drive/MyDrive/weatherAUS.csv'
os.listdir(os.path.dirname(data_dir))
raw_df=pd.read_csv(data_dir)
raw_df.info()
raw_df.describe()
raw_df.dropna(subset='RainTomorrow', inplace=True)