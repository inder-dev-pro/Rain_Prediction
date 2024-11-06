import pandas as pd
from data import raw_df
import numpy as np
from Clean_Data import Clean_Data

class Data:

    def __init__(self, raw_df):
        self.raw_df=pd.read_csv(raw_df)
        self.numerical_cols=None
        self.categorical_cols=None

        data_instance=Clean_Data(raw_df)
        self.train_data=data_instance.train_data
        self.val_data=data_instance.val_data
        self.test_data=data_instance.test_data
        
    def Columns(self):
        numerical_cols=self.raw_df.select_dtypes(include=np.number).columns.to_list()
        categorical_cols=self.raw_df.select_dtypes(include='object').columns.to_list()
        
        return numerical_cols, categorical_cols

    def Encoder(self):
        from sklearn.preprocessing import OneHotEncoder
        encoder=OneHotEncoder(handle_unknown='ignore')
        encoder.fit(self.raw_df[self.categorical_cols])
        names=list(encoder.get_feature_names_out(self.categorical_cols))

        self.train_data[names]=encoder.transform(self.train_data[self.categorical_cols]).toarray()
        self.test_data[names]=encoder.transform(self.test_data[self.categorical_cols]).toarray()
        self.val_data[names]=encoder.transform(self.val_data[self.categorical_cols]).toarray()
        
        self.train_data.drop(colums=self.categorical_cols, inplace=True)
        self.test_data.drop(colums=self.categorical_cols, inplace=True)
        self.val_data.drop(colums=self.categorical_cols, inplace=True)

        return self.train_data, self.test_data, self.val_data


    def Normalizer(self):
        from sklearn.preprocessing import MinMaxScaler
        scaler=MinMaxScaler()
        scaler.fit(raw_df[self.numerical_cols])
        self.train_data[self.numerical_cols]=scaler.transform(self.train_data[self.numerical_cols])
        return self.train_data
