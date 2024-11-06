import pandas as pd
from data import raw_df
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from Data import Data
class Clean_Data:
    
    def __init__(self, raw_df):
        self.raw_df=pd.read_csv(raw_df)
        self.train_data=None
        self.test_data=None
        self.val_data=None

        data_instance=Data(raw_df)
        self.numerical_cols=data_instance.numerical_cols
        self.categorical_cols=data_instance.categorical_cols

    def Drop_Duplicates(self, subset=None):
        if subset:
            self.raw_df.drop_duplicates(subset=subset, inplace=True)
        else:
            self.raw_df.drop_duplicates(subset=subset, inplace=True)

    def Fill_Na(self):
        imputer=SimpleImputer()
        imputer.fit(self.raw_df[self.numerical_cols])
        self.train_data[self.numerical_cols]=imputer.transform(self.train_data[self.numerical_cols])
        self.test_data[self.numerical_cols]=imputer.transform(self.test_data[self.numerical_cols])
        self.val_data[self.numerical_cols]=imputer.transform(self.val_data[self.numerical_cols])

    def Data_Seperator(self):
        train_val_data, self.test_data=train_test_split(raw_df, size=0.2, random_state=42)
        self.train_data, self.val_data=train_test_split(train_val_data, size=0.2, random_state=42)

        return self.train_data, self.test_data, self.val_data
