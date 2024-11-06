# main.py
from data.data import raw_df
from preprocess.Clean_Data import Clean_Data
from models.Random_Forest import Random_Forest
from preprocess.Data import Data
from utils.helper_function import helper_functions

def main():
    # Load the data
    print("Loading data...")
    data = raw_df
    
    # Preprocess the data
    print("Preprocessing data...")
    Clean_Data.Drop_Duplicates()
    Clean_Data.Fill_Na()
    Clean_Data.Data_Seperator()

    print("Taking out columns:")
    columns=dict()
    columns=Data.Columns()
    
    print("Encoding and Min_Max_Scaling")
    Data.Encoder()
    Data.Normalizer()

    #Build the Random Forest model
    print("Building and Training the model...")
    model=Random_Forest()
    
    print("Plotting feature importance...")
    helper_functions.count_plot()
    helper_functions.model_estimator()
    helper_functions.training_error()

if __name__ == "__main__":
    main()
