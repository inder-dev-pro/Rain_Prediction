from data import raw_df
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as mat
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from preprocess.Data import train_inputs
from models.Random_Forest import esti
class helper_functions:
    def __init__(self) -> None:
        self.raw_df=raw_df

    def count_plot(self):
        years=pd.to_datetime(self.raw_df['Date']).dt.year
        sea.countplot(x=years)

    def model_estimator(self):
        model=RandomForestClassifier(n_jobs=-1,random_state=42)
        mat.figure(figsize=(80,40))
        plot_tree(model.estimators_[19], max_depth=2, feature_names=train_inputs.columns, impurity=True, filled=True, fontsize=10)

    def training_error(self):
        from matplotlib import pyplot as plt
        esti.plot(kind='scatter', x='No. of estimators', y='Training_Error', s=32, alpha=.8)
        plt.gca().spines[['top', 'right',]].set_visible(False)