from sklearn.ensemble import RandomForestClassifier
from matplotlib.pyplot import mat
from preprocess.Data import train_inputs, train_target, val_inputs, val_target
from sklearn.tree import plot_tree
import pandas as pd

def Random_Forest():
   model=RandomForestClassifier(n_jobs=-1,random_state=42)
   model.fit(train_inputs, train_target)
   model.score(train_inputs, train_target)
   model.score(val_inputs, val_target)
   model.predict_proba(train_inputs)
   mat.figure(figsize=(80,40))
   plot_tree(model.estimators_[19], max_depth=2, feature_names=train_inputs.columns, impurity=True, filled=True, fontsize=10)
   return model
def Estimator(est):
  model=RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=est).fit(train_inputs, train_target)
  train=1-model.score(train_inputs, train_target)
  val=1-model.score(val_inputs, val_target)
  return {'No. of estimators':est, 'Training_Error':train, 'Validation_Error':val}

for i in range(50, 201):
    esti=pd.DataFrame(Estimator(i) for i in range(50,100))

