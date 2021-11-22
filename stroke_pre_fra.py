#import libraries required
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from pandas.core.common import SettingWithCopyWarning
import warnings 
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

#read the data from the csv and delete blank spaces on columns names
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df.colums = df.columns.str.replace(' ','')

#deleting information where a value is missed
df_rows_complete = df.loc[(df['bmi'] != 'N/A')]

obj = df_rows_complete['stroke']
expl = df_rows_complete.drop(columns='stroke', axis='columns')

expl_enc = pd.get_dummies(expl, columns=['gender','ever_married','work_type','Residence_type','smoking_status'])

target_not_zero_index = obj > 0
obj[target_not_zero_index] = 1


X_train, X_test, y_train, y_test = train_test_split(expl_enc, obj, test_size=0.2)

model = DecisionTreeClassifier(min_samples_split=2, max_depth=3)

model.fit(X_train, y_train)
model.score(X_test, y_test)
DecisionTreeClassifier()

treeFigure = plt.figure(figsize=(20, 10))

ex = plot_tree(model, feature_names=expl_enc.columns, rounded=True, class_names=["No stroke", "Stroke"], filled=True)

fig.savefig("decission.png")

