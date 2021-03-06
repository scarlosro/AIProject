#import libraries required
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from pandas.core.common import SettingWithCopyWarning
import warnings 
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
import matplotlib.pyplot as plt

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

#read the data from the csv and delete blank spaces on columns names
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df.colums = df.columns.str.replace(' ','')
df= pd.get_dummies(df, columns=['gender','ever_married','work_type','Residence_type','smoking_status'])
df = clean_dataset(df)
#deleting information where a value is missed
df_rows_complete = df.loc[(df['bmi'] != 'N/A')]

obj = df_rows_complete['stroke']
expl = df_rows_complete.drop(columns='stroke', axis='columns')
expl_enc = expl
#expl_enc = pd.get_dummies(expl, columns=['gender','ever_married','work_type','Residence_type','smoking_status'])

X_train, X_test, y_train, y_test = train_test_split(expl_enc, obj, test_size=0.2)

model = DecisionTreeClassifier(min_samples_split=2, max_depth=3)

model.fit(X_train, y_train)
model.score(X_test, y_test)
DecisionTreeClassifier()

treeFigure = plt.figure(figsize=(15, 7.5))

ex = plot_tree(model, feature_names=expl_enc, rounded=True, class_names=["No stroke", "Stroke"], filled=True)

