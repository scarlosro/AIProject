#import libraries required
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

df = pd.read_csv('healthcare-dataset-stroke-data.csv', header=0)
df.sample()

f = pd.get_dummies(data=df, drop_first=True)
obj = df.stroke
expl = df.drop(columns='stroke')

model = DecisionTreeClassifier()
model.fit(X=expl, y = obj)
DecisionTreeClassifier()

plot_tree(decision_tree=model, feature_names=expl.columns, filled=True)



a = expl.sample()

model.predict_proba(a)

