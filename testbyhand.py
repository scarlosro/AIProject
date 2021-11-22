from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import category_encoders as enc
from decissiontreebyhand import DecisionTree

names_rows = ['buying','maint','doors','persons','lug_boot','safety','class']

df = pd.read_csv('car_evaluation.csv', skiprows=1, header=None, names=names_rows)
encoder = enc.OrdinalEncoder(cols=['buying','maint','doors','persons','lug_boot','safety'])
df = encoder.fit_transform(df)


X= df.iloc[:,:-1].values

print(X)
y=df.iloc[:,-1].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


classifier = DecisionTree(min_samples_split=3, max_depth=3)
classifier.fit(X_train,y_train)
classifier.print_tree()