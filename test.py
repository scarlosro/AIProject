import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from pandas.core.common import SettingWithCopyWarning
import warnings 
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from decissiontreebyhand import DecisionTree
import category_encoders as enc
from sklearn.metrics import accuracy_score
import graphviz
from sklearn import tree
from sklearn.metrics import classification_report

def accuracy(y_true, y_pred):
    acc = np.sum(y_true == y_pred)/len(y_true)
    return accuracy

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

#read the data from the csv and delete blank spaces on columns names
df = pd.read_csv('car_evaluation.csv', header=None)
names_rows = ['buying','maint','doors','persons','lug_boot','safety','class']

df.columns = names_rows

X= df.drop(['class'],axis=1)
y=df['class']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

encoder = enc.OrdinalEncoder(cols=['buying','maint','doors','persons','lug_boot','safety'])

X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
clf_gini.fit(X_train, y_train)

#clf_handle = DecisionTree(max_depth=4)
#clf_handle.fit(X_train,y_train)

#y_pred_hand = clf_handle.predict(X_train, y_train)

y_pred_gini = clf_gini.predict(X_test)
print("Acuraccy with gini index was", accuracy_score(y_test, y_pred_gini))

y_predict_train_gini = clf_gini.predict(X_train)
print("training Acuraccy with gini index was", accuracy_score(y_train, y_predict_train_gini))

#acc= accuracy(y_test, y_pred_hand)
#print("Accuraccy was of ", acc)

treeFigure = plt.figure(figsize=(150, 155), dpi=42)

ex = plot_tree(clf_gini, feature_names=X_train.columns, rounded=True, class_names=y_train, filled=True)

treeFigure.savefig("decission.jpg")


dot_data = tree.export_graphviz(clf_gini, out_file=None, feature_names=X_train.columns, class_names=y_train, filled=True, rounded=True, special_characters=True)
graph=graphviz.Source(dot_data)


print(classification_report(y_test,y_pred_gini))



