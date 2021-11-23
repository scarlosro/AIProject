from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import category_encoders as enc
from decissiontreebyhand import DecisionTree

#names_rows = ['baseline value',	'accelerations','fetal_movement','uterine_contractions','light_decelerations',	'severe_decelerations',	'prolongued_decelerations','abnormal_short_term_variability',	'mean_value_of_short_term_variability',	'percentage_of_time_with_abnormal_long_term_variability',	'mean_value_of_long_term_variability',	'histogram_width',	'histogram_min',	'histogram_max', 'histogram_number_of_peaks',	'histogram_number_of_zeroes',	'histogram_mode', 'histogram_mean',	'histogram_median',	'histogram_variance','histogram_tendency','fetal_health']
names_rows = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df = pd.read_csv('IRIS.csv', skiprows=1, header=None, names=names_rows)


X= df.iloc[:,:-1].values

print(X)
y=df.iloc[:,-1].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=41)


classifier = DecisionTree(min_samples_split=3, max_depth=3)
classifier.fit(X_train,y_train)
classifier.print_tree()