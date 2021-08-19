import pandas as pd
from sklearn.tree import DecisionTreeClassifier

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre']) #input data set
y = music_data['genre'] #output data set

model = DecisionTreeClassifier()
model.fit(X, y)
predictions = model.predict([ [21, 1], [22, 0] ])  # predicts the type of music for a 21 year old male (male represented by the number 1) and 22 year old female (female represented by the number 0)
predictions  # the result is that a 21 year old male likes 'HipHop', and a 22 year old female like 'Dance'
