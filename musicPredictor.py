import pandas as pd
from sklearn.tree import DecisionTreeClassifier

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre']) #input data set
y = music_data['genre'] #output data set

model = DecisionTreeClassifier()
model.fit(X, y)
predictions = model.predict([ [26, 1], [29, 0] ])
predictions