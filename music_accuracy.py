import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])  #input data set
y = music_data['genre']  #output data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # split the dataset into two sets, one for training and one for testing with 80% of the data for training, and 20% for testing

model = DecisionTreeClassifier()
model.fit(X_train, y_train)  # passes the training dataset to train the model
predictions = model.predict(X_test) 

score = accuracy_score(y_test, predictions)  # y_test contains the expected values, predictions contains the actual value
score  # displays an accuracy score between 0 and 1 (between 0% and 100%)
