
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
from keras.layers import Embedding
from keras.layers import SimpleRNN
# Reading the data into file
dataset = pd.read_csv('dress.csv')

X = dataset.iloc[:,1] # Style of the dress attributes
y = dataset.iloc[:,13] # Recommendation base on attribute
X=pd.get_dummies(X) # Encoding my text data using the pandas get dummies method
# Splitting the data into training and test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

embedding_size = 128
# Training
batch_size = 30
epochs = 2
max_features=600
# RNN model
model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=13))
model.add(Dropout(0.25))
model.add(SimpleRNN(64))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print('Train...')
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
