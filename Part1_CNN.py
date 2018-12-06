
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
# reading my data file into a data frame
dataset = pd.read_csv('dress.csv')

X = dataset.iloc[:,1] # my features describe the style of the dress
y = dataset.iloc[:,13] # target / recommendation of a dress either 0 or 1
X=pd.get_dummies(X) # Encoding my text data/features
# splitting the data into training and test set
from sklearn.model_selection import train_test_split
# Split in ratio of 20:80 for test and training respectively
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Initializing my model parameters
embedding_size = 128
# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 30

# Training
batch_size = 30
epochs = 2
max_features=500
# Creating a sequential model and feeding my data into the model
model = Sequential() # Instance of sequential model
model.add(Embedding(max_features, embedding_size, input_length=13))
model.add(Dropout(0.25))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='sigmoid',
                 strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('sigmoid'))
# Mininize loss and evaluate model's accuracy
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# Fitting the model using training data and validating using test data
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
