# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
# Reading my images, transforming and reshaping to fit into the model.
traingenerate = ImageDataGenerator(rescale = 1./255)
testgenerate = ImageDataGenerator(rescale = 1./255)
training = traingenerate.flow_from_directory('mytraining',target_size = (64, 64),batch_size = 32,class_mode = 'binary')
test = testgenerate.flow_from_directory('mytesting',target_size = (64, 64),batch_size = 32,class_mode = 'binary')
# Reading my test image and reshaping it
testimage = image.load_img('dog.jpg', target_size = (64, 64))
testimage = image.img_to_array(testimage)
testimage = np.expand_dims(testimage, axis = 0)

# Creating a Sequential model
model = Sequential()
# adding 2 Dimensional convolutional neural nets layers
model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2))) # pool a max value in a 2x2 matrix
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
# Adding fully connected layers, first with rectifier activation and the next sigmoid
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))
# Minimize loss measure the model accuracy
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting model on training
model.fit_generator(training,steps_per_epoch=100,epochs=2,validation_data=test,validation_steps=10)
# making prediction on wether the input image is a dog or cat
predict_image = model.predict(testimage)
if predict_image[0][0] == 1:
    prediction = 'dog'
    print("Image is for dog")
else:
    prediction = 'cat'
    print("Image is for cat")