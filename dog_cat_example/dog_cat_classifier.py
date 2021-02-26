import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import os

root_folder = os.path.dirname(os.path.realpath(__file__))

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                    metrics = ['accuracy'])

train_generator = ImageDataGenerator(rescale = 1./255,
                                    rotation_range = 7,
                                    horizontal_flip = True,
                                    shear_range = 0.2,
                                    height_shift_range = 0.07,
                                    zoom_range = 0.2)

test_generator = ImageDataGenerator(rescale = 1./255)

train_base = train_generator.flow_from_directory(os.path.join(root_folder, 'training_set'),
                                                  target_size = (64, 64),
                                                  batch_size = 32,
                                                  class_mode = 'binary')

test_base = test_generator.flow_from_directory(os.path.join(root_folder, 'test_set'),
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')

classifier.fit_generator(train_base, steps_per_epoch = 4000/32,
                          epochs = 100, validation_data = test_base,
                          validation_steps = 1000/32)

test_image = image.load_img(os.path.join(root_folder, 'test_set\\cachorro\\dog.3521.jpg'),
                            target_size = (64, 64))

test_image = image.img_to_array(test_image)
test_image /= 255
test_image = np.expand_dims(test_image, axis = 0)

predicted = classifier.predict(test_image)
predicted = 'dog' if predicted > 0.5 else 'cat'
