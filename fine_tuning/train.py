from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers import Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import env
import os

matplotlib.use('Agg')

def plot_training(H, N, plotPath):
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arrange(0, N), H.history['loss'], label = 'train_loss')
    plt.plot(np.arrange(0, N), H.history['val_loss'], label = 'val_loss')
    plt.plot(np.arrange(0, N), H.history['accuracy'], label = 'train_acc')
    plt.plot(np.arrange(0, N), H.history['val_accuracy'], label = 'val_acc')
    plt.title('Training loss and accuracy')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.savefig(plotPath)

trainPath = os.path.sep.join([env.BASE_PATH, env.TRAIN_BASE])
testPath = os.path.sep.join([env.BASE_PATH, env.TEST_BASE])
valPath = os.path.sep.join([env.BASE_PATH, env.VALIDATION_BASE])

totalTrain = len(list(paths.list_images(trainPath)))
totalVal = len(list(paths.list_images(valPath)))
totalTest = len(list(paths.list_images(testPath)))

trainAug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=(True),
    fill_mode='nearest')

valAug = ImageDataGenerator()

mean = np.array([123.68, 116,779, 103.939], dtype='float32')
trainAug.mean = mean
valAug.mean = mean

trainGen = trainAug.flow_from_directory(
    trainPath,
    class_mode='categorical',
    target_size=(224, 224),
    color_mode='rgb',
    shuffle=True,
    batch_size=env.BATCH_SIZE)

valGen = valAug.flow_from_directory(
    valPath,
    class_mode='categorical',
    target_size=(224, 224),
    color_mode='rgb',
    shuffle=False,
    batch_size=env.BATCH_SIZE)

testGen = valAug.flow_from_directory(
    valPath,
    class_mode='categorical',
    target_size=(224, 224),
    color_mode='rgb',
    shuffle=False,
    batch_size=env.BATCH_SIZE)

baseModel = VGG16(weights='imagenet', include_top=False,
                  input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = Flatten(name='Flatten')(headModel)
headModel = Dense(512, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(env.CLASSES), activation='softmax')(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

#freezing all conv layers in the body of VGG16
for layer in baseModel.layers:
    layer.trainable = False

print('Compiling model...')
opt = SGD(lr=1e-4, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])

print('Training head...')
H = model.fit(
    x = trainGen,
    steps_per_epoch=totalTrain // env.BATCH_SIZE,
    validation_data=valGen,
    validation_steps=totalVal // env.BATCH_SIZE,
    epochs=50)

print('Evaluating after fine-tuning...')
testGen.reset()
predIdxs = model.predict(x=testGen, steps=(totalTest // env.BATCH_SIZE + 1))
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testGen.classes, predIdxs,
                            target_names=testGen.class_indices.keys()))
plot_training(H, 50, env.WARMUP_PLOT_PATH)

trainGen.reset()
valGen.reset()

for layer in baseModel.layers[15:]:
    layer.trainable = True
    
for layer in baseModel.layers:
    print('{}: {}'.format(layer, layer.trainable))
    
print('Recompiling model...')
opt = SGD(lr=1e-4, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])

H = model.fit(x=trainGen,
              steps_per_epoch=totalTrain // env.BATCH_SIZE,
              validation_data=valGen,
              validation_steps=totalVal // env.BATCH_SIZE,
              epochs=20)

print('Evaluating after fine-tuning')
testGen.reset()
predIdxs = model.predict(x=testGen,
                          steps=(totalTest // env.BATCH_SIZE + 1))
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testGen.classes, predIdxs,
                            target_names=testGen.class_indices.keys()))
plot_training(H, 20, env.UNFROZEN_PLOT_PATH)

print('Serializing network...')
model.save(env.MODEL_PATH, save_format='h5')