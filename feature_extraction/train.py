from sklearn.metrics import classification_report
from keras.utils import to_categorical
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense
import numpy as np
import pickle
import env
import os

def csv_feature_generator(inputPath, bs, numClasses, mode = 'train'):
    f = open(inputPath, 'r')
    
    while True:
        data = []
        labels = []
        
        while len(data) < bs:
            row = f.readline()
            
            if row == '':
                f.seek(0)
                row = f.readline()
                
                if mode == 'eval':
                    break
            
            row = row.strip().split(',')
            label = row[0]
            label = to_categorical(label, num_classes = numClasses)
            features = np.array(row[1:], dtype = 'float')
            
            data.append(features)
            labels.append(label)
            
        yield(np.array(data), np.array(labels))
        
le = pickle.loads(open(env.LENCODER_PATH, 'rb').read())

trainPath = os.path.sep.join([env.BASE_CSV_PATH, '{}.csv'.format(env.TRAIN_BASE)])
valPath = os.path.sep.join([env.BASE_CSV_PATH, '{}.csv'.format(env.VALIDATION_BASE)])
testPath = os.path.sep.join([env.BASE_CSV_PATH, '{}.csv'.format(env.TEST_BASE)])

totalTrain = sum([1 for l in open(trainPath)])
totalVal = sum([1 for l in open(valPath)])

testLabels = [int(row.split(',')[0]) for row in open(testPath)]
totalTest = len(testLabels)

trainGen = csv_feature_generator(trainPath, env.BATCH_SIZE, len(env.CLASSES), mode = 'train')

valGen = csv_feature_generator(valPath, env.BATCH_SIZE, len(env.CLASSES), mode = 'eval')

testGen = csv_feature_generator(testPath, env.BATCH_SIZE, len(env.CLASSES), mode = 'eval')

model = Sequential()
model.add(Dense(256, input_shape = (7 * 7 * 2048,), activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(len(env.CLASSES), activation = 'softmax'))

opt = SGD(lr = 1e-3, momentum=0.9, decay = 1e-3 / 25)

model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])

print('Training simple network...')
H = model.fit(x = trainGen, steps_per_epoch = totalTrain // env.BATCH_SIZE,
              validation_data = valGen, validation_steps = totalVal // env.BATCH_SIZE,
              epochs = 25)

print('Evaluationg network...')
predIxs = model.predict(x = testGen, steps = (totalTest // env.BATCH_SIZE) + 1)
predIxs = np.argmax(predIxs, axis = 1)
print(classification_report(testLabels, predIxs, target_names = le.classes_))