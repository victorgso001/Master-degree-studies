from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
import pickle
import env
import os

def load_data_split(splitPath):
    data = []
    labels = []
    
    for row in open(splitPath):
        row = row.strip().split(',')
        label = row[0]
        features = np.array(row[1:], dtype='float')
        
        data.append(features)
        labels.append(label)
        
    data = np.array(data)
    labels = np.array(labels)
    
    return (data, labels)

trainingPath = os.path.sep.join([env.BASE_CSV_PATH, '{}.csv'.format(env.TRAIN_BASE)])
testingPath = os.path.sep.join([env.BASE_CSV_PATH, '{}.csv'.format(env.TEST_BASE)])

print('Loading data...')
(trainX, trainY) = load_data_split(trainingPath)
(testX, testY) = load_data_split(testingPath)

le = pickle.loads(open(env.LENCODER_PATH, 'rb').read())

print("Training model...")
model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=150)
model.fit(trainX, trainY)

print('Evaluating...')
preds = model.predict(testX)
print(classification_report(testY, preds, target_names=le.classes_))

print('Saving model...')
f = open(env.MODEL_PATH, 'wb')
f.write(pickle.dumps(model))
f.close()