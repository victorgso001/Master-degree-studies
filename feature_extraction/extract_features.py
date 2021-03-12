from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input
from keras.applications import ResNet50
from sklearn.preprocessing import LabelEncoder
from imutils import paths
import numpy as np
import pickle
import random
import env
import os

print('Loading network...')
model = ResNet50(weights='imagenet', include_top=False)
le = None

for split in (env.TRAIN_BASE, env.TEST_BASE, env.VALIDATION_BASE):
    print('Processing {} split...'.format(split))
    p = os.path.sep.join([env.BASE_PATH, split])
    imagePaths = list(paths.list_images(p))
    
    random.shuffle(imagePaths)
    labels = [p.split(os.path.sep)[-2] for p in imagePaths]
    
    if le is None:
        le = LabelEncoder()
        le.fit(labels)
    
    csvPath = os.path.sep.join([env.BASE_CSV_PATH, '{}.csv'.format(split)])
    csv = open(csvPath, 'w')
    
    for (b, i) in enumerate(range(0, len(imagePaths), env.BATCH_SIZE)):
        print('Processing batch {}/{}'.format(b + 1,
            int(np.ceil(len(imagePaths) / float(env.BATCH_SIZE)))))
        batchPaths = imagePaths[i:i + env.BATCH_SIZE]
        batchLabels = le.transform(labels[i:i + env.BATCH_SIZE])
        batchImages = []
        
        for imagePath in batchPaths:
            #Resizing images and converting to array
            image = load_img(imagePath, target_size = (224,224))
            image = img_to_array(image)
            
            #Image preprocessing with dimension expansion and mean 
            #RGB pixel intensity subtraction
            image = np.expand_dims(image, axis = 0)
            image = preprocess_input(image)
            
            batchImages.append(image)
            
        batchImages = np.vstack(batchImages)
        features = model.predict(batchImages, batch_size = env.BATCH_SIZE)
        features = features.reshape((features.shape[0], 7 * 7 * 2048))
        
        for (label, vec) in zip(batchLabels, features):
            vec = ','.join([str(v) for v in vec])
            csv.write('{},{}\n'.format(label, vec))
    
    csv.close()
    
f = open(env.LENCODER_PATH, 'wb')
f.write(pickle.dumps(le))
f.close()