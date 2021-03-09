from imutils import paths
import shutil
import env
import os

for split in (env.TRAIN_BASE, env.TEST_BASE, env.VALIDATION_BASE):
    print('Processing {} split...'.format(split))
    p = os.path.sep.join([env.ORIG_INPUT_DATASET, split])
    imagePaths = list(paths.list_images(p))
    
    for imagePath in imagePaths:
        filename = imagePath.split(os.path.sep)[-1]
        label = env.CLASSES[int(filename.split('_')[0])]
        
        dirPath = os.path.sep.join([env.BASE_PATH, split, label])
        
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
            
        p = os.path.sep.join([dirPath, filename])
        shutil.copy2(imagePath, p)