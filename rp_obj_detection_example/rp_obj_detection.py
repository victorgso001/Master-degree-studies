"""
Image selective search adapted from Adrian Rosebrock's blog post
available in https://www.pyimagesearch.com/2020/07/06/region-proposal-object-detection-with-opencv-keras-and-tensorflow/
Access in 2021-03-01
"""

import cv2
import numpy as np
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import img_to_array
from cv2.ximgproc.segmentation import createSelectiveSearchSegmentation
from imutils.object_detection import non_max_suppression

def selective_search(image):
    ss = createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()

    rects = ss.process()

    return rects

print('Loading ResNet...')
model = ResNet50(weights = 'imagenet')

image = cv2.imread('dog.jpg')
(H, W) = image.shape[:2]

print('Selective search initializing...')
rects = selective_search(image)
print('Found {} regions.'.format(len(rects)))

proposals = []
boxes = []

for (x, y, w, h) in rects:
    #If the region < 10% size of the image, ignore it.
    if w/float(W) < 0.1 or h/float(H) < 0.1:
        continue

    #Get the regions, convert BGR to RGB and resize to fit the input dimensions
    #of the ResNet50 imagenet pre-trained CNN (224 x 224 pixels)
    roi = image[y:y + h, x:x + w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (224, 224))

    roi = img_to_array(roi)
    roi = preprocess_input(roi)

    #Add the proposal and bounding boxes to the list
    proposals.append(roi)
    boxes.append((x, y, w, h))

proposals = np.array(proposals)
print("Proposals shape: {}".format(proposals.shape))

print("Classifying proposals...")
preds = model.predict(proposals)
preds = imagenet_utils.decode_predictions(preds, top = 1)

labels = {}

for (i, p) in enumerate(preds):
    (imagenet_id, label, prob) = p[0]

    if prob >= 0.9:
        (x, y, w, h) = boxes[i]
        box = (x, y, x + w, y + h)

        new_label = labels.get(label, [])
        new_label.append((box, prob))
        labels[label] = new_label

for label in labels.keys():
    print("Results for: {}".format(label))
    clone = image.copy()

    #Image before applying Non-Maximum Supression (NMS)
    for (box, prob) in labels[label]:
        (startX, startY, endX, endY) = box
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 0, 255), 2)

    cv2.imshow("Before", clone)
    clone = image.copy()

    #Preparing and applying NMS
    boxes = np.array([p[0] for p in labels[label]])
    proba = np.array([p[1] for p in labels[label]])
    boxes = non_max_suppression(boxes, proba)

    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 0, 255), 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(clone, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 255), 2)

    #Image after NMS
    cv2.imshow("After", clone)
    cv2.waitKey(0)

cv2.destroyAllWindows()
