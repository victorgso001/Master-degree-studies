"""
Image selective search adapted from Adrian Rosebrock's blog post
available in https://www.pyimagesearch.com/2020/06/29/opencv-selective-search-for-object-detection/
Access in 2021-03-01
"""
import cv2
import argparse
import random
import time
from cv2.ximgproc.segmentation import createSelectiveSearchSegmentation

image = cv2.imread('dog.jpg')

selective_search = createSelectiveSearchSegmentation()
selective_search.setBaseImage(image)

selective_search.switchToSelectiveSearchQuality()

start = time.time()
rects = selective_search.process()
end = time.time()

print("Search took {:.4f} seconds".format(end - start))
print("Found {} total region proposals".format(len(rects)))

for i in range(0, len(rects), 100):
    output = image.copy()
    
    for (x, y, w, h) in rects[i:i + 100]:
        color = [random.randint(0, 255) for j in range(0, 3)]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        
    cv2.imshow("Output", output)
    key = cv2.waitKey(0) & 0xFF
    
    if (key == ord('q')):
        cv2.destroyAllWindows()
        break