import os
import cv2
import numpy as np
from utils import *

images = os.listdir(TEST_PATH)

labels = {}

for i, image in enumerate(images):
    option = 0
    while True:
        im = cv2.imread(TEST_PATH+"/"+image)
        im = cv2.resize(im, dsize=(512,512))
        cv2.imshow(image, im)
        key = cv2.waitKey(10)
        if key == ord('1') or key == ord('2') or key == ord('3'):
            cv2.destroyAllWindows()
            option = key - ord('0')
            break

    labels[image] = option
    print(labels)
with open("test_labels.txt", "w") as of:
    for image in labels:
        of.write(image+" "+str(int(labels[image]))+"\n")