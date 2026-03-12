import os
import cv2
import numpy as np
PATH = "./memes"

images = os.listdir(PATH)

labels = {}

for i, image in enumerate(images):
    option = 0
    while True:
        im = cv2.imread(PATH+"/"+image)
        im = cv2.resize(im, dsize=(512,512))
        cv2.imshow(image, im)
        key = cv2.waitKey(10)
        if key == ord('1') or key == ord('2') or key == ord('3'):
            cv2.destroyAllWindows()
            option = key - ord('0')
            break

    labels[image] = option
    print(labels)
with open("labels.txt", "w") as of:
    for image in labels:
        of.write(image+" "+str(int(labels[image]))+"\n")