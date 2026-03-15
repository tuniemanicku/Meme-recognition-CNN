import os
import cv2
import numpy as np
from utils import *

def label_folder(path="memes"):
    images = os.listdir(path)
    labels = {}
    
    for i, image in enumerate(images):
        option = 0
        try:
            while True:
                im = cv2.imread(path+"/"+image)
                im = cv2.resize(im, dsize=(512,512))
                cv2.imshow(image, im)
                key = cv2.waitKey(10)
                if key >= ord('1') and key <= ord('0')+N_CLASSES:
                    cv2.destroyAllWindows()
                    option = key - ord('0')
                    break
        except:
            print(image)

        labels[image] = option
        # print(labels)
    with open(path+"_labels.txt", "w") as of:
        for image in labels:
            of.write(image+" "+str(int(labels[image]))+"\n")

def main():
    print("label train data: ")
    label_folder(TRAIN_PATH)
    print("label test data: ")
    label_folder(TEST_PATH)

if __name__ == "__main__":
    main()