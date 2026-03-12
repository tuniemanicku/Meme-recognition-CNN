import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

PATH = "./memes"

def load_labels():
    labels = {}
    with open("labels.txt","r") as file:
        label = file.readline()
        while label:
            txt = label.split()
            labels[txt[0]] = int(txt[1])
            label = file.readline()
    return labels

class personDetector(nn.Module):
    def __init__(self, n_input, n_output): #
        super().__init__()
        self.relu = torch.relu
        self.input_layer = nn.Conv2d(3, 3, kernel_size=(4,4))
        self.maxPooling = nn.MaxPool2d(kernel_size=(2,2))

    def forward(self, x):
        x = self.input_layer(x)

def main():
    images = os.listdir(PATH)
    labels = load_labels()
    initial_list = []
    for image in images:
        try:
            initial_list.append(cv2.resize(cv2.imread(PATH+"/"+image), dsize=(512,512)))
        except:
            print(image)
    image_data = torch.tensor(np.array(initial_list))

    #data loader
    dataset = TensorDataset(image_data)


    #model
    #train
    #evaluate

if __name__ == "__main__":
    main()