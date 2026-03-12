import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from utils import *
import time

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
    def __init__(self, n_input, n_classes): #TODO - design layers and forward
        super(personDetector, self).__init__()
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
    dataset = TensorDataset(image_data, torch.tensor([labels[key] for key in labels]))

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    # for batch_images, batch_labels in dataloader:
    #     print(f"Batch shape: {batch_images.shape}, Labels: {batch_labels}")

    #model
    model = personDetector(n_input=1, n_classes=3)
    lossFunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    #train
    training_start = time.time()
    print("started")
    for epoch in range(N_EPOCHS):
        epoch_start = time.time()
        for i, (images, labels) in enumerate(dataloader):
            outputs = model(images)
            loss = lossFunction(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if epoch % 5 == 0:
            print(f"epoch: {epoch}, time: {time.time() - epoch_start}")
    print(f"training time: {time.time() - training_start}")

    #evaluate TODO: add test data and evaluate accuracy
    # test_dataloader
    # output = model(test_x)
    # labels_pred = argmax(output)+1
    # accuracy = np.mean(test_labels == labels_pred)*100
if __name__ == "__main__":
    main()