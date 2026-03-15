import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from utils import *
import time

def load_labels(test=False):
    labels = {}
    if not test:
        with open("labels.txt","r") as file:
            label = file.readline()
            while label:
                txt = label.split()
                labels[txt[0]] = int(txt[1])-1
                label = file.readline()
    else:
        with open("test_labels.txt","r") as file:
            label = file.readline()
            while label:
                txt = label.split()
                labels[txt[0]] = int(txt[1])-1
                label = file.readline()
    return labels

class personDetector(nn.Module):
    def __init__(self): # tensors are: (batch_size, 3, 128, 128)
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5) # bs, 6, 124, 124
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2) # bs, 6, 62, 62
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=8, kernel_size=5) # bs, 8, 58, 58
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # bs, 8, 29, 29
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=12, kernel_size=4) # bs, 12, 26, 26
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # bs, 12, 13, 13
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=4) # bs, 16, 10, 10
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # bs, 16, 5, 5

        self.fc1 = nn.Linear(16*5*5, 100)
        self.fc2 = nn.Linear(100, 64)
        self.fc3 = nn.Linear(64, 3)

        self.relu = torch.relu
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.pool4(self.relu(self.conv4(x)))

        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    images = os.listdir(TRAIN_PATH)
    labels = load_labels()
    initial_list = []
    for image in images:
        try:
            initial_list.append(cv2.resize(cv2.imread(TRAIN_PATH+"/"+image), dsize=(128,128)))
        except:
            print(image)
    temp = np.array(initial_list)
    temp = np.transpose(temp, (0, 3, 1, 2))
    image_data = torch.tensor(temp, dtype=torch.float)

    #Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #data loader
    dataset = TensorDataset(image_data, torch.tensor([labels[key] for key in labels]))

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    # for batch_images, batch_labels in dataloader:
    #     print(f"Batch shape: {batch_images.shape}, Labels: {batch_labels}")

    #model
    model = personDetector().to(device=device)
    lossFunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    #train
    training_start = time.time()
    print("started")
    for epoch in range(N_EPOCHS):
        epoch_start = time.time()
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = lossFunction(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if epoch % 5 == 0:
            print(f"epoch: {epoch}, time: {time.time() - epoch_start}")
    print(f"training time: {time.time() - training_start}")

    test_images = os.listdir(TEST_PATH)
    test_labels = load_labels(test=True)
    initial_list = []
    for image in test_images:
        try:
            initial_list.append(cv2.resize(cv2.imread(TEST_PATH+"/"+image), dsize=(128,128)))
        except:
            print(image)
    temp = np.array(initial_list)
    temp = np.transpose(temp, (0, 3, 1, 2))
    test_image_data = torch.tensor(temp, dtype=torch.float)
    print(test_image_data.shape)

    #data loader
    test_dataset = TensorDataset(test_image_data, torch.tensor([test_labels[key] for key in test_labels]))

    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = np.array([0 for i in range(3)])
        n_class_samples = np.array([0 for i in range(3)])

        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (pred == labels).sum().item()
            print("labels:",labels)
            print("predicted:",pred)
            for i in range(min(BATCH_SIZE, len(labels))):
                label = labels[i]
                pre = pred[i]
                if (label == pre):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1
        acc = 100.0 * (n_correct / n_samples)
        print(f"acc is: {acc}")

        for i in range(3):
            acc = 100.0 * (n_class_correct[i] / n_class_samples[i])
            print(f"class {i} acc is: {acc}")

if __name__ == "__main__":
    main()