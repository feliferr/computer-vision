import torch
import argparse
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
import albumentations
import torch.optim as optim
import os

from torch.optim import optimizer
import cnn_models
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas as pd

matplotlib.style.use("ggplot")

from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, images, labels=None, tfms=None):
        self.X = images
        self.y = labels

        # augmentations
        if tfms == 0: # if validation is enabled
            self.aug = albumentations.Compose([
                albumentations.Resize(224, 224, always_apply=True),
            ])
        else: # if training is enabled
            self.aug = albumentations.Compose([
                albumentations.Resize(224, 224, always_apply=True),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.ShiftScaleRotate(
                    shift_limit=0.3,
                    scale_limit=0.3,
                    rotate_limit=15,
                    p=0.5,
                )
            ])
    
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, index):
        image = Image.open(self.X[index])
        image = image.convert("RGB")

        # applying augmentation
        image = self.aug(image=np.array(image))['image']

        # making channels-first (c, h, w)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        label = self.y[index]

        # getting image and labels as torch tensors
        tuple = (torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=torch.long))
        
        return tuple 

def plot_graph(train_metric, val_metric, metric_name, colors, output):
    plt.figure(figsize=(10,7))
    plt.plot(train_metric, color=colors[0], label=f"train {metric_name}")
    plt.plot(val_metric, color=colors[1], label=f"validation {metric_name}")
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(f"{output}/{metric_name}.png")
    plt.show()

def fit(model, 
        optimizer, 
        criterion, 
        total_instances, 
        train_dataloader, 
        device):

    print("Training model....")
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    for i, data in tqdm(enumerate(train_dataloader), total=int(total_instances/train_dataloader.batch_size)):
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()
    
    train_loss = train_running_loss/len(train_dataloader.dataset)
    train_accuracy = 100. * train_running_correct/len(train_dataloader.dataset)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:,.2f}")
    return train_loss, train_accuracy

def validate(model,
            criterion,
            total_instances, 
            test_dataloader, 
            device):

    print("Validating model....")
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_dataloader), total=int(total_instances/test_dataloader.batch_size)):
            data, target = data[0].to(device), data[1].to(device)
            outputs = model(data)
            loss = criterion(outputs, target)

            val_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            val_running_correct += (preds == target).sum().item()
        
        val_loss = val_running_loss/len(test_dataloader.dataset)
        val_accuracy = 100. * val_running_correct/len(test_dataloader.dataset)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}")
        return val_loss, val_accuracy

def main(input, 
          output, 
          epochs,
          learning_rate,
          batch_size,
          device):
    
    df = pd.read_csv(input)
    X = df['image_path'].values
    y = df['target'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=100)

    print(f"Training instances: {len(X_train)}")
    print(f"Test instances: {len(X_test)}")

    train_data = ImageDataset(X_train, y_train, tfms=1)
    test_data = ImageDataset(X_test, y_test, tfms=0)

    # dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # initializing the Neural Network model
    model = cnn_models.SimpleCNN().to("cpu:0") # TODO put a parameter to set de device
    print(model)

    # total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"{total_trainable_params:,} training parameters.")

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=5,
        factor=0.5,
        min_lr=1e-6,
        verbose=True
    )

    # Executing the training and validation
    train_loss, train_accuracy = [], []
    val_loss, val_accuracy = [], []
    start = time.time()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_accuracy = fit(model, 
                                                    optimizer, 
                                                    criterion, 
                                                    len(train_data), 
                                                    train_loader, 
                                                    device)

        val_epoch_loss, val_epoch_accuracy = validate(model, 
                                                    criterion, 
                                                    len(test_data), 
                                                    test_loader, 
                                                    device)
        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)
        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)

        scheduler.step(val_epoch_loss)
    end = time.time()

    print(f"{(end-start)/60:.3f} minutes")

    plot_graph(train_accuracy, val_accuracy, "accuracy", ['green', 'blue'], output)
    plot_graph(train_loss, val_loss, "loss", ['orange', 'red'], output)

    print("Saving model...")
    torch.save(model.state_dict(), os.path.join(output, "model"))

    print("Training Finished!!!")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Path to the csv data")
    ap.add_argument("-o", "--output", required=True, help="Path to save the trained model")
    ap.add_argument("-e","--epochs",type=int, default=75, help="Number of epochs to train the network")
    ap.add_argument("-lr","--learning_rate",type=float, default=1e-3, help="Learning rate to the network")
    ap.add_argument("-bs","--batch_size",type=int, default=32, help="Batch size to train the network")
    ap.add_argument("-d","--device", default="cpu:0", help="Device to be used in the training")

    args = ap.parse_args()

    arguments = args.__dict__
    
    main(**arguments)