import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
import albumentations
import torch.optim as optim
import os
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


def train(input, model, epochs):

    df = pd.read_csv(input)
    X = df['image_path'].values
    y = df['target'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=100)

    print(f"Training instances: {len(X_train)}")
    print(f"Test instances: {len(X_test)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Path to the csv data")
    ap.add_argument("-m", "--model", required=True, help="Path to save the trained model")
    ap.add_argument("-e","--epochs",type=int, default=75, help="Number of epochs to train the network")

    args = ap.parse_args()

    arguments = args.__dict__
    
    train(**arguments)