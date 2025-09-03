import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../tactile files')))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from touch_rest_comparison import get_touch_data
from box_plots import filter_bad_data
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # (1, 22, 1001) -> (32, 22, 1001)
        self.pool = nn.MaxPool2d(2, 2) #(32, 22, 1001) -> (32, 11, 500)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # (32, 11, 500) -> (64, 11, 500)
        #pool2: (64, 11, 500) -> (64, 5, 250)
        self.fc1 = nn.Linear(64 * 5 * 250, 128) # changed because after first 2 layers: grayscale: 1 -> 64, channels: 22 -> 5, times: 1001 -> 250
        self.fc2 = nn.Linear(128, 12) #because there's 12 labels
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 250)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class EEGDataset(Dataset):
    def __init__(self, data, labels, mean=None, std=None):
        self.data = torch.stack(data)  # shape: (N, 22, 1001)
        self.labels = torch.tensor(labels)
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].float()
        y = self.labels[idx]
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, 22, 1001]
        if self.mean is not None and self.std is not None:
            x = (x - self.mean[None, :, None]) / self.std[None, :, None]
        return x, y
    
def preprocess_eeg_data():
    """
    Preprocesses EEG data. 
    Tensor form: 22 (channels) x 1001 (time points)
    for 32 subjects x 12 conditions = 300 trials
    """
    label_map = {
        "Air1": 0, "Air2": 1, "Air3": 2, "Air4": 3,
        "Car1": 4, "Car2": 5, "Car3": 6, "Car4": 7,
        "Vib1": 8, "Vib2": 9, "Vib3": 10, "Vib4": 11
    }
    data_list = []
    label_list = []

    for condition_name, label in label_map.items():
        if label == 2:
            break
        eeg_data, _ = get_touch_data(condition_name)  # shape: [34, 22, 1001]
        filtered, _, _ = filter_bad_data(eeg_data, eeg_data) # [25, 22, 1001]
        for trial in filtered:
            # Convert each trial to tensor (no channel dimension)
            tensor = torch.tensor(trial, dtype=torch.float32)  # [22, 1001]
            data_list.append(tensor)
            label_list.append(label)
    return data_list, label_list

def load_data(train_data, train_labels, test_data, test_labels):
    train_tensor = torch.stack(train_data)  # (N, 22, 1001)
    
    #Calculate mean and std per channel (across all samples and time points)
    mean = train_tensor.mean(dim=(0, 2))  # shape: (22,)
    std = train_tensor.std(dim=(0, 2))    # shape: (22,)
    
    # Create Datasets with normalization
    train_dataset = EEGDataset(train_data, train_labels, mean, std)
    test_dataset = EEGDataset(test_data, test_labels, mean, std)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

def train_cnn(model, train_loader, device, test_loader=None, epochs=5, lr=0.008):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

def evaluate_cnn(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

def main():
    # Example: EEG data preprocessing
    eeg_data, eeg_labels = preprocess_eeg_data() 
    print(f"PREPROCESSING COMPLETE. \nEEG data: {len(eeg_data)} samples, labels: {len(eeg_labels)}")
    train_data, test_data, train_labels, test_labels = train_test_split(
    eeg_data, eeg_labels, test_size=0.2, stratify=eeg_labels, random_state=42
    ) #splits the data into training and test
    train_loader, test_loader = load_data(train_data, train_labels, test_data, test_labels) #loads data into dataloader
    print("DATA LOADING COMPLETE")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    train_cnn(model, train_loader, device)
    evaluate_cnn(model, test_loader, device)

if __name__ == "__main__":
    main()