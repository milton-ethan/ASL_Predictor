import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import LabelEncoder

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 26)  # For 26 letters (a-z)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
        x = x.view(-1, 128 * 3 * 3)  # Flatten layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # Output layer (softmax is applied in loss function)
        return x

# Custom dataset class to load the data
class SignLanguageDataset(Dataset):
    def __init__(self, data_path):
        data = pd.read_csv(data_path)
        # Separate the labels and features
        self.labels = data.iloc[:, 0].values
        self.features = data.iloc[:, 1:].values
        self.features = self.features.reshape(-1, 28, 28)  # Reshaping to (28, 28)
        self.features = self.features / 255.0  # Normalize to [0, 1]
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx].unsqueeze(0), self.labels[idx]  # Add channel dimension

# Load data
train_dataset = SignLanguageDataset('sign_mnist_train.csv')
test_dataset = SignLanguageDataset('sign_mnist_test.csv')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model, loss function, and optimizer
model = CNNModel()
loss_function = nn.CrossEntropyLoss()  # Multi-class classification loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Check if the model file exists
model_filename = 'sign_language_model.pth'
if os.path.exists(model_filename):
    # Load the model
    model.load_state_dict(torch.load(model_filename))
    print("Model loaded successfully!")
else:
    print("Model not found. Training a new model.")
    # Training function
    def train_model():
        num_epochs = 5
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Accuracy: {correct/total*100}%')

    # Train the model
    train_model()
    
    # Save the trained model
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as {model_filename}")

# Testing function
def test_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {correct/total*100}%')

# Test the model
test_model()

# Letters array (a-z)
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def visualize_and_predict():
    idx = np.random.randint(len(test_dataset))
    image, label = test_dataset[idx]
    image = image.squeeze().numpy()  # Remove any extra dimensions

    # Predict the letter
    outputs = model(torch.tensor(image).unsqueeze(0).unsqueeze(0))  # Add batch and channel dimensions
    _, predicted = torch.max(outputs, 1)
    predicted_label = letters[predicted.item()]
    correct_label = letters[label]

    # Resize the image to fit into the left half of the canvas (640x640)
    image_resized = cv2.resize(image, (640, 640))  # Resize the image to 640x640

    # If the image is grayscale, ensure it has values between 0 and 255
    image_resized = np.uint8(image_resized * 255)  # Scale the image to 0-255 if it's in the 0-1 range

    # Create the window with two parts: left for image, right for text
    img_display = np.zeros((640, 1280), dtype=np.uint8)  # Creating a blank canvas with size 640x1280
    img_display[:, :640] = image_resized  # Left half for the resized image

    # Set right half to white for the text area
    img_display[:, 640:] = 255  # Set right half to white

    # Print text to the console
    print(f"Predicted Letter: {predicted_label}")
    print(f"Correct Answer: {correct_label}")

    # Set font size and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2  # Larger font size
    thickness = 2  # Thicker text for better visibility

    # Calculate text size for centering
    text_predicted = f'Guess Letter: {predicted_label}'
    text_correct = f'Correct Answer: {correct_label}'
    (w_pred, h_pred), _ = cv2.getTextSize(text_predicted, font, font_scale, thickness)
    (w_correct, h_correct), _ = cv2.getTextSize(text_correct, font, font_scale, thickness)

    # Position the text in the middle of the right half
    x_pred = 640 + (640 - w_pred) // 2  # Center the 'Predicted' text in the right half
    y_pred = 160  # Vertical position of the 'Predicted' text
    x_correct = 640 + (640 - w_correct) // 2  # Center the 'Correct' text in the right half
    y_correct = 240  # Vertical position of the 'Correct' text

    # Display the text on the image (centered in the right half)
    cv2.putText(img_display, text_predicted, (x_pred, y_pred), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(img_display, text_correct, (x_correct, y_correct), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # Display the image and wait for space bar press
    cv2.imshow("Sign Language Letter Prediction", img_display)
    key = cv2.waitKey(0)

    if key == 32:  # Spacebar key press
        visualize_and_predict()  # Recursively display the next guess

# Call function to start visualizing and making predictions
visualize_and_predict()

# Close OpenCV windows
cv2.destroyAllWindows()
