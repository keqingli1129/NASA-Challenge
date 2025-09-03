import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import BoxLeastSquares
import lightkurve as lk
from astropy.io import fits
import os
from PIL import Image
import io
from lightcurvedataset import LightCurveDataset
from exoplanet import ExoplanetCNN

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 3. Data Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. Example usage with dummy data
def create_dummy_dataset():
    """Create a small dummy dataset for demonstration"""
    # In practice, you would have real FITS file paths and labels
    dummy_files = ['dummy_file_1.fits', 'dummy_file_2.fits'] * 5
    dummy_labels = [1, 0] * 5  # Alternating labels
    
    return LightCurveDataset(dummy_files, dummy_labels, transform=transform)

# 5. Training Function
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            predicted = (output.squeeze() > 0.5).float()
            correct += (predicted == target).sum().item()
            total += target.size(0)
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total
        print(f'Epoch {epoch+1} completed. Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    
    return model

# 6. Main execution
if __name__ == "__main__":
    # Create dummy dataset (replace with your actual data)
    dataset = create_dummy_dataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    # Initialize model, loss, and optimizer
    model = ExoplanetCNN().to(device)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train the model
    print("Starting training...")
    trained_model = train_model(model, dataloader, criterion, optimizer, num_epochs=5)
    
    # Save the trained model
    torch.save(trained_model.state_dict(), 'exoplanet_cnn.pth')
    print("Model saved as 'exoplanet_cnn.pth'")
    
    # Example prediction function
    def predict_exoplanet(fits_path, model, transform):
        """Predict if a FITS file contains an exoplanet"""
        model.eval()
        with torch.no_grad():
            # Create a single-item dataset
            dummy_dataset = LightCurveDataset([fits_path], [0], transform=transform)
            image, _ = dummy_dataset[0]
            image = image.unsqueeze(0).to(device)  # Add batch dimension
            
            output = model(image)
            probability = output.item()
            prediction = "Exoplanet" if probability > 0.5 else "No exoplanet"
            
            print(f"Prediction: {prediction} (Probability: {probability:.4f})")
            return probability
    
    # Example prediction (would need a real FITS file)
    # predict_exoplanet('your_file.fits', trained_model, transform)