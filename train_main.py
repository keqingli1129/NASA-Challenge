import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import BoxLeastSquares
import lightkurve as lk
from astropy.io import fits
import os
from PIL import Image
import io
from lightcurvedataset import LightCurveDataset
from exoplanet import ExoplanetCNN
from exoplanetresnet import ExoplanetResNet
from nasa_main import load_json_to_dict
from torch.utils.tensorboard import SummaryWriter
import datetime
from torch.utils.data import random_split

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 3. Data Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. Example usage with dummy data
def create_dataset(data_folder):
    """Create a small dummy dataset for demonstration"""
    # In practice, you would have real FITS file paths and labels
    # dummy_files = ['dummy_file_1.fits', 'dummy_file_2.fits'] * 5
    # dummy_labels = [1, 0] * 5  # Alternating labels
    labels_dict = load_json_to_dict('merged_label_dict.json')
    fits_files = []
    labels = []
    """
    Counts ID folders under data_folder, and only plots .fits files from the first ID folder found.
    """
    id_folders = [item for item in os.listdir(data_folder)
                  if os.path.isdir(os.path.join(data_folder, item))]
    print(f"Found {len(id_folders)} ID folders in {data_folder}.")
    if not id_folders:
        print("No ID folders found.")
        return
    for id_folder in id_folders:
        id_path = os.path.join(data_folder, id_folder)
        for fname in os.listdir(id_path):
            if fname.lower().endswith('.fits'):
                fits_path = os.path.join(id_path, fname)
                fits_files.append(fits_path)
                labels.append(labels_dict.get(id_folder, 0))  # Get label for the ID folder
    return LightCurveDataset(fits_files, labels, transform=transform)

# 5. Training Function
def train_model(model, dataloader, criterion, optimizer, num_epochs=10, log_dir="exoplanet", stop_loss=None):
    model.train()
    best_loss = float('inf')
     # Add date and time to log folder name
    if log_dir is None:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"exoplanet_{now}"
    else:
        log_dir = f"{log_dir}_{now}"
    writer = SummaryWriter(log_dir=log_dir)  # TensorBoard writer

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
            
            # Log batch loss to TensorBoard
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Batch/Loss', loss.item(), global_step)
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total
        print(f'Epoch {epoch+1} completed. Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

        # Log epoch loss and accuracy to TensorBoard
        writer.add_scalar('Epoch/Loss', epoch_loss, epoch)
        writer.add_scalar('Epoch/Accuracy', epoch_acc, epoch)
        # Save model with loss in filename if loss improves
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            model_filename = f"exoplanet_cnn_loss_{best_loss:.4f}.pth"
            torch.save(model.state_dict(), model_filename)
            print(f"Model saved as '{model_filename}'")
        # Stop training if loss is below threshold
        if stop_loss is not None and epoch_loss <= stop_loss:
            print(f"Stopping early: loss {epoch_loss:.4f} reached threshold {stop_loss}")
            break
    writer.close()
    return model

# 6. Main execution
if __name__ == "__main__":
    # Create dataset (replace with your actual data)
    dataset = create_dataset('koi_data')
     # Split dataset into train and test sets (e.g., 80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders for train and test sets
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    # Initialize model, loss, and optimizer
    model = ExoplanetCNN().to(device)
    # using ResNet18
    # model = ExoplanetResNet().to(device)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train the model
    print("Starting training...")
    trained_model = train_model(model, train_loader, criterion, optimizer, num_epochs=5)
    
    # Save the trained model
    torch.save(trained_model.state_dict(), 'exoplanet_cnn.pth')
    print("Model saved as 'exoplanet_cnn.pth'")
    
    # Example prediction function
    def predict_exoplanet(test_loader, model):
        """
        Predict exoplanet probabilities for all samples in the test_loader.
        Prints prediction and probability for each sample.
        """
        model.eval()
        results = []
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(test_loader):
                data = data.to(device)
                outputs = model(data)
                probabilities = outputs.squeeze().cpu().numpy()
                for i, prob in enumerate(np.atleast_1d(probabilities)):
                    prediction = "Exoplanet" if prob > 0.5 else "No exoplanet"
                    print(f"Sample {batch_idx * test_loader.batch_size + i}: Prediction: {prediction} (Probability: {prob:.4f})")
                    results.append(prob)
        return results

    # Example usage after training:
    # predict_exoplanet(test_loader, trained_model)