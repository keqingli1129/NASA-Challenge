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

# 1. Define the Dataset Class
class LightCurveDataset(Dataset):
    def __init__(self, fits_files, labels, transform=None, img_size=(128, 128)):
        """
        Args:
            fits_files (list): List of paths to FITS files
            labels (list): Corresponding labels (1 for exoplanet, 0 for none)
            transform (callable, optional): Optional transform to be applied on image
            img_size (tuple): Size of the output image
        """
        self.fits_files = fits_files
        self.labels = labels
        self.transform = transform
        self.img_size = img_size
        
    def __len__(self):
        return len(self.fits_files)
    
    def __getitem__(self, idx):
        fits_path = self.fits_files[idx]
        label = self.labels[idx]
        
        try:
            # Load and process light curve
            with fits.open(fits_path) as hdul:
                # Simple extraction - in practice, use lightkurve for robust processing
                data = hdul[1].data
                time = data['TIME']
                flux = data['PDCSAP_FLUX']
                
                # Remove NaNs
                mask = np.isfinite(time) & np.isfinite(flux)
                time = time[mask]
                flux = flux[mask]
                
                # Normalize
                flux = flux / np.nanmedian(flux)
                
            # Create folded light curve image
            image = self.create_folded_image(time, flux)
            
            if self.transform:
                image = self.transform(image)
                
            return image, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error processing {fits_path}: {e}")
            # Return a dummy image and label
            dummy_image = torch.zeros(3, self.img_size[0], self.img_size[1])
            return dummy_image, torch.tensor(0.0, dtype=torch.float32)
    
    def create_folded_image(self, time, flux):
        """Create a folded light curve image using BLS periodogram"""
        try:
            # Find period using BLS
            model = BoxLeastSquares(time, flux)
            periodogram = model.autopower(0.1)  # Assume 0.1-day duration
            best_period = periodogram.period[np.argmax(periodogram.power)]
            
            # Fold the light curve
            folded_time = (time % best_period) / best_period
            
            # Create figure in memory
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 4))
            
            # Plot periodogram
            ax1.plot(periodogram.period, periodogram.power, 'b-', linewidth=1)
            ax1.axvline(best_period, color='r', linestyle='--', alpha=0.7)
            ax1.set_title(f'Period: {best_period:.2f} days')
            ax1.set_xlabel('Period (days)')
            ax1.set_ylabel('Power')
            
            # Plot folded light curve
            ax2.plot(folded_time, flux, 'k.', markersize=2, alpha=0.7)
            ax2.set_xlabel('Phase')
            ax2.set_ylabel('Normalized Flux')
            
            plt.tight_layout()
            
            # Save figure to memory buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=50, bbox_inches='tight')
            buf.seek(0)
            
            # Convert to PIL Image and resize
            image = Image.open(buf).convert('RGB')
            image = image.resize(self.img_size)
            plt.close(fig)
            
            return image
            
        except Exception as e:
            print(f"Error creating image: {e}")
            # Return blank image if processing fails
            return Image.new('RGB', self.img_size, color='white')