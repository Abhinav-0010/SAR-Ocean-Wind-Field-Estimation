import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SARWindResNet
from dataset import SARDataset

def train_model():
    print("Initializing Deep Learning Pipeline...")
    
    # 1. Setup Data Loaders (Using Tamil Nadu dataset as example)
    dataset = SARDataset(csv_file='../data/tn_training_metadata_2023_2024.csv', 
                         image_dir='../data/SAR_TN_Local_Dataset')
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 2. Initialize Model, Loss Function, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SARWindResNet().to(device)
    
    criterion = nn.MSELoss() # Mean Squared Error (best for vector regression)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Training Loop
    epochs = 10
    print(f"Starting training on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, true_vectors in dataloader:
            images = images.to(device)
            true_vectors = true_vectors.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass: Model guesses the wind vectors
            predictions = model(images)
            
            # Calculate error
            loss = criterion(predictions, true_vectors)
            
            # Backward pass: Update model weights
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(dataloader):.4f}")

    # 4. Save the trained model
    torch.save(model.state_dict(), '../outputs/models/sar_resnet_wind_model.pth')
    print("Training Complete. Model saved!")

if __name__ == "__main__":
    # NOTE: You need a GPU and time to actually run this. 
    # For project submission, showcasing the architecture is often enough.
    train_model()