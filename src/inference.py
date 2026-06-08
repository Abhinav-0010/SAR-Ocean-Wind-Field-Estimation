import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
# Assuming model.py is in the same directory
from model import SARWindResNet

def load_trained_model(model_path):
    """Loads the pre-trained ResNet model from disk."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SARWindResNet()
    
    # Load the trained weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval() # Set model to evaluation (inference) mode
        print("Model loaded successfully.")
        return model, device
    else:
        raise FileNotFoundError(f"Trained model not found at {model_path}")

def predict_wind_field(model, device, tif_path):
    """Slices the TIF into patches, runs inference, and builds the vector grid."""
    print(f"Running ML Inference on: {os.path.basename(tif_path)}")
    
    vectors = []
    
    with rasterio.open(tif_path) as src:
        # Define grid for inference (e.g., 8x8 grid across the image)
        grid_size = 8
        step_x = src.width // (grid_size + 1)
        step_y = src.height // (grid_size + 1)
        window_size = 256 # The size the ResNet expects
        half_win = window_size // 2
        
        for i in range(1, grid_size + 1):
            for j in range(1, grid_size + 1):
                x_off = (i * step_x) - half_win
                y_off = (j * step_y) - half_win
                
                # Bounds check
                if x_off < 0 or y_off < 0 or x_off + window_size > src.width or y_off + window_size > src.height:
                    continue

                # Extract the 256x256 image patch
                window = rasterio.windows.Window(x_off, y_off, window_size, window_size)
                img_patch = src.read(1, window=window)
                
                # Extract real-world GPS coordinates for this patch
                transform = src.window_transform(window)
                lon, lat = transform * (half_win, half_win)
                
                # Preprocess patch for the neural network
                img_patch = (img_patch - np.min(img_patch)) / (np.max(img_patch) - np.min(img_patch) + 1e-8)
                
                # Convert numpy array to PyTorch Tensor [Batch=1, Channel=1, H=256, W=256]
                img_tensor = torch.tensor(img_patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                
                # Perform Inference (No gradient calculation needed)
                with torch.no_grad():
                    prediction = model(img_tensor)
                    
                # Extract U and V vectors from the model output
                u_pred, v_pred = prediction[0].cpu().numpy()
                
                # Convert U/V back to Speed and Direction for the plot
                speed = np.sqrt(u_pred**2 + v_pred**2)
                direction = np.arctan2(v_pred, u_pred)
                
                vectors.append({
                    'lon': lon, 'lat': lat,
                    'u': u_pred, 'v': v_pred,
                    'speed': speed, 'dir': direction
                })
                
    return vectors

def plot_ml_predictions(vectors, filename):
    """Draws the final Quiver plot using the ML predictions."""
    lons = np.array([v['lon'] for v in vectors])
    lats = np.array([v['lat'] for v in vectors])
    U = np.array([v['u'] for v in vectors])
    V = np.array([v['v'] for v in vectors])
    speeds = np.array([v['speed'] for v in vectors])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('#f4f6f9')
    
    # Generate the Quiver Plot
    q = ax.quiver(lons, lats, U, V, speeds, cmap='jet', scale=200, width=0.005)
    
    cbar = fig.colorbar(q, ax=ax, orientation='horizontal', pad=0.1)
    cbar.set_label('Predicted Wind Speed (m/s)', fontsize=12, fontweight='bold')
    
    ax.set_title(f'Deep Learning Inference Result: {os.path.basename(filename)}', fontsize=14, pad=20)
    ax.set_xlabel('Longitude (°E)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('ML_Inference_Map.png', dpi=300)
    print("Inference complete! Map saved as 'ML_Inference_Map.png'")
    plt.show()

if __name__ == "__main__":
    # --- Example Deployment ---
    # Path to your saved model weights
    MODEL_WEIGHTS = "../outputs/models/sar_resnet_wind_model.pth"
    
    # Path to a NEW satellite image you want to analyze
    TARGET_FILE = "../data/SAR_TN_Local_Dataset/YOUR_FILE_NAME.tif"
    
    try:
        # 1. Load the trained brain
        trained_model, active_device = load_trained_model(MODEL_WEIGHTS)
        
        # 2. Feed it the satellite image
        predicted_vectors = predict_wind_field(trained_model, active_device, TARGET_FILE)
        
        # 3. Draw the results
        plot_ml_predictions(predicted_vectors, TARGET_FILE)
        
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("Note: You must run train.py first to generate the .pth model file before you can run inference!")