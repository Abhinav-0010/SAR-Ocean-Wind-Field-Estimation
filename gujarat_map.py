import numpy as np
import rasterio
from scipy.fftpack import fft2, fftshift
import matplotlib.pyplot as plt
import os
import glob

def process_and_save_sar(tif_path, region):
    filename = os.path.basename(tif_path)
    print(f"\n[+] Extracting data from: {filename}...")

    try:
        vectors = []
        with rasterio.open(tif_path) as src:
            img_w, img_h = src.width, src.height
            full_img = src.read(1)
            global_max = np.nanmax(full_img)
            
            if global_max == 0 or np.isnan(global_max):
                print("[-] ERROR: TIF file is completely empty.")
                return
                
            window_size = min(64, img_w // 3, img_h // 3) 
            half_win = window_size // 2
            grid_size = 10 
            step_x = img_w // grid_size
            step_y = img_h // grid_size
            
            for i in range(grid_size):
                for j in range(grid_size):
                    cx = (i * step_x) + (step_x // 2)
                    cy = (j * step_y) + (step_y // 2)
                    x_off, y_off = cx - half_win, cy - half_win
                    
                    if x_off < 0 or y_off < 0 or x_off + window_size > img_w or y_off + window_size > img_h:
                        continue

                    window = rasterio.windows.Window(x_off, y_off, window_size, window_size)
                    img_data = src.read(1, window=window)
                    
                    if np.nanmax(img_data) == 0 or np.all(np.isnan(img_data)):
                        continue
                    
                    transform = src.window_transform(window)
                    lon, lat = transform * (half_win, half_win)
                    
                    img_data = np.nan_to_num(img_data, 0) 
                    f = fft2(img_data)
                    fshift = fftshift(f)
                    mag = 20 * np.log(np.abs(fshift) + 1)
                    
                    cy_mag, cx_mag = mag.shape[0]//2, mag.shape[1]//2
                    mag[cy_mag-3:cy_mag+3, cx_mag-3:cx_mag+3] = 0
                    
                    y, x = np.unravel_index(np.argmax(mag), mag.shape)
                    angle_rad = np.arctan2(y - cy_mag, x - cx_mag)
                    wind_dir = (np.degrees(angle_rad) + 90) % 360 
                    
                    mean_intensity = np.nanmean(img_data)
                    estimated_speed = 3.0 + (mean_intensity / global_max) * 9.0 
                    
                    vectors.append({'lon': lon, 'lat': lat, 'speed': estimated_speed, 'dir': np.radians(wind_dir)})

        if not vectors:
             print("[-] ERROR: No valid ocean features found.")
             return

        print(f"[+] Extracted {len(vectors)} wind vectors. Rendering map...")

        lons, lats = np.array([v['lon'] for v in vectors]), np.array([v['lat'] for v in vectors])
        speeds, dirs = np.array([v['speed'] for v in vectors]), np.array([v['dir'] for v in vectors])
        U, V = speeds * np.cos(dirs), speeds * np.sin(dirs)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_facecolor('#f4f6f9') 
        q = ax.quiver(lons, lats, U, V, speeds, cmap='jet', scale=150, width=0.005)
        cbar = fig.colorbar(q, ax=ax, orientation='horizontal', pad=0.1)
        cbar.set_label('Wind Speed (m/s)', fontsize=12, fontweight='bold')
        ax.set_title(f'SAR Ocean Wind Field ({region}) - {filename}', fontsize=12, pad=20)
        ax.set_xlabel('Longitude (°E)', fontsize=12)
        ax.set_ylabel('Latitude (°N)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save to outputs
        output_dir = os.path.join("outputs", "maps")
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'{region.replace(" ", "_")}_{filename.replace(".tif", "")}.png')
        plt.savefig(save_path, dpi=300)
        print(f"[+] Success! Map saved to: {save_path}")
        
        # Display the interactive window
        plt.show() 

    except Exception as e:
        print(f"[-] ERROR: {e}")

# ... (Paste the exact same process_and_save_sar function here) ...

if __name__ == "__main__":
    region = "Gujarat"
    folder = "data/SAR_Local_Dataset"  # Your Gujarat folder
    
    print("=========================================")
    print(f"    {region} Wind Map Generator")
    print("=========================================")
    
    files = glob.glob(os.path.join(folder, "*.tif"))
    if not files:
        print(f"No .tif files found in {folder}!")
    else:
        for i, f in enumerate(files):
            print(f" [{i+1}] {os.path.basename(f)}")
            
        try:
            choice = int(input("\nEnter the number of the file to process: "))
            if 1 <= choice <= len(files):
                process_and_save_sar(files[choice-1], region)
            else:
                print("Invalid selection.")
        except ValueError:
            print("Please enter a valid number.")