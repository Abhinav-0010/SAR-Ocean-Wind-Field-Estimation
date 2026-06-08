# Ocean Wind Estimation

This project estimates and visualizes ocean wind fields from SAR satellite GeoTIFF images. It includes two workflows:

- A signal-processing workflow that extracts wind-vector patterns from SAR image texture and saves quiver-map PNGs.
- A deep-learning workflow that trains a ResNet18 model to predict wind-vector components from SAR image patches.

The current datasets focus on two coastal regions:

- Gujarat, stored in `data/SAR_Local_Dataset`
- Tamil Nadu, stored in `data/SAR_TN_Local_Dataset`

## Project Structure

```text
.
|-- final_map.py                         # Batch map generator for Gujarat and Tamil Nadu
|-- gujarat_map.py                       # Interactive Gujarat single-file map generator
|-- tn_map.py                            # Interactive Tamil Nadu single-file map generator
|-- src/
|   |-- dataset.py                       # PyTorch dataset loader for SAR images and metadata labels
|   |-- inference.py                     # Runs trained model on a SAR image and plots predictions
|   |-- model.py                         # ResNet18 model adapted for 1-channel SAR inputs
|   `-- train.py                         # Training loop for SAR wind-vector regression
|-- data/
|   |-- training_metadata_2023_2024.csv  # Gujarat wind labels and SAR filenames
|   |-- tn_training_metadata_2023_2024.csv
|   |-- SAR_Local_Dataset/               # Gujarat SAR GeoTIFF files
|   `-- SAR_TN_Local_Dataset/            # Tamil Nadu SAR GeoTIFF files
`-- outputs/
    |-- maps/                            # Generated map PNGs
    `-- models/                          # Trained model weights, created before training
```

## Requirements

Use Python 3.9 or newer. The main Python packages are:

```powershell
pip install numpy pandas matplotlib scipy rasterio torch torchvision
```

On Windows, `rasterio` can sometimes be easier to install through Conda:

```powershell
conda install -c conda-forge rasterio
```

## Quick Start

From the project root, activate your virtual environment if you use one:

```powershell
.\.venv\Scripts\Activate.ps1
```

Generate maps for all available Gujarat and Tamil Nadu SAR files:

```powershell
python final_map.py
```

Generated maps are saved in:

```text
outputs/maps
```

## Single-Region Map Scripts

To choose one SAR file interactively for Gujarat:

```powershell
python gujarat_map.py
```

To choose one SAR file interactively for Tamil Nadu:

```powershell
python tn_map.py
```

Both scripts list the available `.tif` files, ask for a number, then save a wind-map PNG in `outputs/maps`.

## Deep Learning Workflow

The model pipeline is inside `src/`. Run these commands from the `src` folder because the scripts use paths like `../data` and `../outputs`.

```powershell
cd src
New-Item -ItemType Directory -Force ..\outputs\models
python train.py
```

Training uses `data/tn_training_metadata_2023_2024.csv` and the Tamil Nadu SAR dataset by default. It saves model weights to:

```text
outputs/models/sar_resnet_wind_model.pth
```

For inference, first edit `TARGET_FILE` in `src/inference.py` so it points to an existing SAR `.tif` file. Then run:

```powershell
python inference.py
```

The inference script loads the trained model, predicts U/V wind-vector components across an image grid, and saves:

```text
ML_Inference_Map.png
```

## How It Works

The map scripts scan each SAR image in small windows, use FFT texture orientation to estimate wind direction, estimate wind speed from normalized SAR intensity, and plot the resulting vector field with `matplotlib`.

The ML pipeline converts true wind speed and direction from the metadata CSV files into U/V vector targets. `src/model.py` adapts ResNet18 for single-channel SAR image patches and outputs two regression values: U and V.

## Notes

- The SAR `.tif` datasets can be large. For sharing on GitHub, consider keeping large data and generated outputs outside Git, or use a separate data storage method.
- `final_map.py` is the easiest script for generating maps because it processes both regions automatically.
- `train.py` currently trains for 10 epochs with batch size 16 and learning rate 0.001.
- Inference requires a trained `.pth` model file. Run training first or provide existing weights at `outputs/models/sar_resnet_wind_model.pth`.
