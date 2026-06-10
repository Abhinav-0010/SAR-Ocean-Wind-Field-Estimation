import io
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from scipy.fftpack import fft2, fftshift


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIRS = {
    "tn": {
        "label": "Tamil Nadu",
        "folder": os.path.join(BASE_DIR, "data", "SAR_TN_Local_Dataset"),
    },
    "gujarat": {
        "label": "Gujarat",
        "folder": os.path.join(BASE_DIR, "data", "SAR_Local_Dataset"),
    },
}

app = Flask(__name__, static_folder="Webpage", static_url_path="/webpage")


def _region_config(region_key):
    config = DATA_DIRS.get(region_key)
    if config is None:
        abort(400, description="Unknown region")
    return config


def _available_files(region_key):
    config = _region_config(region_key)
    folder = config["folder"]
    if not os.path.isdir(folder):
        return []
    files = [name for name in os.listdir(folder) if name.lower().endswith(".tif")]
    return sorted(files)


def _safe_tif_path(region_key, filename):
    config = _region_config(region_key)
    safe_name = os.path.basename(filename)
    folder = config["folder"]
    candidate = os.path.join(folder, safe_name)
    if not os.path.isfile(candidate):
        abort(404, description="SAR file not found")
    return candidate, safe_name, config["label"]


def _render_wind_figure(tif_path, region_label, filename):
    vectors = []

    with rasterio.open(tif_path) as src:
        img_w, img_h = src.width, src.height
        full_img = src.read(1)
        global_max = np.nanmax(full_img)

        if global_max == 0 or np.isnan(global_max):
            raise ValueError("SAR file is empty")

        window_size = min(64, img_w // 3, img_h // 3)
        half_win = window_size // 2
        grid_size = 10
        step_x = img_w // grid_size
        step_y = img_h // grid_size

        for i in range(grid_size):
            for j in range(grid_size):
                cx = (i * step_x) + (step_x // 2)
                cy = (j * step_y) + (step_y // 2)
                x_off = cx - half_win
                y_off = cy - half_win

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

                cy_mag, cx_mag = mag.shape[0] // 2, mag.shape[1] // 2
                mag[cy_mag - 3:cy_mag + 3, cx_mag - 3:cx_mag + 3] = 0

                y, x = np.unravel_index(np.argmax(mag), mag.shape)
                angle_rad = np.arctan2(y - cy_mag, x - cx_mag)
                wind_dir = (np.degrees(angle_rad) + 90) % 360

                mean_intensity = np.nanmean(img_data)
                estimated_speed = 3.0 + (mean_intensity / global_max) * 9.0

                vectors.append(
                    {
                        "lon": lon,
                        "lat": lat,
                        "speed": estimated_speed,
                        "dir": np.radians(wind_dir),
                    }
                )

    if not vectors:
        raise ValueError("No valid ocean features found")

    lons = np.array([v["lon"] for v in vectors])
    lats = np.array([v["lat"] for v in vectors])
    speeds = np.array([v["speed"] for v in vectors])
    dirs = np.array([v["dir"] for v in vectors])
    u = speeds * np.cos(dirs)
    v = speeds * np.sin(dirs)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("#f4f6f9")
    quiver = ax.quiver(lons, lats, u, v, speeds, cmap="jet", scale=150, width=0.005)
    colorbar = fig.colorbar(quiver, ax=ax, orientation="horizontal", pad=0.1)
    colorbar.set_label("Wind Speed (m/s)", fontsize=12, fontweight="bold")
    ax.set_title(f"SAR Ocean Wind Field ({region_label}) - {filename}", fontsize=12, pad=20)
    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    return fig


@app.route("/")
def index():
    return send_from_directory(os.path.join(BASE_DIR, "Webpage"), "index.html")


@app.route("/api/regions")
def regions():
    return jsonify(
        {
            "regions": [
                {"key": key, "label": value["label"]}
                for key, value in DATA_DIRS.items()
            ]
        }
    )


@app.route("/api/files")
def files():
    region_key = request.args.get("region", "tn")
    _region_config(region_key)
    return jsonify({"region": region_key, "files": _available_files(region_key)})


@app.route("/api/render")
def render():
    region_key = request.args.get("region", "tn")
    filename = request.args.get("file", "")
    tif_path, safe_name, region_label = _safe_tif_path(region_key, filename)

    try:
        fig = _render_wind_figure(tif_path, region_label, safe_name)
    except Exception as exc:
        abort(422, description=str(exc))

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300)
    plt.close(fig)
    buffer.seek(0)
    return send_file(buffer, mimetype="image/png", download_name=f"{safe_name}.png")


if __name__ == "__main__":
    app.run(debug=True)