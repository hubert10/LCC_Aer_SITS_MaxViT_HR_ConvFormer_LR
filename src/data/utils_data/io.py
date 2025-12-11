import os
import zipfile
import rasterio
import numpy as np
from pathlib import Path

try:
    DATA_DIR = os.environ["DATA_DIR"] + "/"
except Exception:
    # DATA_DIR = "D:\kanyamahanga\Datasets"
    DATA_DIR = "/my_data"

# def read_patch(raster_file: str, channels: list = None) -> np.ndarray:   
#     """
#     Reads patch data from a raster file.
#     Args:
#         raster_file (str): Path to the raster file.
#         channels (list, optional): List of channel indices to read. If None, reads all channels.
#     Returns:
#         np.ndarray: The extracted patch data.
#     """
#     with rasterio.open(os.path.join(DATA_DIR, raster_file)) as src_img:
#         array = src_img.read(channels) if channels else src_img.read()
#     return array

def extract_zip_root(path_str: str) -> str:
    p = Path(path_str)
    # We want the two directory levels above the TIFF file
    return str(Path(*p.parts[-4:-2]))


def read_patch(raster_file: str, channels: list = None) -> np.ndarray:
    """
    Reads a TIFF either directly or from inside a ZIP archive.
    """

    # Full path (may not exist on disk)
    full_path = Path(DATA_DIR) / raster_file

    # Extract the TIFF filename
    base_name = full_path.name   # e.g. "D006-2020_AERIAL_RGBI_UN-S1-4_18-5.tif"

    # Identify the ZIP folder name (2 levels above the file)
    zip_folder = extract_zip_root(full_path)    # e.g. "D006-2020_AERIAL_RGBI"

    # Build ZIP path (this is the real existing file)
    zip_path = Path(DATA_DIR) / f"{zip_folder}.zip"

    # Case 1: raw TIFF exists on disk → direct read
    if full_path.exists():
        with rasterio.open(full_path) as src:
            return src.read(channels) if channels else src.read()

    # Case 2: look inside the ZIP
    if not zip_path.exists():
        raise FileNotFoundError(
            f"ZIP archive not found:\n  {zip_path}\nExtracted from path:\n  {full_path}"
        )

    # Search inside ZIP for the TIFF
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.namelist():
            if member.endswith(base_name):        # match file inside ZIP
                vsi_path = f"/vsizip/{str(zip_path).replace(os.sep, '/')}/{member}"
                with rasterio.open(vsi_path) as src:
                    return src.read(channels) if channels else src.read()

    # If not found
    raise FileNotFoundError(
        f"File '{base_name}' not found inside ZIP archive:\n  {zip_path}"
    )
