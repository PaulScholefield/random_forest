# random_forest
A short script to take a multiband raster and a set of training data, and classify the raster using segmentation

# Geospatial Image Segmentation and Classification Script

## Installation Instructions

### Prerequisites
- Python 3.x

### Steps
1. **Clone the Repository**:
2. **Create and Activate Virtual Environment**:

cd path/to/your-project
python -m venv venv

Windows
.\venv\Scripts\activate

macOS and Linux
source venv/bin/activate

3. **Install Dependencies**:
python -m pip install --upgrade pip
pip install numpy geopandas rasterio rasterstats scikit-learn scikit-image imbalanced-learn tqdm matplotlib


### Notes
- Ensure you have a multiband raster image (`.tif`) and a training data shapefile (`.shp`).
- Modify `raster_path` and `training_data_path` in the script according to your file paths.
- Familiarity with Python and geospatial data processing is recommended for script customization.

## Troubleshooting
For issues during installation or execution, refer to the documentation of the respective libraries or Python.
