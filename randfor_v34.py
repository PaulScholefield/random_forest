# -*- coding: utf-8 -*-
"""
Geospatial Image Segmentation and Classification Script

Author: Paul Scholefield
Created on: Nov 6, 2023

Description:
This script is designed for advanced geospatial analysis, focusing on the segmentation of raster images, 
feature extraction from these segments, and classification using a Random Forest model. It is particularly 
useful in environmental studies, land cover mapping, and similar fields requiring detailed image analysis.

Key Functionalities:
1. Image Segmentation: Utilizes methods such as 'quickshift', 'slic', and 'watershed' for segmenting multiband 
   raster images.
2. Feature Extraction: Extracts statistical features (mean and standard deviation) from each segment of the 
   raster image.
3. Random Forest Classification: Trains a Random Forest classifier on the extracted features and classifies 
   each segment.
4. Output Generation: Produces various outputs including segmented image files, shapefiles with segmentation 
   data, and classified raster images.

Usage:
1. Ensure all dependencies are installed (geopandas, rasterio, scikit-learn, etc.).
2. Place the raster image and training data shapefile in the designated directories.
3. Run the script in a Python environment. Modify parameters like `raster_path` and `training_data_path` as needed.
4. Review the output files in the specified output directory.

This script requires a basic understanding of Python programming and geospatial analysis concepts.
"""

for name in dir():
    if not name.startswith('_'):
        del globals()[name]


import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.features import shapes
from rasterstats import zonal_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.metrics import confusion_matrix, f1_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from skimage import io, exposure
from skimage.color import label2rgb, rgb2gray
from skimage.segmentation import slic, quickshift, watershed
from skimage.feature import peak_local_max
from scipy import ndimage
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd 
import rasterio.mask
import shapely.geometry
import rasterio


output_directory='H:\P12812_Bridgend_INNS\Segmentation'

def segment_image(method, image_multi, use_multiband_for_quickshift=False):
    if use_multiband_for_quickshift and method == 'quickshift':
        # Use the multiband image directly for quickshift
        return quickshift(image_multi, kernel_size=3, max_dist=6, ratio=0.5, convert2lab=True)
    else:
        # Convert to grayscale for other methods or if not using multiband for quickshift
        image_gray = rgb2gray(image_multi) if image_multi.ndim == 3 else image_multi
        if method == 'slic':
            return slic(image_gray, n_segments=500, compactness=10, channel_axis=None)
        elif method == 'watershed':
            image_eq = exposure.equalize_hist(image_gray)
            distance = ndimage.distance_transform_edt(image_eq)
            local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=image_eq)
            markers = ndimage.label(local_maxi)[0]
            return watershed(-distance, markers, mask=image_eq)
        else:
            raise ValueError("Unknown segmentation method: {}".format(method))

# Load the raster data
raster_path = 'H:\P12812_Bridgend_INNS\Segmentation\Composite_16band_1sqkm.tif'
#raster_path='land_cover.tif'
#raster_path='Composite_raster_1m.tif'

with rasterio.open(raster_path) as src:
    image_multi = src.read()  # multiband data
    # Select three bands for RGB representation (e.g., bands 1, 2, and 3)
    rgb_image = np.dstack((image_multi[0], image_multi[1], image_multi[3]))
# Select segmentation method: 'slic', 'watershed', or 'quickshift'
segmentation_method = 'quickshift'

# Perform segmentation using the selected RGB image
print("Performing image segmentation using {}...".format(segmentation_method))
segments = segment_image(segmentation_method, rgb_image, use_multiband_for_quickshift=True)
print("Segmentation completed.")


# Assuming the first three bands of your multiband data are Red, Green, and Blue
with rasterio.open(raster_path) as src:
    # Read the multiband data
    image_multi = src.read()  # multiband data

# Assuming the first three bands can be combined into an RGB image
rgb_image = np.dstack((image_multi[0], image_multi[1], image_multi[2]))

# Create an RGB image for visualization (optional)
# Use the RGB image for the label2rgb function
segmented_image = label2rgb(segments, rgb_image, kind='avg', bg_label=0)

# Save the segmented image for visualization (optional)
segmented_image_path = os.path.join(output_directory, 'segmented_image.jpg')

io.imsave(segmented_image_path, segmented_image.astype(np.uint8))
print(f"Segmented image saved to {segmented_image_path}")

# Extract shapes and attributes from the segments
with rasterio.open(raster_path) as src:
    transform = src.transform
    results = ({'properties': {'raster_val': v}, 'geometry': s}
               for i, (s, v) in enumerate(shapes(segments.astype(np.int16), mask=None, transform=transform)))

# Convert the shapes and attributes to a GeoDataFrame
geoms = list(results)
gdf = gpd.GeoDataFrame.from_features(geoms)

# Save the GeoDataFrame as a shapefile
segmented_data_path = os.path.join(output_directory, 'segmented_data.shp')
gdf.to_file(segmented_data_path)
print(f"Segmented data saved to {segmented_data_path}")

# Load the training data from shapefile
#training_data = gpd.read_file('land_use_training.shp')
training_data = gpd.read_file('INNS_training.shp')

# Load the raster data
with rasterio.open(raster_path) as src:
    raster_img = src.read()
    raster_meta = src.meta

print("Class distribution in training data:")
print(training_data['Value'].value_counts())


band_count=4
default_nodata = 0
def debug_nan_values(gdf, raster_path, band=1):
    with rasterio.open(raster_path) as src:
        for index, row in gdf.iterrows():
            geom = row.geometry
            if geom.is_empty or not geom.is_valid:
                continue  # Skip empty or invalid geometries

            # Mask the raster with the geometry
            out_image, out_transform = rasterio.mask.mask(src, [geom], crop=True, nodata=default_nodata)
            out_band = out_image[band - 1]  # Remember that Python uses 0-indexing

            # Check if the masked area has any valid data
            if np.all(out_band == default_nodata):
                print(f"All data is nodata for geometry {index} in band {band}")
            else:
                # Plot the valid data for inspection
                plt.imshow(out_band, cmap='gray')
                plt.title(f'Valid data for geometry {index} in band {band}')
                plt.show()

# Call the function for each band
for band in range(1, band_count + 1):
    debug_nan_values(training_data, raster_path, band=band)


def get_nodata_value(raster, default_nodata=-9999):
    """
    Retrieve the nodata value from the raster, or use a default if undefined.
    """
    nodata_value = raster.nodata
    if nodata_value is None:
        print("Raster nodata value is undefined. Using default:", default_nodata)
        nodata_value = default_nodata
    return nodata_value

# Rasterize the training data polygons to create a training raster
#class_values = training_data['Value'].astype(int).values
class_values = training_data['Class'].astype(int).values

shapes = ((geom, value) for geom, value in zip(training_data.geometry, class_values))
training_raster = rasterize(shapes, out_shape=(raster_meta['height'], raster_meta['width']),
                            transform=raster_meta['transform'], fill=0, all_touched=True)

# Function to calculate zonal statistics per feature with added checks and verbose output


import rasterio.mask

def calculate_zonal_stats(gdf, raster_path, band_count, nodata_value=None):
    """
    Calculate zonal statistics and handle NaN values.
    
    Args:
    - gdf: GeoDataFrame containing the geometries for which to calculate zonal statistics.
    - raster_path: Path to the raster file.
    - band_count: Number of bands in the raster file.
    - nodata_value: The nodata value to use if the raster file does not specify one.
    
    Returns:
    - GeoDataFrame with the original data and additional columns for zonal statistics.
    """
    stats_list = ['mean', 'std']
    
    with rasterio.open(raster_path) as src:
        # Use the raster's nodata value if not specified
        if nodata_value is None:
            nodata_value = src.nodata
        print(f"Using nodata value: {nodata_value}")
        
        # Ensure CRS match between the GeoDataFrame and the raster
        if str(src.crs) != str(gdf.crs):
            raise ValueError(f"CRS mismatch between raster and shapefile: {src.crs} != {gdf.crs}")
        
        for index, row in gdf.iterrows():
            geom = row['geometry']
            # Skip the geometry if it is empty or invalid
            if geom.is_empty or not geom.is_valid:
                print(f"Skipping empty or invalid geometry at index {index}.")
                continue

            # Initialize a dictionary to store stats
            geom_stats = {}

            # Calculate zonal statistics for each band
            for band in range(1, band_count + 1):
                stats = zonal_stats(geom, raster_path, stats=stats_list, affine=src.transform, 
                                    nodata=nodata_value, all_touched=True, band_num=band)
                if stats:
                    for stat in stats_list:
                        col_name = f'{stat}_band{band}'
                        stat_val = stats[0].get(f"{stat}", np.nan)
                        if pd.isna(stat_val):
                            print(f"NaN value found for {col_name} in geometry {index}")
                            geom_stats[col_name] = np.nan
                        else:
                            geom_stats[col_name] = stat_val
            
            # Assign stats to GeoDataFrame if any valid stats were found
            if geom_stats and not all(np.isnan(val) for val in geom_stats.values()):
                for key, val in geom_stats.items():
                    gdf.at[index, key] = val
            else:
                print(f"No valid data for geometry {index}, stats will not be added.")
    
    return gdf

# Use the correct nodata value from the raster, or specify the correct one if you know it
nodata_value_from_raster = get_nodata_value(rasterio.open(raster_path))

# Calculate zonal statistics for the training data
print("Calculating zonal statistics for training data...")
training_data_with_stats = calculate_zonal_stats(training_data, raster_path, band_count, nodata_value_from_raster)
print("Zonal statistics added. Verifying DataFrame...")
print(training_data_with_stats.head())


# Prepare the feature columns for training
feature_columns = [f'{stat}_band{band}' for band in range(1, band_count + 1) for stat in ['mean', 'std']]
X = training_data[feature_columns].values
y = training_data['Value'].values

# Remove low variance features
print("Removing low variance features...")
var_thresh = VarianceThreshold(threshold=0.1)  # Threshold is arbitrary; adjust it as needed.
X_var_thresh = var_thresh.fit_transform(X)
feature_columns = np.array(feature_columns)[var_thresh.get_support()]

# Initialize imputer for missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_var_thresh)

# Find the smallest class size
min_class_size = training_data['Value'].value_counts().min()

# SMOTE cannot have more neighbors than the smallest class size minus one
k_neighbors_smote = min_class_size - 1

# Apply SMOTE with the corrected k_neighbors value
print("Applying SMOTE...")
smote = SMOTE(k_neighbors=k_neighbors_smote)
X_smote, y_smote = smote.fit_resample(X_imputed, y)

# Initialize the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Apply RFE for feature selection
print("\nUsing RFE for feature selection...")
selector = RFE(clf, n_features_to_select=10, step=1)
selector = selector.fit(X_smote, y_smote)
X_selected = selector.transform(X_smote)
print(f"Selected features: {feature_columns[selector.support_]}")

# Split the data into training and validation sets using stratified sampling
X_train_selected, X_val_selected, y_train, y_val = train_test_split(
    X_selected, y_smote, test_size=0.2, stratify=y_smote, random_state=42
)

# Normalize the features with StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_val_scaled = scaler.transform(X_val_selected)

# Apply PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)

# Before fitting the model, set up hyperparameter tuning with appropriate n_splits
min_class_size = np.min(np.bincount(y_smote))
n_splits = min(5, min_class_size)  # Use min_class_size or 5, whichever is smaller

print("\nPerforming hyperparameter tuning...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
# Ensure y_train is the target variable post-SMOTE
# Print out the class distribution to check balance
print("Class distribution in y_train after SMOTE:")
print(np.bincount(y_train))

# Find the smallest class size
min_class_size = np.min(np.bincount(y_train))
print("Minimum class size:", min_class_size)

# Ensure n_splits is at least 2 and does not exceed the smallest class size
n_splits = max(min(5, min_class_size), 2)  # Use min_class_size or 5, whichever is smaller, but at least 2
print("n_splits set to:", n_splits)

# Proceed with GridSearchCV using the correct number of splits
grid_search = GridSearchCV(clf, param_grid, cv=StratifiedKFold(n_splits=n_splits), scoring='f1_weighted')
grid_search.fit(X_train_pca, y_train)

print(f"Best parameters: {grid_search.best_params_}")
clf = grid_search.best_estimator_

# Fit the model on the training set
print("Fitting the model on the training set...")
clf.fit(X_train_pca, y_train)

# Predict on the validation set
print("Predicting on the validation set...")
y_pred = clf.predict(X_val_pca)

# Calculate F1 score and print the confusion matrix
val_f1_score = f1_score(y_val, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_val, y_pred)
print(f"Random Forest Validation F1 Score: {val_f1_score}")
print("Random Forest Confusion Matrix:")
print(conf_matrix)

# After making predictions, add more metrics for evaluation
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# If the data is binary, calculate and print ROC AUC score
if len(np.unique(y)) == 2:  # Binary classification check
    roc_auc = roc_auc_score(y_val, clf.predict_proba(X_val_selected)[:, 1])
    print("ROC AUC Score:", roc_auc)

# Plot learning curve
# Find the smallest class size after resampling with SMOTE
min_class_size = np.min(np.bincount(y_train))
n_splits = max(min(5, min_class_size), 2)  # Use min_class_size or 5, whichever is smaller, but at least 2

# Ensure the StratifiedKFold inside the learning_curve uses the adjusted n_splits
train_sizes, train_scores, test_scores = learning_curve(
    clf, X_train_pca, y_train, cv=StratifiedKFold(n_splits=n_splits), scoring='f1_weighted'
)

# The rest of your plotting code remains unchanged
plt.figure()
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores.mean(axis=1), 'o-', color="g", label="Cross-validation score")
plt.title('Learning curve')
plt.xlabel('Training examples')
plt.ylabel('F1 score')
plt.legend(loc="best")
plt.show()

# Calculate F1 score and print the confusion matrix
val_f1_score = f1_score(y_val, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_val, y_pred)
print(f"Random Forest Validation F1 Score: {val_f1_score}")
print("Random Forest Confusion Matrix:")
print(conf_matrix)

# Insert the new code for segment classification here
# Load the segmented data from shapefile (assuming the file name and path)
segmented_data_path = 'segmented_data.shp'
segmented_data = gpd.read_file(segmented_data_path)

# Calculate zonal statistics for the segmented data using the same functions as in training

def add_zonal_stats_to_df(gdf, raster_path, band_count, stats_list=['mean', 'std'], nodata_value=-9999):
    """
    Calculate zonal statistics for each geometry in a GeoDataFrame and add them as new columns.

    Args:
    - gdf: The GeoDataFrame containing the geometries.
    - raster_path: The path to the raster file.
    - band_count: The number of bands in the raster file.
    - stats_list: The list of statistics to calculate.
    - nodata_value: The nodata value for the raster.

    Returns:
    - The GeoDataFrame with added zonal statistics.
    """
    # Loop over each feature (geometry) in the GeoDataFrame
    for index, row in gdf.iterrows():
        geom = row['geometry']
        if geom.is_empty or not geom.is_valid:
            continue  # Skip empty or invalid geometries

        # Calculate zonal statistics for the geometry
        for band in range(1, band_count + 1):
            band_stats = zonal_stats(geom, raster_path, stats=stats_list, affine=None,
                                     nodata=nodata_value, all_touched=True, band_num=band)
            band_stats = band_stats[0]  # We expect one dict per geometry
            
            # Add each statistic as a new column for this geometry
            for stat in stats_list:
                col_name = f'{stat}_band{band}'
                gdf.at[index, col_name] = band_stats.get(f"{stat}", np.nan)
    
    return gdf

# Now you can use the function as intended
segmented_data = add_zonal_stats_to_df(segmented_data, raster_path, band_count)
# Prepare the feature columns for classification
X_segmented = segmented_data[feature_columns].values

# Normalize the features with StandardScaler and apply PCA (trained on training data)
X_segmented_scaled = scaler.transform(X_segmented)  # Use the scaler fitted on training data
X_segmented_pca = pca.transform(X_segmented_scaled)  # Use the PCA fitted on training data

# Predict the classes for the segmented data
segmented_predictions = clf.predict(X_segmented_pca)

# Add the predictions back to the segmented_data GeoDataFrame
segmented_data['predicted_class'] = segmented_predictions

# Save the classified segments to a shapefile
classified_segments_path = os.path.join(output_directory, 'classified_segments.shp')
segmented_data.to_file(classified_segments_path)

print(f"Classified segments saved to {classified_segments_path}")

no_data=default_nodata
# Define the metadata for the output classified raster
output_meta = raster_meta.copy()
output_meta.update({
    'count': 1,
    'dtype': 'uint8',
    'compress': 'lzw',
    'nodata': no_data
})

# Define output file paths
output_classified_raster_path = os.path.join(output_directory, 'classified_land_cover.tif')
prob_raster_path = os.path.join(output_directory, 'class_probabilities.tif')
thresh_raster_path = os.path.join(output_directory, 'thresholded_classified.tif')

# Normalize the features and apply PCA
scaler = StandardScaler()
scaler.fit(training_data[feature_columns].values)  # Fit scaler to training data

# Define stats_list outside the function to be used inside
stats_list = ['mean', 'std']


# Function to predict the land cover for each block in the raster
def predict_land_cover_in_chunks_with_prob(raster_path, clf, scaler, pca, output_meta, output_classified_raster_path, prob_raster_path):
    # Open the raster dataset
    with rasterio.open(raster_path) as src:
        # Initialize empty arrays to hold the classified data and probability data
        classified_data = np.full((src.height, src.width), output_meta['nodata'], dtype=output_meta['dtype'])
        probability_data = np.zeros((src.height, src.width, clf.n_classes_), dtype=np.float32)
        
        # Determine the number of windows
        num_windows = (src.height // src.block_shapes[0][0]) * (src.width // src.block_shapes[0][1])
        
        # Process each block/window with a progress bar
        print("Processing blocks for classification...")
        for ji, window in tqdm(src.block_windows(1), desc="Processing windows", total=num_windows):
            raster_block = src.read(window=window)
            valid_mask = raster_block[0] != no_data
            if raster_block.ndim == 3:
                raster_block = raster_block.squeeze()  # If there's only one band, remove the single band dimension
            
            window_transform = src.window_transform(window)
            window_geom = shapely.geometry.box(*rasterio.windows.bounds(window, src.transform))
            
            # Calculate zonal statistics for the block
            block_stats = zonal_stats(window_geom, raster_path, stats=stats_list, affine=window_transform, nodata=no_data, geojson_out=True)
            
            # Extract the feature properties as a list of lists, each sublist corresponding to one feature
            block_features = []
            for feat in block_stats:
                feature_values = [feat['properties'].get(f"{stat}_{band}", np.nan) for band in range(1, band_count + 1) for stat in stats_list]
                block_features.append(feature_values)
            
            block_features = np.array(block_features, dtype=np.float64)
            if not np.isnan(block_features).all():
                valid_data_scaled = scaler.transform(block_features)
                valid_data_scaled_imputed = imputer.transform(valid_data_scaled)  # Make sure imputer is defined somewhere in the script
                
                valid_data_pca = pca.transform(valid_data_scaled_imputed)
                valid_predictions = clf.predict(valid_data_pca)
                valid_probabilities = clf.predict_proba(valid_data_pca)
                
                classified_data[window.row_off:window.row_off+window.height, window.col_off:window.col_off+window.width][valid_mask] = valid_predictions
                for k in range(clf.n_classes_):
                    probability_data[window.row_off:window.row_off+window.height, window.col_off:window.col_off+window.width, k][valid_mask] = valid_probabilities[:, k]
    
    # Save the classified raster to a file
    with rasterio.open(output_classified_raster_path, 'w', **output_meta) as dst:
        dst.write(classified_data, 1)
    
    # Save the probability raster to a file
    prob_meta = output_meta.copy()
    prob_meta.update({'count': clf.n_classes_, 'dtype': 'float32', 'nodata': None})
    with rasterio.open(prob_raster_path, 'w', **prob_meta) as dst:
        for k in range(clf.n_classes_):
            dst.write(probability_data[:, :, k], k+1)
    
    return classified_data, probability_data


# Call the function to predict and obtain the classified raster and probabilities
classified_image, probability_image = predict_land_cover_in_chunks_with_prob(
    raster_path, clf, scaler, pca, output_meta, output_classified_raster_path, prob_raster_path
)

print(f"Classified raster saved to {output_classified_raster_path}")
print(f"Probability raster saved to {prob_raster_path}")

# Apply a threshold to the maximum probability to create a thresholded image
threshold = 0.75  # 75% probability
max_prob = np.max(probability_image, axis=2)
thresholded_classification = np.where(max_prob > threshold, classified_image, output_meta['nodata'])

# Save the thresholded classified raster to a file
with rasterio.open(thresh_raster_path, 'w', **output_meta) as dst:
    dst.write(thresholded_classification.astype(output_meta['dtype']), 1)

print(f"Thresholded classified raster saved to {thresh_raster_path}")

# Feature space plotting
# Select two bands to plot
xBand = 1  # Assuming bands are 1-indexed in R, subtract 1 for 0-indexed Python
yBand = 2

# Feature space plotting
plt.figure(figsize=(10, 8))
for class_value in np.unique(y_train):
    indices = np.where(y_train == class_value)
    plt.scatter(X_train_pca[indices, 0], X_train_pca[indices, 1], label=f'Class {class_value}', alpha=0.7)

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.title('Feature Space Plot')
plt.show()

