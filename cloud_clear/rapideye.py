import json
import rasterio
import numpy as np
from math import radians, sin, cos, tan
from scipy.ndimage import binary_dilation
import os
import re
import matplotlib.pyplot as plt
from .base import CloudClearBase

class RapidEye(CloudClearBase):
    def __init__(self, tmp_dir, output_dir, aoi):
        super().__init__(tmp_dir, output_dir, aoi)
        
        # RapidEye-specific parameters (5m resolution)
        self.nir_band_index = 4  # RapidEye NIR is band 5 (0-indexed as 4)
        self.max_cloud_score = 0.5
        self.nir_dark_threshold = 0.04  # Adjusted for RapidEye's NIR sensitivity
        self.shadow_projection_distance = 100  # pixels (500m at 5m resolution)
        self.cloud_score_threshold = 0.1
        self.min_shadow_length = 20  # pixels (100m minimum shadow length)
        self.cloud_edge_dilation = 2  # pixels to dilate cloud edges

    def _find_metadata_file(self, analytic_path):
        # Extract scene ID from analytic filename
        filename = os.path.basename(analytic_path)
        scene_id = filename.split('_')[0]  # e.g., "6022308"

        # Look for metadata file containing the scene ID
        metadata_dir = os.path.dirname(analytic_path)
        for file in os.listdir(metadata_dir):
            if scene_id in file and file.endswith("metadata.json"):
                return os.path.join(metadata_dir, file)

        raise FileNotFoundError(f"No metadata file found for {analytic_path}")

        
        scene_id, date, satellite = match.groups()
        date_compact = date.replace("-", "")  # "2015-12-19" -> "20151219"
        
        # Search for metadata in the same directory
        dir_path = os.path.dirname(analytic_path)
        for f in os.listdir(dir_path):
            if (f.startswith(date_compact) and
                (scene_id in f) and
                (satellite in f) and
                f.endswith("metadata.json")):
                return os.path.join(dir_path, f)

        
        raise FileNotFoundError(f"No metadata file found for {analytic_path}")

    def _read_metadata(self, metadata_file):
        """Extract solar angles from RapidEye metadata.json"""
        with open(metadata_file) as f:
            metadata = json.load(f)
        props = metadata.get('properties', {})
        return {
            'solar_azimuth': float(props.get('sun_azimuth', 180)),
            'solar_elevation': float(props.get('sun_elevation', 45)),
            'cloud_cover': float(props.get('cloud_cover', 0))
        }

    def _calculate_cloud_score(self, img_data):
        """Calculate cloud probability score for RapidEye"""
        blue = img_data[0, :, :].astype('float32') / 10000.0
        green = img_data[1, :, :] / 10000.0
        red = img_data[2, :, :] / 10000.0
        rededge = img_data[3, :, :] / 10000.0
        nir = img_data[self.nir_band_index, :, :] / 10000.0

        score = np.ones_like(blue)
        blue_score = (np.clip(blue, 0.05, 0.3) - 0.05) / 0.25
        score = np.minimum(score, blue_score)
        visible_score = (np.clip(red + green + blue, 0.1, 0.8) - 0.1) / 0.7
        score = np.minimum(score, visible_score)
        infrared_score = (np.clip(nir + rededge, 0.15, 0.8) - 0.15) / 0.65
        score = np.minimum(score, infrared_score)
        return score

    def _directional_distance_transform(self, mask, angle_deg, max_distance):
        """Efficient shadow projection using Bresenham's line algorithm"""
        angle_rad = radians(angle_deg)
        dx, dy = cos(angle_rad), -sin(angle_rad)  # Negative dy for image coords
        
        h, w = mask.shape
        output = np.zeros_like(mask, dtype=bool)
        y, x = np.where(mask)
        
        for cy, cx in zip(y, x):
            end_x = int(cx + dx * max_distance)
            end_y = int(cy + dy * max_distance)
            
            num_points = max(abs(end_x - cx), abs(end_y - cy)) + 1
            x_line = np.linspace(cx, end_x, num_points).astype(int)
            y_line = np.linspace(cy, end_y, num_points).astype(int)
            
            # Clip to image bounds
            valid = (x_line >= 0) & (x_line < w) & (y_line >= 0) & (y_line < h)
            output[y_line[valid], x_line[valid]] = True
            
        return output

    def _calculate_shadow_mask(self, cloud_mask, nir_band, solar_azimuth, solar_elevation):
        """Optimized cloud shadow detection for RapidEye"""
        # Calculate shadow direction (opposite of sun)
        shadow_dir = (solar_azimuth + 180) % 360
        
        # Dynamic shadow length based on sun elevation
        shadow_length = int(min(
            self.shadow_projection_distance,
            self.shadow_projection_distance * tan(radians(90 - solar_elevation)))
        )
        shadow_length = max(shadow_length, self.min_shadow_length)
        
        # Project from cloud edges only (reduces false positives)
        cloud_edges = binary_dilation(cloud_mask, iterations=self.cloud_edge_dilation) ^ cloud_mask
        projected_shadow = self._directional_distance_transform(
            cloud_edges, shadow_dir, shadow_length
        )
        
        # Confirm with NIR reflectance
        dark_pixels = nir_band < self.nir_dark_threshold
        shadow_mask = projected_shadow & dark_pixels
        
        return shadow_mask

    def apply_udm_mask(self, udm_file, analytic_file):
        """Main processing with automatic metadata discovery"""
        try:
            # 1. Find and load metadata
            metadata_file = self._find_metadata_file(analytic_file)
            metadata = self._read_metadata(metadata_file)
            print(f"Processing with solar azimuth={metadata['solar_azimuth']:.1f}°, elevation={metadata['solar_elevation']:.1f}°")
            
            # 2. Load imagery
            with rasterio.open(analytic_file) as src:
                img_data = src.read()
                meta = src.meta.copy()
                nir_band = img_data[self.nir_band_index, :, :] / 10000.0
            
            # 3. Cloud detection
            with rasterio.open(udm_file) as src:
                udm = src.read(1)
            udm_mask = (udm == 2)
            cloud_score = self._calculate_cloud_score(img_data)
            cloud_mask = udm_mask | (cloud_score > self.cloud_score_threshold)
            
            # 4. Shadow detection
            shadow_mask = self._calculate_shadow_mask(
                cloud_mask, nir_band, 
                metadata['solar_azimuth'], 
                metadata['solar_elevation']
            )
            
            # 5. Apply masks
            final_mask = cloud_mask | shadow_mask
            clear_mask = np.where(final_mask, 0, 1)
            masked_data = (img_data / 10000.0) * clear_mask[np.newaxis, :, :]
            
            # 6. Export results
            output_path = os.path.join(
                self.output_dir,
                os.path.basename(analytic_file).replace('.tif', '_masked.tif')
            )
            meta.update({'dtype': 'float32'})
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(masked_data.astype('float32'))
            
            return output_path
            
        except Exception as e:
            print(f"Error processing {analytic_file}: {str(e)}")
            raise
