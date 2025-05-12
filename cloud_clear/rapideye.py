from .base import CloudClearBase
import rasterio
import numpy as np
from scipy.ndimage import binary_dilation
import os
import json
import math
from pathlib import Path

class RapidEye(CloudClearBase):
    def __init__(self, tmp_dir, output_dir, aoi):
        super().__init__(tmp_dir, output_dir, aoi)
        self.use_custom_cloud_score = True
        self.dark_threshold = 0.05  # 5% reflectance threshold
        self.shadow_search_distance = 50  # pixels (adjust based on resolution)
        self.shadow_search_angle_width = 15  # degrees
        self.max_cloud_pixels = 1000  # for performance

    def _find_metadata_file(self, analytic_file):
        """Find the most appropriate metadata JSON file."""
        base_path = Path(analytic_file).parent
        base_stem = Path(analytic_file).stem.split('_')[0]  # Get date portion
        
        # Try different naming patterns in priority order
        patterns = [
            f"{base_stem}*RapidEye*.json",  # Matches your example
            f"{base_stem}*metadata.json",
            "*metadata*.json"
        ]
        
        for pattern in patterns:
            matches = list(base_path.glob(pattern))
            if matches:
                return str(matches[0])  # Return first match
        return None

    def _get_solar_azimuth(self, analytic_file):
        """Extract and validate solar azimuth."""
        metadata_file = self._find_metadata_file(analytic_file)
        if not metadata_file:
            print(f"Warning: No metadata file found for {analytic_file}")
            return None

        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
                azimuth = float(metadata['sun_azimuth'])
                if 0 <= azimuth <= 360:
                    return azimuth
                print(f"Warning: Invalid azimuth {azimuth} in {metadata_file}")
        except (KeyError, json.JSONDecodeError, ValueError) as e:
            print(f"Error reading {metadata_file}: {str(e)}")
        return None

    def _mask_cloud_shadows(self, scaled_data, cloud_mask, solar_azimuth):
        """Mask only shadows cast by clouds using solar geometry."""
        if solar_azimuth is None:
            print("Using fallback dark pixel masking")
            return self._mask_dark_pixels(scaled_data)

        # Calculate dark areas
        mean_brightness = np.mean(scaled_data, axis=0)
        dark_pixels = mean_brightness < self.dark_threshold
        
        if not np.any(cloud_mask) or not np.any(dark_pixels):
            return np.ones_like(dark_pixels, dtype=bool)

        height, width = dark_pixels.shape
        shadow_direction = math.radians(solar_azimuth + 180)  # Opposite of sun
        
        # Calculate expected shadow displacement
        dx = int(math.cos(shadow_direction) * self.shadow_search_distance)
        dy = int(math.sin(shadow_direction) * self.shadow_search_distance)
        
        shadow_mask = np.zeros_like(dark_pixels, dtype=bool)
        cloud_coords = np.argwhere(cloud_mask)[:self.max_cloud_pixels]

        for cy, cx in cloud_coords:
            # Calculate expected shadow location
            sx, sy = cx + dx, cy + dy
            
            # Define search area
            x_min = max(0, sx - 10)
            x_max = min(width, sx + 10)
            y_min = max(0, sy - 10)
            y_max = min(height, sy + 10)
            
            # Mark dark pixels in this area
            if x_min < x_max and y_min < y_max:
                shadow_area = dark_pixels[y_min:y_max, x_min:x_max]
                shadow_mask[y_min:y_max, x_min:x_max] |= shadow_area

        return ~binary_dilation(shadow_mask, iterations=2)

    def _mask_dark_pixels(self, scaled_data):
        """Fallback: mask all dark pixels."""
        mean_brightness = np.mean(scaled_data, axis=0)
        dark_mask = mean_brightness < self.dark_threshold
        return ~binary_dilation(dark_mask, iterations=2)

    def _calculate_cloud_score(self, scaled_data):
        """Calculate cloud probability scores."""
        blue, green, red, nir, rededge = scaled_data[:5]
        score = np.ones_like(blue)
        
        blue_score = (np.clip(blue, 0.05, 0.3) - 0.05) / 0.25
        visible_score = (np.clip(red + green + blue, 0.1, 0.8) - 0.1) / 0.7
        nir_score = (np.clip(nir + rededge, 0.15, 0.8) - 0.15) / 0.65
        
        return np.minimum.reduce([score, blue_score, visible_score, nir_score])

    def apply_udm_mask(self, udm_file, analytic_file):
        """Main processing method."""
        with rasterio.open(analytic_file) as src:
            scaled_data = src.read().astype('float32') / 10000.0
            meta = src.meta.copy()

        if self.use_custom_cloud_score:
            cloud_score = self._calculate_cloud_score(scaled_data)
            cloud_mask = binary_dilation(cloud_score > 0.05, iterations=3)
            solar_azimuth = self._get_solar_azimuth(analytic_file)
            shadow_mask = self._mask_cloud_shadows(scaled_data, cloud_mask, solar_azimuth)
            final_mask = np.logical_and(~cloud_mask, shadow_mask).astype('float32')
        else:
            with rasterio.open(udm_file) as src:
                final_mask = np.where(binary_dilation(src.read(1) == 2, iterations=3), 0, 1)

        # Apply mask and save
        output_file = os.path.join(self.output_dir, os.path.basename(analytic_file).replace('.tif', '_cleaned.tif'))
        meta.update({'dtype': 'float32'})
        
        with rasterio.open(output_file, 'w', **meta) as dst:
            dst.write((scaled_data * final_mask[np.newaxis, :, :]).astype('float32'))
        
        print(f"Saved processed file to: {output_file}")
        return output_file
