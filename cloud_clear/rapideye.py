from .base import CloudClearBase
import rasterio
import numpy as np
from scipy.ndimage import binary_dilation
import os

class RapidEye(CloudClearBase):
    def _calculate_cloud_score(self, img_data):
        """
        Custom cloud scoring for RapidEye bands [B, G, R, RedEdge, NIR]
        Returns score between 0 (cloudy) and 1 (clear)
        """
        # Extract and scale bands to reflectance (0-1)
        blue = img_data[0, :, :].astype('float32') / 10000.0
        green = img_data[1, :, :] / 10000.0
        red = img_data[2, :, :] / 10000.0
        rededge = img_data[3, :, :] / 10000.0
        nir = img_data[4, :, :] / 10000.0

        # Initialize score (1 = clear, lower = more cloudy)
        score = np.ones_like(blue)

        # Rule 1: Brightness in Blue Band
        blue_score = (np.clip(blue, 0.05, 0.3) - 0.05) / 0.25
        score = np.minimum(score, blue_score)

        # Rule 2: Brightness in Visible (R + G + B)
        visible_score = (np.clip(red + green + blue, 0.1, 0.8) - 0.1) / 0.7
        score = np.minimum(score, visible_score)

        # Rule 3: Brightness in NIR + RedEdge
        infrared_score = (np.clip(nir + rededge, 0.15, 0.8) - 0.15) / 0.65
        score = np.minimum(score, infrared_score)

        return score

    def apply_udm_mask(self, udm_file, analytic_file):
        """
        Enhanced masking that combines UDM and custom cloud detection
        1. Applies UDM mask first with 3-pixel buffer
        2. Applies custom cloud detection to remaining areas
        3. Combines both masks
        """
        with rasterio.open(analytic_file) as src_analytic, \
             rasterio.open(udm_file) as src_udm:

            # Read data
            img_data = src_analytic.read()
            udm = src_udm.read(1)
            meta = src_analytic.meta.copy()

            # 1. Apply UDM mask first (primary)
            udm_mask = (udm == 2)  # UDM cloud/shadow pixels
            buffered_udm = binary_dilation(udm_mask, iterations=3)
            
            # 2. Calculate custom cloud score ONLY for UDM-cleared areas
            clear_pixels = ~buffered_udm  # Areas UDM didn't mask
            cloud_score = self._calculate_cloud_score(img_data)
            
            # Only consider custom score in clear areas
            constrained_score = np.where(clear_pixels, cloud_score, 0)
            custom_mask = (constrained_score > 0.1)  # Conservative threshold
            buffered_custom = binary_dilation(custom_mask, iterations=2)

            # 3. Combine masks (UDM always takes priority)
            final_mask = np.where(buffered_udm | buffered_custom, 0, 1)

            # Apply mask and save
            masked_data = (img_data / 10000.0) * final_mask[np.newaxis, :, :]
            output_path = os.path.join(
                self.output_dir,
                os.path.basename(analytic_file).replace('.tif', '_cleaned.tif')
            )
            
            meta.update({'dtype': 'float32'})
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(masked_data.astype('float32'))

        return output_path
