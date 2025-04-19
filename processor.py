import os
import tifffile
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from skimage.exposure import rescale_intensity
from cellpose import models

class MuscleImageProcessor:
    def __init__(self, config):
        self.config = config
        self.result_dir = os.path.join(self.config['image_path'], 'results')
        self.each_image_dir = os.path.join(self.result_dir, 'EachImage')
        os.makedirs(self.each_image_dir, exist_ok=True)
        self.model = models.Cellpose(gpu=self.config['use_gpu'], model_type=self.config['model_type'])
    
    def process_images(self):
        files = [f for f in os.listdir(self.config['image_path']) if f.endswith('.tif')]
        all_stats_df = pd.DataFrame()
        
        for idx, file in enumerate(files, start=1):
            img = tifffile.imread(os.path.join(self.config['image_path'], file))
            wga_channel = img[self.config['wga_channel_index']]
            wga_channel = rescale_intensity(wga_channel, in_range='image', out_range='dtype')
            
            eval_result = self.model.eval(
                wga_channel,
                diameter=self.config['diameter'],
                flow_threshold=self.config['flow_threshold'],
                channels=[0, 0]
            )
            
            masks = self._extract_masks(eval_result)
            masks = self._remove_edge_masks(masks)
            
            mask_filename = os.path.splitext(file)[0] + '_mask.npy'
            np.save(os.path.join(self.each_image_dir, mask_filename), masks)
            
            props = regionprops_table(
                masks, 
                properties=('area', 'perimeter', 'major_axis_length', 'minor_axis_length', 'eccentricity')
            )
            
            props['circularity'] = 4 * np.pi * props['area'] / (props['perimeter'] ** 2)
            props['aspect_ratio'] = props['major_axis_length'] / props['minor_axis_length']
            
            temp_df = pd.DataFrame(props)
            if temp_df.empty:
                print(f"{idx}/{len(files)} - {file}: 未检测到纤维")
                continue
                
            self._analyze_other_channels(img, masks, temp_df)
            
            dapi_img = img[self.config['dapi_channel_index']]
            dapi_mean = np.mean(dapi_img)
            dapi_max = np.max(dapi_img)
            dapi_cv = np.std(dapi_img) / dapi_mean * 100 if dapi_mean != 0 else np.nan
            
            csv_name = os.path.splitext(file)[0] + '.csv'
            csv_path = os.path.join(self.each_image_dir, csv_name)
            temp_df.to_csv(csv_path, index=False)
            
            record = {
                'image_name': file,
                'total_fibers': len(temp_df),
                'DAPI_image_mean_brightness': dapi_mean,
                'DAPI_image_max_brightness': dapi_max,
                'DAPI_image_cv': dapi_cv
            }
            
            for col in temp_df.select_dtypes(include=np.number).columns:
                mean = temp_df[col].mean()
                std = temp_df[col].std()
                record[f'{col}_mean'] = mean
                record[f'{col}_cv'] = (std / mean) * 100 if mean != 0 else np.nan
            
            all_stats_df = pd.concat([all_stats_df, pd.DataFrame([record])], ignore_index=True)
        
        all_stats_df.insert(0, 'Number', range(1, len(all_stats_df) + 1))
        output_path = os.path.join(self.result_dir, 'all_parameters.csv')
        all_stats_df.to_csv(output_path, index=False)
        
        return all_stats_df
    
    def _extract_masks(self, eval_result):
        if len(eval_result) == 4:
            return eval_result[0]
        elif len(eval_result) == 3:
            return eval_result[0]
        elif len(eval_result) == 2:
            return eval_result[0]
        else:
            raise ValueError("Unexpected model output format")
    
    def _remove_edge_masks(self, masks):
        height, width = masks.shape
        edge_threshold = 5
        
        for mask_id in np.unique(masks):
            if mask_id == 0:
                continue
                
            y_coords, x_coords = np.where(masks == mask_id)
            
            if (np.any(y_coords < edge_threshold) or np.any(y_coords >= height - edge_threshold) or
                np.any(x_coords < edge_threshold) or np.any(x_coords >= width - edge_threshold)):
                masks[masks == mask_id] = 0
                
        return masks
    
    def _analyze_other_channels(self, img, masks, df):
        for mask_id in np.unique(masks):
            if mask_id == 0:
                continue
                
            y_coords, x_coords = np.where(masks == mask_id)
            
            for channel_idx, channel_name in enumerate(self.config['channels']):
                if channel_idx == self.config['dapi_channel_index'] or channel_idx == self.config['wga_channel_index']:
                    continue
                    
                channel_img = img[channel_idx]
                mask_pixels = channel_img[y_coords, x_coords]
                
                mean = np.mean(mask_pixels)
                max_val = np.max(mask_pixels)
                cv = np.std(mask_pixels) / mean * 100 if mean != 0 else np.nan
                
                df.loc[df.index == mask_id - 1, f'{channel_name}_mean_brightness'] = mean
                df.loc[df.index == mask_id - 1, f'{channel_name}_max_brightness'] = max_val
                df.loc[df.index == mask_id - 1, f'{channel_name}_cv'] = cv