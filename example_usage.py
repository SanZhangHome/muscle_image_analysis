from muscle_image_analysis.processor import MuscleImageProcessor
from muscle_image_analysis.visualizer import MuscleImageVisualizer
from muscle_image_analysis.config import default_config
from muscle_image_analysis.utils import show_segmentation
import os

# 自定义配置
custom_config = default_config.copy()
custom_config['image_path'] = r"D:\Users\87093\Desktop\Images"  # 修改为你的图片路径
custom_config['diameter'] = 150  # 修改为你的肌纤维大小
custom_config['model_type'] = 'cyto'  # 修改为你的分割模型
custom_config['channels'] = ['DAPI', 'TOM70', 'PAX7', 'WGA']  # 修改为你的通道名称
custom_config['dapi_channel_index'] = 0  # DAPI通道索引
custom_config['wga_channel_index'] = 3  # WGA通道索引

# 创建处理器并处理图像
processor = MuscleImageProcessor(custom_config)
results = processor.process_images()

# 显示结果
visualizer = MuscleImageVisualizer(custom_config['image_path'], custom_config['channels'])
visualizer.show_images()

# 显示分割结果
# 假设你有 wga_channel, masks, flows 这些变量
# show_segmentation(wga_channel, masks, flows)