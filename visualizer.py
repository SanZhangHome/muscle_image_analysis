import os
import tifffile
import matplotlib.pyplot as plt

class MuscleImageVisualizer:
    def __init__(self, image_path, channels):
        self.image_path = image_path
        self.channels = channels
    
    def show_images(self):
        files = [f for f in os.listdir(self.image_path) if f.endswith('.tif')]
        
        for i in range(len(files)):
            file = files[i]
            img = tifffile.imread(os.path.join(self.image_path, file))
            
            plt.figure(figsize=(20, 5))
            plt.suptitle(file, y=1.05)
            
            for ch in range(len(self.channels)):
                plt.subplot(1, len(self.channels), ch+1)
                plt.imshow(img[ch], cmap='gray')
                plt.title(self.channels[ch])
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()