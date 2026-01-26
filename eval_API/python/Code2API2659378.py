from PIL import Image
import numpy as np

def create_and_save_image(width=512, height=512, color=[255,0,0], save_path='my.png', show_image=True):
    # Create array filled with zeros
    data = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Set color patch in upper left
    data[0:height//2, 0:width//2] = color
    
    # Create image from array
    img = Image.fromarray(data, 'RGB')
    
    # Save image
    img.save(save_path)
    
    # Show image if requested
    if show_image:
        img.show()
    
    return img
