from PIL import Image

def get_image_size(image_path):
    im = Image.open(image_path)
    return im.size
