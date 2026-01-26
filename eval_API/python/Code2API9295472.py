from numpy import random
import matplotlib.pyplot as plt

def save_heatmap_without_padding(data_shape=(5,5), output_path="test.png", cmap='hot'):
    data = random.random(data_shape)
    img = plt.imshow(data, interpolation='nearest')
    img.set_cmap(cmap)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()  # Added to clean up the figure
