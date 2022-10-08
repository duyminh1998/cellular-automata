from PIL import Image
import numpy as np

im = Image.open("../img/sample_CA_init_config_300x300.bmp")
p = np.array(im)
p_scaled = 1 * (p) / (255)
print(p)