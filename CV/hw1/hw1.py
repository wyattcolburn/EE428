import rawpy
import argparse
import numpy as np
from scipy.ndimage import correlate
import matplotlib 
matplotlib.use('gtk3agg') # use this for my vim setup
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
# Parse command line arguments.
# Usage:
#    python hw1.py <DNG filename>
parser = argparse.ArgumentParser()
parser.add_argument('path')
args = parser.parse_args()


# read 16-bit integer data from RAW file into raw_image array

with rawpy.imread(args.path) as raw:
    raw_image = raw.raw_image.copy()
    print(raw_image)
    image_array = np.array(raw_image) #uint16_t, 2^16 -1 to normalize
    print(type(image_array[0][0])) 
    image_array = image_array.astype('float32') / 65535 

print(image_array)
print("shape", image_array.shape)


"""
Bayer Pattern is 

RGRG
GBGB
RGRG
GBGB
"""
height, weight = image_array.shape
red_array = np.zeros((height,weight), dtype = float)
green_array = np.zeros((height,weight), dtype = float)
blue_array = np.zeros((height,weight), dtype = float)
red_avg_array = np.zeros((height,weight), dtype = float)
green_avg_array = np.zeros((height,weight), dtype = float)
blue_avg_array = np.zeros((height,weight), dtype = float)

green_kernel = np.array([[0, .25, 0], 
                        [.25, 0, 0],
                        [0,.25,0]])


blue_red_kernel = np.array([[.25, 0, .25], 
                             [0,0,0], 
                            [.25, 0, .25]])


red_array[0::2,0::2]= image_array[0::2, 0::2]
green_array[1::2, 1::2] =  image_array[1::2, 1::2]
blue_array[1::2, 1::2] = image_array[1::2, 1::2] 


green_avg_array = correlate(image_array, green_kernel)
red_avg_array = correlate(image_array, blue_red_kernel)
blue_avg_array = correlate(image_array, blue_red_kernel)

print(red_avg_array)

green_output_array = green_array + green_avg_array
red_output_array = red_array + red_avg_array
blue_output_array = blue_array + blue_avg_array

output_array = np.zeros((height, weight, 3), dtype = float) # 3 because need a spot for rgb
print("output shape", output_array.shape)
output_array[...,0] = red_output_array
output_array[...,1] = green_output_array
output_array[...,2] = blue_output_array

plt.imshow(output_array)
plt.axis('off')
plt.show()

# filename for JPEG output
path_out  = args.path.split('.')[0] +'jpg'
