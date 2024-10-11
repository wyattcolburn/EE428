import rawpy
import argparse
import numpy as np
from scipy.ndimage import correlate


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


red_array[0::2,0::2]= raw_image[0::2, 0::2]
green_array[1::2, 1::2] =  raw_image[1::2, 1::2]
blue_array[1::2, 1::2] = raw_image[1::2, 1::2] 


green_avg_array = correlate(image_array, green_kernel)

print(red_avg_array)

green_output_array = green_array + green_avg_array


# filename for JPEG output
path_out  = args.path.split('.')[0] +'jpg'
