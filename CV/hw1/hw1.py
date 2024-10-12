import rawpy
import argparse
import imageio
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

GAMMA = .55

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
                        [.25, 0, .25],
                        [0,.25,0]])


blue_red_kernel = np.array([[.25, 0, .25], 
                             [0,0,0], 
                            [.25, 0, .25]])

red_data_for_green_horinztonal = np.array([[0, .5, 0], [0,0,0],[0,.5,0]])

red_data_for_green_vertical = np.array([[0,0,0], [.5,0,.5],[0,0,0]])


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

green_mean = np.mean(green_output_array)
red_mean = np.mean(red_output_array)
blue_mean = np.mean(blue_output_array)

green_scaling = .25 / green_mean
red_scaling = .25 / red_mean 
blue_scaling = .25 / blue_mean 

green_output_array = green_output_array * green_scaling
red_output_array = red_output_array * red_scaling
blue_output_array = blue_output_array * blue_scaling 

print(f"Green mean: {green_mean} Red_mean: {red_mean} Blue Mean: {blue_mean}")

green_mean_2 = np.mean(green_output_array)
red_mean_2 = np.mean(red_output_array)
blue_mean_2 = np.mean(blue_output_array)
print(f"Green mean: {green_mean_2} Red_mean: {red_mean_2} Blue Mean: {blue_mean_2}")

green_output_array = np.power(green_output_array,  GAMMA)
red_output_array = np.power(red_output_array, GAMMA)
blue_output_array = np.power(blue_output_array, GAMMA)
print(f"Red channel shape: {red_output_array.shape}, max: {np.max(red_output_array)}, min: {np.min(red_output_array)}")
print(f"Green channel shape: {green_output_array.shape}, max: {np.max(green_output_array)}, min: {np.min(green_output_array)}")
print(f"Blue channel shape: {blue_output_array.shape}, max: {np.max(blue_output_array)}, min: {np.min(blue_output_array)}")
plt.figure()
plt.imshow(red_output_array, cmap='gray')
plt.title("Red Channel")
plt.axis('off')
plt.show()

plt.figure()
plt.imshow(green_output_array, cmap='gray')
plt.title("Green Channel")
plt.axis('off')
plt.show()

plt.figure()
plt.imshow(blue_output_array, cmap='gray')
plt.title("Blue Channel")
plt.axis('off')
plt.show()

output_array = np.zeros((height, weight, 3), dtype = float) # 3 because need a spot for rgb
print("output shape", output_array.shape)
output_array[...,0] = red_output_array
output_array[...,1] = green_output_array
output_array[...,2] = blue_output_array
output_array = np.clip(output_array, 0, 1)

output_array = output_array * 255

output_array_final = output_array.astype(np.uint8)
print("output array", output_array)
plt.imshow(output_array_final)
plt.axis('off')
plt.show()

# filename for JPEG output
path_out  = args.path.split('.')[0] +'processed.jpg'

imageio.imwrite(path_out, output_array_final)
