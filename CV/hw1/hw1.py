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
def main():

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
    green_nearby_array = np.zeros((height,weight), dtype = float)
    blue_array = np.zeros((height,weight), dtype = float)
    red_avg_array = np.zeros((height,weight), dtype = float)
    green_avg_array = np.zeros((height,weight), dtype = float)
    blue_avg_array = np.zeros((height,weight), dtype = float)

    green_kernel = np.array([[1, .5],[.5,0]]) 


    blue_red_kernel = np.array([[.25, .5, .25], 
                                 [.5,1, .5], 
                                [.25, .5, .25]])



    red_array[0::2,0::2]= image_array[0::2, 0::2]
    green_array[0::2, 1::2] =  image_array[0::2, 1::2]
    green_array[1::2, 0::2] = image_array[1::2, 0::2]
    blue_array[1::2, 1::2] = image_array[1::2, 1::2] 

    # Absolute no idea what I am doing with the averaging was trying to get the colors better
    green_array = correlate(green_array, green_kernel)
    blue_array= correlate(blue_array, blue_red_kernel)
    red_array = correlate(red_array, blue_red_kernel)

    output_array = np.stack((red_array, green_array,  blue_array), axis =2)

    output_array[...,0] = (.25 / np.mean(output_array[...,0])) * output_array[...,0]
    output_array[...,1] = (.25 / np.mean(output_array[...,1])) * output_array[...,1]
    output_array[...,2] = (.25 / np.mean(output_array[...,2])) * output_array[...,2]

    output_array = output_array** GAMMA

    output_array = np.clip(output_array, 0, 1)

    output_array = output_array * 255

    output_array_final = output_array.astype(np.uint8)
    
    plt.imshow(output_array_final)
    plt.axis('off')
    plt.show()

# filename for JPEG output
    path_out  = args.path.split('.')[0] +'processed.jpg'

    imageio.imwrite(path_out, output_array_final)

if __name__ == "__main__":
    main()

